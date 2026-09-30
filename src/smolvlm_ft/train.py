"""Training entry point: fine-tune SmolVLM2 on DCSASS with MLflow tracking.

Flow:
    1. Load parquet rows (built by ``build-data``).
    2. Build processor + LoRA model.
    3. Run ``SFTTrainer`` with the video-aware collator (fixed batch of 1).
    4. Merge LoRA into the base weights and save a standalone model.

Everything is logged to MLflow (params, metrics, checkpoint artifacts).
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import TrainerCallback

from .collator import VideoCollatorConfig, VideoSFTCollator
from .config import Settings
from .data import build_parquets
from .lora import build_lora_model, load_processor, resolve_device

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stdout,
    )
    for name in ("datasets", "huggingface_hub", "urllib3", "mlflow", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


class CheckpointArtifactCallback(TrainerCallback):
    """Log each saved checkpoint as an MLflow artifact under ``artifacts/checkpoints``."""

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if ckpt_dir.exists():
                import mlflow

                mlflow.log_artifacts(str(ckpt_dir), artifact_path="artifacts/checkpoints")
        return control


class BestEvalCheckpointCallback(TrainerCallback):
    """Save the first checkpoint at ``first_checkpoint_step``, then only on new best eval loss.

    Replaces the default periodic ``save_steps`` cadence: with it, checkpoints
    after the first one would land on arbitrary step multiples even when the
    model got worse. Requires ``save_strategy="no"`` so saves only ever happen
    through ``control.should_save`` set here.
    """

    def __init__(self, first_checkpoint_step: int) -> None:
        self.first_checkpoint_step = first_checkpoint_step
        self.best_eval_loss: float | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or state.global_step < self.first_checkpoint_step:
            return control
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return control
        if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            control.should_save = True
        return control



def _collect_params(settings: Settings, device: torch.device, train_rows: int, val_rows: int) -> dict[str, object]:
    """Hyper-parameters worth tracking that the HF Trainer does NOT log itself.

    With ``report_to=["mlflow"]`` the Trainer already logs the standard
    TrainingArguments (learning_rate, warmup_ratio, weight_decay, max_steps,
    gradient_accumulation_steps, bf16, ...). Logging those here too would trip
    MLflow's "params already logged" guard, so we only add the project-specific
    ones.
    """
    return {
        "model_name": settings.model_name,
        "device": str(device),
        "attn_implementation": settings.attn_implementation,
        "lora_r": settings.lora_r,
        "lora_alpha": settings.lora_alpha,
        "lora_dropout": settings.lora_dropout,
        "lora_targets": ",".join(settings.lora_targets),
        "max_seq_length": settings.max_seq_length,
        "num_frames": settings.num_frames,
        "max_image_size": settings.max_image_size,
        "train_rows": train_rows,
        "val_rows": val_rows,
    }


def _gpu_info() -> dict[str, str]:
    """Best-effort GPU description for the MLflow run tag."""
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return {"gpu": out.replace("\n", "; ")}
    except Exception:
        return {"gpu": "unknown"}


def _configure_mlflow_tracking(settings: Settings) -> None:
    """Make the process honour the configured tracking URI and private CA.

    * ``MLFLOW_TRACKING_URI`` so callbacks that read the env var (e.g. the
      HF ``MLflowCallback`` behind ``report_to=["mlflow"]``) agree with
      ``mlflow.set_tracking_uri``.
    * ``MLFLOW_TRACKING_SERVER_CERT_PATH`` so requests verify the Caddy
      homelab root instead of the system store. MLflow re-reads it per
      request (it feeds ``MlflowHostCreds.verify``), but it must exist
      before the first API call. An externally exported value always wins
      (setdefault).
    """
    os.environ.setdefault("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)
    if not settings.mlflow_tracking_uri.startswith("https://"):
        return
    cert = settings.mlflow_server_cert_path
    if cert is None:
        return
    if cert.is_file():
        os.environ.setdefault("MLFLOW_TRACKING_SERVER_CERT_PATH", str(cert.resolve()))
        logger.info("MLflow TLS: trusting CA bundle %s", cert.resolve())
    else:
        logger.warning(
            "MLflow TLS: CA bundle %r is not a file; certificate verification may fail for %s",
            cert,
            settings.mlflow_tracking_uri,
        )


def _resolve_video_path(example: dict, dcsass_root: Path) -> dict:
    """Rewrite the parquet-relative video path to an absolute one for this machine."""
    abs_path = str((dcsass_root / example["video_path"]).resolve())
    example["video_path"] = abs_path
    for item in example["messages"][0]["content"]:
        if item.get("type") == "video":
            item["path"] = abs_path
    return example


def run_training(settings: Settings) -> int:
    """Execute the full fine-tuning job. Returns a process exit code."""
    settings.check_data_files()
    # Restrict to a single GPU: with >1 CUDA device visible, HF Trainer wraps
    # the model in nn.DataParallel, which conflicts with pinning the model to
    # one device and breaks gradient checkpointing. Must be set before any
    # CUDA call initializes the device list. Keeps the first device from an
    # existing CUDA_VISIBLE_DEVICES (e.g. shell-exported "0,1") since this
    # trainer only ever targets a single device.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = visible or "0"
    _configure_mlflow_tracking(settings)
    device = resolve_device(settings)
    logger.info("Training on %s | torch %s | %s", device, torch.__version__, platform.platform())

    # ------------------------------------------------------------------ data
    data = load_dataset("parquet", data_files={
        "train": str(settings.train_parquet),
        "validation": str(settings.val_parquet),
    })
    data = data.map(lambda ex: _resolve_video_path(ex, settings.dcsass_root))
    logger.info("Loaded train=%d val=%d rows", len(data["train"]), len(data["validation"]))

    # ----------------------------------------------------------------- model
    processor = load_processor(settings)
    model = build_lora_model(settings, device)

    # ---------------------------------------------------------------- trainer
    import mlflow
    from trl import SFTConfig, SFTTrainer

    settings.run_output_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = settings.bf16 and device.type == "cuda"
    args = SFTConfig(
        output_dir=str(settings.run_output_dir),
        per_device_train_batch_size=settings.per_device_batch_size,
        per_device_eval_batch_size=settings.per_device_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        max_steps=settings.max_steps,
        num_train_epochs=settings.num_train_epochs,
        learning_rate=settings.learning_rate,
        warmup_ratio=settings.warmup_ratio,
        lr_scheduler_type=settings.lr_scheduler_type,
        weight_decay=settings.weight_decay,
        max_grad_norm=settings.max_grad_norm,
        logging_steps=settings.logging_steps,
        # Saving is driven entirely by BestEvalCheckpointCallback, not a fixed step cadence.
        save_strategy="no",
        save_total_limit=settings.save_total_limit,
        eval_strategy="steps",
        eval_steps=settings.eval_steps,
        bf16=use_bf16,
        fp16=settings.fp16 and device.type == "cuda",
        gradient_checkpointing=settings.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=settings.dataloader_num_workers,
        seed=settings.seed,
        report_to=["mlflow"],
        run_name=settings.run_name,
        remove_unused_columns=False,
        logging_dir=str(settings.run_output_dir / "logs"),
        # trl-specific: keep raw rows (the collator does all processing)
        max_length=settings.max_seq_length,
        shuffle_dataset=True,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    collator = VideoSFTCollator(
        processor=processor,
        config=VideoCollatorConfig(
            max_seq_length=settings.max_seq_length,
            num_frames=settings.num_frames,
            max_image_size=settings.max_image_size,
        ),
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=processor,
        data_collator=collator,
        callbacks=[
            BestEvalCheckpointCallback(settings.first_checkpoint_step),
            CheckpointArtifactCallback(),
        ],
    )

    # ----------------------------------------------------------------- mlflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    params = _collect_params(settings, device, len(data["train"]), len(data["validation"]))

    started = time.time()
    with mlflow.start_run(run_name=settings.run_name) as run:
        mlflow.log_params(params)
        mlflow.set_tags({
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            **_gpu_info(),
        })
        try:
            train_out = trainer.train()
        except torch.cuda.OutOfMemoryError as exc:
            logger.error("CUDA OOM: %s. Lower num_frames/max_image_size and retry.", exc)
            mlflow.set_tag("status", "oom")
            return 3

        # ---------------------------------------------------------------- save
        trainer.save_model(str(settings.run_output_dir))
        processor.save_pretrained(str(settings.run_output_dir))

        if settings.save_merged_model:
            logger.info("Merging LoRA into base weights -> %s", settings.merged_output_dir)
            merged = model.merge_and_unload()
            merged.save_pretrained(str(settings.merged_output_dir), safe_serialization=True)
            processor.save_pretrained(str(settings.merged_output_dir))

        # --------------------------------------------------------- mlflow tail
        elapsed = time.time() - started
        train_metrics = train_out.metrics or {}
        mlflow.log_metrics({
            "train/epoch": train_metrics.get("epoch", settings.num_train_epochs),
            "train/total_time_sec": round(elapsed, 1),
            "train/steps_per_sec": round(max(1, trainer.state.global_step) / max(1.0, elapsed), 4),
        })
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metrics({f"train/{key}": value})
        mlflow.log_artifacts(str(settings.run_output_dir), artifact_path="artifacts/final_adapter")
        mlflow.set_tag("status", "ok")
        logger.info("Done in %.1fs. Run: %s", elapsed, run.info.run_id)

    return 0


def main_cli(argv: list[str] | None = None) -> int:
    """CLI dispatcher: ``build-data`` or ``train``."""
    import argparse

    parser = argparse.ArgumentParser(prog="smolvlm-ft", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-data", help="Scan raw DCSASS and write train/val parquet")
    sub.add_parser("train", help="Fine-tune the model and log to MLflow")
    args = parser.parse_args(argv)

    _setup_logging()
    settings = Settings()

    if args.command == "build-data":
        build_parquets(settings)
        return 0
    if args.command == "train":
        return run_training(settings)
    return 2


if __name__ == "__main__":
    raise SystemExit(main_cli())
