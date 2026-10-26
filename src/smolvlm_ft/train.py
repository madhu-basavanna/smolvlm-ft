"""Training entry point: fine-tune SmolVLM2 on DCSASS with MLflow tracking.

Flow:
    1. Load parquet rows (built by ``build-data`` or ``build-clip-data``).
    2. Build processor + LoRA model.
    3. Run ``SFTTrainer`` with the video-aware collator (fixed batch of 1).
    4. Merge LoRA into the base weights and save a standalone model.

Everything is logged to MLflow (params, metrics, checkpoint artifacts).

This module is also the CLI: ``build-data``, ``build-clips``, ``caption-clips``,
``build-clip-data``, ``train`` and ``evaluate``. The two tasks are selected with
``--task``: ``yesno`` is the original per-clip Yes/No dataset, ``clips`` is the
concatenated incident dataset with the VLM captions and derived event timings.
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

from .collator import ClipSamplingConfig, VideoCollatorConfig, VideoSFTCollator
from .config import Device, Settings
from .data import build_parquets, load_clip_dataset, resolve_video_path
from .lora import build_lora_model, load_processor, resolve_device
from .mlflow_observability import configure_mlflow_tracking, enable_openai_autolog

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

    When there is no validation split at all (a small captioned set can round to
    zero validation incidents), there is no eval loss to select on, so the
    fallback is a single checkpoint at ``first_checkpoint_step``. Otherwise such a
    run would train and save nothing.
    """

    def __init__(self, first_checkpoint_step: int, has_eval: bool = True) -> None:
        self.first_checkpoint_step = first_checkpoint_step
        self.has_eval = has_eval
        self.saved_fallback = False
        self.best_eval_loss: float | None = None

    def on_step_end(self, args, state, control, **kwargs):
        if not self.has_eval and not self.saved_fallback and state.global_step >= self.first_checkpoint_step:
            self.saved_fallback = True
            control.should_save = True
        return control

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




def _collect_params(
    settings: Settings,
    device: torch.device,
    train_rows: int,
    val_rows: int,
    *,
    task: str = "yesno",
    run_name: str | None = None,
    num_frames: int | None = None,
    max_seq_length: int | None = None,
) -> dict[str, object]:
    """Hyper-parameters worth tracking that the HF Trainer does NOT log itself.

    With ``report_to=["mlflow"]`` the Trainer already logs the standard
    TrainingArguments (learning_rate, warmup_ratio, weight_decay, max_steps,
    gradient_accumulation_steps, bf16, ...). Logging those here too would trip
    MLflow's "params already logged" guard, so we only add the project-specific
    ones. The frame count and token budget are passed in rather than read from
    Settings because the clip task uses its own (larger) values.
    """
    return {
        "task": task,
        "model_name": settings.model_name,
        "device": str(device),
        "attn_implementation": settings.attn_implementation,
        "lora_r": settings.lora_r,
        "lora_alpha": settings.lora_alpha,
        "lora_dropout": settings.lora_dropout,
        "lora_targets": ",".join(settings.lora_targets),
        "max_seq_length": max_seq_length or settings.max_seq_length,
        "num_frames": num_frames or settings.num_frames,
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
    """Point MLflow at the configured tracking URI and trust the private CA.

    Thin alias for :func:`smolvlm_ft.mlflow_observability.configure_mlflow_tracking`,
    kept because ``evaluate.py`` imports this name. The implementation lives in
    one place so the data-building, captioning and training commands cannot
    disagree about the tracking URI or the CA bundle.
    """
    configure_mlflow_tracking(settings)


def run_training(settings: Settings, task: str = "yesno") -> int:
    """Execute the full fine-tuning job. Returns a process exit code.

    ``task`` selects the dataset and its token budget; everything else (LoRA,
    optimizer, callbacks, MLflow wiring) is identical for both tasks:

    * ``yesno`` — the original per-clip Yes/No dataset.
    * ``clips`` — concatenated incident videos with a VLM caption and the
      label-derived event span, which needs more frames per sample and therefore
      a larger token budget (``clip_max_seq_length``) and event-window sampling.
    """
    if task not in ("yesno", "clips"):
        raise ValueError(f"Unknown task {task!r}; expected 'yesno' or 'clips'")
    clip_task = task == "clips"

    run_name = settings.clip_run_name if clip_task else settings.run_name
    run_output_dir = settings.clip_run_output_dir if clip_task else settings.run_output_dir
    merged_output_dir = settings.clip_merged_output_dir if clip_task else settings.merged_output_dir
    num_frames = settings.clip_num_frames if clip_task else settings.num_frames
    max_seq_length = settings.clip_max_seq_length if clip_task else settings.max_seq_length

    if clip_task:
        # A clearer message than check_data_files()'s "run build-data first".
        # Only the train split is required: build_clip_parquets deliberately
        # writes no validation file when the split rounds to zero incidents.
        if not settings.train_clips_parquet.is_file():
            raise FileNotFoundError(
                f"Missing clip dataset file: {settings.train_clips_parquet}. Run `smolvlm-ft build-clips`, "
                "`caption-clips` and `build-clip-data` first."
            )
    else:
        settings.check_data_files()
    # Hide CUDA completely when CPU was requested. Must happen before any CUDA
    # call initializes the device list: even with use_cpu=True and the model on
    # CPU, torch's optimizer probes the current accelerator on the first step
    # (`torch.accelerator.current_stream()`), which initializes a primary CUDA
    # context. On a box whose GPUs are already full that fails with
    # `CUDA error: out of memory` and kills a CPU-only run. An empty
    # CUDA_VISIBLE_DEVICES makes the accelerator unavailable instead, so the probe
    # is skipped.
    if settings.device is Device.CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # Restrict to a single GPU: with >1 CUDA device visible, HF Trainer wraps
        # the model in nn.DataParallel, which conflicts with pinning the model to
        # one device and breaks gradient checkpointing. Keeps the first device from
        # an existing CUDA_VISIBLE_DEVICES (e.g. shell-exported "0,1") since this
        # trainer only ever targets a single device.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = visible or "0"

    _configure_mlflow_tracking(settings)
    device = resolve_device(settings)
    logger.info("Training task=%s on %s | torch %s | %s", task, device, torch.__version__, platform.platform())

    # ------------------------------------------------------------------ data
    if clip_task:
        data = {
            "train": load_clip_dataset(settings, "train"),
            "validation": load_clip_dataset(settings, "validation"),
        }
    else:
        data = load_dataset("parquet", data_files={
            "train": str(settings.train_parquet),
            "validation": str(settings.val_parquet),
        })
        data = data.map(lambda ex: resolve_video_path(ex, settings.dcsass_root))
    logger.info("Loaded train=%d val=%d rows", len(data["train"]), len(data["validation"]))
    has_eval = len(data["validation"]) > 0
    if not has_eval:
        logger.warning(
            "Validation split is empty: evaluation is disabled and the first checkpoint will be "
            "saved at step %d with no metric. Caption more incidents, or raise "
            "SMOLVLM_FT_CLIP_VAL_FRAC, to get a validation split.",
            settings.first_checkpoint_step,
        )

    # ----------------------------------------------------------------- model
    processor = load_processor(settings, num_frames=num_frames)
    model = build_lora_model(settings, device)

    # ---------------------------------------------------------------- trainer
    import mlflow
    from trl import SFTConfig, SFTTrainer

    run_output_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = settings.bf16 and device.type == "cuda"
    args = SFTConfig(
        output_dir=str(run_output_dir),
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
        # An empty validation split (see build_clip_parquets) has nothing to
        # evaluate, so evaluation is switched off rather than left to fail.
        eval_strategy="steps" if has_eval else "no",
        eval_steps=settings.eval_steps,
        bf16=use_bf16,
        fp16=settings.fp16 and device.type == "cuda",
        # TrainingArguments initializes CUDA unless told not to: on a CPU run it
        # still calls torch.cuda.set_device(), which fails outright when another
        # process already holds all the GPU memory. The device was resolved from
        # settings above, so state it explicitly instead of letting it probe.
        use_cpu=device.type == "cpu",
        gradient_checkpointing=settings.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=settings.dataloader_num_workers,
        seed=settings.seed,
        report_to=["mlflow"],
        run_name=run_name,
        remove_unused_columns=False,
        logging_dir=str(run_output_dir / "logs"),
        # trl-specific: keep raw rows (the collator does all processing)
        max_length=max_seq_length,
        shuffle_dataset=True,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    collator = VideoSFTCollator(
        processor=processor,
        config=VideoCollatorConfig(
            max_seq_length=max_seq_length,
            num_frames=num_frames,
            max_image_size=settings.max_image_size,
            clip_sampling=ClipSamplingConfig() if clip_task else None,
            ground_timestamps=clip_task,
        ),
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"] if has_eval else None,
        processing_class=processor,
        data_collator=collator,
        callbacks=[
            BestEvalCheckpointCallback(settings.first_checkpoint_step, has_eval=has_eval),
            CheckpointArtifactCallback(),
        ],
    )

    # ----------------------------------------------------------------- mlflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    params = _collect_params(
        settings, device, len(data["train"]), len(data["validation"]),
        task=task, run_name=run_name, num_frames=num_frames, max_seq_length=max_seq_length,
    )

    started = time.time()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.set_tags({
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "task": task,
            **_gpu_info(),
        })
        try:
            train_out = trainer.train()
        except torch.cuda.OutOfMemoryError as exc:
            logger.error("CUDA OOM: %s. Lower num_frames/max_image_size and retry.", exc)
            mlflow.set_tag("status", "oom")
            return 3

        # ---------------------------------------------------------------- save
        trainer.save_model(str(run_output_dir))
        processor.save_pretrained(str(run_output_dir))

        if settings.save_merged_model:
            logger.info("Merging LoRA into base weights -> %s", merged_output_dir)
            merged = model.merge_and_unload()
            merged.save_pretrained(str(merged_output_dir), safe_serialization=True)
            processor.save_pretrained(str(merged_output_dir))

            # "Models from code": MLflow loads mlflow_model.py itself instead of
            # cloudpickling a live Python object, which is the safer, supported
            # pattern for custom pyfunc models.
            mlflow.pyfunc.log_model(
                name="model",
                python_model=str(Path(__file__).parent / "mlflow_model.py"),
                artifacts={"model_path": str(merged_output_dir)},
                pip_requirements=["transformers", "torch", "decord==0.6.0", "pillow", "numpy"],
            )

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
        mlflow.log_artifacts(str(run_output_dir), artifact_path="artifacts/final_adapter")
        mlflow.set_tag("status", "ok")
        logger.info("Done in %.1fs. Run: %s", elapsed, run.info.run_id)

    return 0


def main_cli(argv: list[str] | None = None) -> int:
    """CLI dispatcher for every command in the project.

    Datasets:
        build-data      raw DCSASS clips -> train/val parquet (Yes/No task)
        build-clips     incidents -> one concatenated video per incident + timing
        caption-clips   ask the gateway VLM for a one-line caption + event span
        build-clip-data captions + labels -> train/val parquet (caption/timing task)

    Models:
        train      --task yesno|clips
        evaluate   --task yesno|clips
    """
    import argparse

    parser = argparse.ArgumentParser(prog="smolvlm-ft", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-data", help="Scan raw DCSASS and write train/val parquet")

    build_clips = sub.add_parser(
        "build-clips", help="Index incidents, derive event timestamps, concatenate each incident into one video"
    )
    build_clips.add_argument("--limit", type=int, default=None, help="Only process the first N incidents (pilot run)")
    build_clips.add_argument("--overwrite", action="store_true", help="Rewrite existing incident videos")
    build_clips.add_argument("--report", action="store_true", help="Only write qc_report.md, do not concat anything")

    caption = sub.add_parser("caption-clips", help="Caption every incident video through the MLflow gateway VLM")
    caption.add_argument("--limit", type=int, default=None, help="Only caption the first N incidents (pilot run)")
    caption.add_argument("--workers", type=int, default=None, help="Concurrent requests (default SMOLVLM_FT_VLM_WORKERS)")
    caption.add_argument(
        "--mode", choices=["contact_sheet", "keyframes"], default=None,
        help="How keyframes are sent. contact_sheet (default) is the only layout this gateway "
             "accepts above 2 images; keyframes needs --keyframes<=2",
    )
    caption.add_argument("--keyframes", type=int, default=None, help="Keyframes sampled per incident")
    caption.add_argument("--overwrite", action="store_true", help="Ignore the caption cache and re-ask the model")
    caption.add_argument("--dry-run", action="store_true", help="Print the request/token estimate and exit")
    caption.add_argument(
        "--no-autolog", action="store_true",
        help="Do not enable mlflow.autolog (for offline experiments; the calls are then not traced)",
    )

    sub.add_parser("build-clip-data", help="Build the caption/timing train/val parquet from the manifest")

    train_parser = sub.add_parser("train", help="Fine-tune the model and log to MLflow")
    train_parser.add_argument("--task", choices=["yesno", "clips"], default="yesno", help="Which dataset to fine-tune on")

    eval_parser = sub.add_parser("evaluate", help="Compare base vs fine-tuned model on the val split")
    eval_parser.add_argument("--max-samples", type=int, default=None, help="Cap the number of val rows evaluated")
    eval_parser.add_argument("--task", choices=["yesno", "clips"], default="yesno", help="Which task to evaluate")

    args = parser.parse_args(argv)

    _setup_logging()
    settings = Settings()

    if args.command == "build-data":
        build_parquets(settings)
        return 0
    if args.command == "build-clips":
        from .build_dataset import run_build_clips

        run_build_clips(settings, limit=args.limit, overwrite=args.overwrite, report_only=args.report)
        return 0
    if args.command == "caption-clips":
        from .build_dataset import run_caption_clips
        from .config import CaptionMode

        if args.mode:
            settings.caption_mode = CaptionMode(args.mode)
        if args.keyframes:
            settings.caption_keyframes = args.keyframes

        run_caption_clips(

            settings, limit=args.limit, workers=args.workers, overwrite=args.overwrite,
            dry_run=args.dry_run, autolog=not args.no_autolog,
        )
        return 0
    if args.command == "build-clip-data":
        from .build_dataset import run_build_clip_data

        run_build_clip_data(settings)
        from .data import build_clip_parquets

        build_clip_parquets(settings)
        return 0
    if args.command == "train":
        # Autolog is enabled before any openai import so the gateway calls a
        # caption-judge evaluation might make are traced too; transformers
        # autologging is excluded because this function logs the run itself
        # (report_to=["mlflow"] + mlflow.pyfunc.log_model).
        enable_openai_autolog(settings, exclude_transformers=True)
        return run_training(settings, task=args.task)
    if args.command == "evaluate":
        from .evaluate import run_evaluation

        return run_evaluation(settings, max_samples=args.max_samples, task=args.task)
    return 2



if __name__ == "__main__":
    raise SystemExit(main_cli())
