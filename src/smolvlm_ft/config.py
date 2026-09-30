"""Runtime configuration.

All knobs are declared as Pydantic fields. Values are resolved from (in
order of precedence): environment variables with the ``SMOLVLM_FT_`` prefix,
a local ``.env`` file, then the defaults below. Keeping the settings object
as the single source of truth means the data builder, the training script
and the CLI all agree on paths and hyper-parameters without argument
passing.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Project root: .../masters_project/smolvlm
PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: Default root of the extracted Kaggle dataset.
DCSASS_ROOT = PROJECT_ROOT / "data" / "dcsass dataset" / "DCSASS Dataset"


class Device(str, Enum):
    """Where the model runs. ``auto`` picks CUDA when available, else CPU."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class Settings(BaseSettings):
    """Central, overridable configuration for data building and training."""

    model_config = SettingsConfigDict(
        env_prefix="SMOLVLM_FT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ data
    dcsass_root: Path = Field(default=DCSASS_ROOT, description="Extracted DCSASS dataset root")
    dataset_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "dataset",
        description="Where train/val parquet files are written and read from",
    )
    val_frac: float = Field(default=0.1, ge=0.0, lt=1.0, description="Fraction of rows held out for validation")
    seed: int = Field(default=42, description="Deterministic seed for all splits and training")

    # ----------------------------------------------------------------- model
    model_name: str = Field(
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        description="HF model id (or local path) to fine-tune",
    )
    device: Device = Field(default=Device.AUTO, description="Device to train on")
    attn_implementation: str = Field(
        default="sdpa",
        description="Attention backend: sdpa is the safe default; flash_attention_2 is faster on newer GPUs",
    )
    lora_r: int = Field(default=16, ge=1, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA scaling alpha")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="LoRA dropout")
    lora_targets: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="Linear layer names to attach LoRA adapters to",
    )

    # -------------------------------------------------------------- training
    per_device_batch_size: int = Field(default=1, ge=1, description="Batch size per device (keep 1 on ~3GB free VRAM)")
    gradient_accumulation_steps: int = Field(default=4, ge=1, description="Gradients to accumulate before an optimizer step")
    max_steps: int = Field(default=200, ge=1, description="Total optimizer steps")
    learning_rate: float = Field(default=2e-5, gt=0.0, description="Peak learning rate")
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=1.0, description="Fraction of steps for LR warmup")
    lr_scheduler_type: str = Field(default="cosine", description="HF Trainer scheduler name")
    weight_decay: float = Field(default=0.01, ge=0.0, description="AdamW weight decay")
    max_grad_norm: float = Field(default=1.0, gt=0.0, description="Gradient clipping norm")
    max_seq_length: int = Field(default=512, ge=64, description="Token budget per sample; longer samples are dropped")
    num_frames: int = Field(default=4, ge=1, description="Frames sampled per clip (clips are 1-5s)")
    max_image_size: int = Field(default=512, ge=56, description="Frame side (px). Must be the model native 512 for this architecture; the merger requires exactly 64 tokens/frame")
    logging_steps: int = Field(default=5, ge=1, description="Log a metric line every N steps")
    eval_steps: int = Field(default=50, ge=1, description="Run validation every N steps")
    first_checkpoint_step: int = Field(
        default=100, ge=1,
        description="Step of the first unconditional checkpoint; every save after that requires a new best eval loss",
    )
    num_train_epochs: int = Field(default=1, ge=1, description="Fallback epochs field (max_steps wins)")
    bf16: bool = Field(default=True, description="Use bfloat16 (needs CUDA; CPU test runs use fp32)")
    fp16: bool = Field(default=False, description="Use float16")
    gradient_checkpointing: bool = Field(default=True, description="Trade compute for memory via activation checkpointing")
    dataloader_num_workers: int = Field(default=0, ge=0, description="DataLoader worker processes (0 = decode in main process)")
    save_total_limit: int = Field(default=2, ge=0, description="Keep at most this many checkpoints (0 = all)")

    # ---------------------------------------------------------------- output
    output_dir: Path = Field(default=PROJECT_ROOT / "outputs", description="Root for checkpoints and the final merge")
    run_name: str = Field(default="smolvlm2-dcsass-lora", description="MLflow run name")
    save_merged_model: bool = Field(default=True, description="Merge LoRA into the base weights and save a standalone model")

    # ---------------------------------------------------------------- mlflow
    mlflow_tracking_uri: str = Field(default="https://mlflow.home", description="MLflow tracking server URI")
    mlflow_experiment_name: str = Field(default="smolvlm2-dcsass", description="MLflow experiment name")
    mlflow_server_cert_path: Path | None = Field(
        default=PROJECT_ROOT / "caddy-root-homelab.crt",
        description=(
            "CA bundle for https tracking URIs signed by a private CA (Caddy homelab root). "
            "Exported as MLFLOW_TRACKING_SERVER_CERT_PATH; null keeps the system trust store"
        ),
    )

    # ------------------------------------------------------------- derived
    @property
    def train_parquet(self) -> Path:
        return self.dataset_dir / "train.parquet"

    @property
    def val_parquet(self) -> Path:
        return self.dataset_dir / "val.parquet"

    @property
    def run_output_dir(self) -> Path:
        return self.output_dir / self.run_name

    @property
    def merged_output_dir(self) -> Path:
        return self.output_dir / f"{self.run_name}-merged"

    def check_data_files(self) -> None:
        """Fail fast with an actionable message if the dataset has not been built."""
        missing = [p for p in (self.train_parquet, self.val_parquet) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing dataset file(s): "
                + ", ".join(str(p) for p in missing)
                + ". Run `smolvlm-ft build-data` first."
            )
