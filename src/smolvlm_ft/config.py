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

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Project root: .../masters_project/smolvlm
PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: Default root of the extracted Kaggle dataset.
DCSASS_ROOT = PROJECT_ROOT / "data" / "dcsass dataset" / "DCSASS Dataset"
#: OpenAI-compatible route served by the MLflow server itself, derived from the
#: tracking URI (the same rule MLflow uses internally in
#: ``mlflow.assistant.providers.mlflow_gateway``).
GATEWAY_PATH = "gateway/mlflow/v1"


class Device(str, Enum):
    """Where the model runs. ``auto`` picks CUDA when available, else CPU."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class CaptionMode(str, Enum):
    """How keyframes are handed to the captioning VLM.

    ``keyframes`` sends every keyframe as its own ``image_url`` part: the most
    faithful layout, but the measured gateway tolerates only 1-2 images before it
    starts answering 500. ``contact_sheet`` composites the keyframes into ONE
    labelled image, which the same endpoint accepts at 8-16 tiles. Default is the
    layout that actually works here; the other one stays selectable.
    """

    KEYFRAMES = "keyframes"
    CONTACT_SHEET = "contact_sheet"


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
    per_device_batch_size: int = Field(default=1, ge=1, description="Batch size per device")
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

    # ------------------------------------------------- VLM (MLflow AI gateway)
    vlm_base_url: str = Field(
        default="",
        description=(
            "OpenAI-compatible base URL used for caption generation. Empty means the "
            "tracking server's own gateway: {mlflow_tracking_uri}/gateway/mlflow/v1"
        ),
    )
    vlm_model: str = Field(
        default="qwen3.8-flash-next-local",
        description=(
            "Name passed as the OpenAI ``model`` parameter. Through the MLflow gateway this "
            "must be a configured *endpoint name* (see GET {base_url}/models), not an upstream model id"
        ),
    )
    vlm_api_key: str = Field(
        default="",
        description=(
            "Bearer token for the chat endpoint. Empty sends no auth: the MLflow gateway is "
            "authenticated server-side. Kept configurable so a token can be added without code changes"
        ),
    )
    vlm_timeout: float = Field(
        default=240.0, gt=0.0,
        description=(
            "Per-request timeout (s) for the chat endpoint. A 12-tile contact sheet measured 20-165s "
            "on this gateway, so anything under ~200s mostly produces timeouts"
        ),
    )
    vlm_max_retries: int = Field(default=2, ge=0, description="SDK-level retries for 429/5xx/connection errors")
    caption_parse_attempts: int = Field(
        default=2, ge=1,

        description=(
            "How many times to ask the endpoint for one incident. The SDK handles transport "
            "errors; this covers replies we cannot parse as JSON. A failed incident still gets "
            "a manifest row (caption_ok=false) rather than failing the whole job"
        ),
    )
    vlm_workers: int = Field(default=4, ge=1, description="Concurrent caption requests")
    caption_experiment_name: str = Field(
        default="smolvlm2-dcsass-caption",
        description="MLflow experiment that receives the autologged caption traces",
    )
    caption_mode: CaptionMode = Field(
        default=CaptionMode.CONTACT_SHEET,
        description=(
            "contact_sheet = one labelled tile-grid image per incident (works on this gateway at "
            "8-16 tiles); keyframes = one image_url per keyframe (only 1-2 images are tolerated by "
            "the measured endpoint before it answers 500)"
        ),
    )
    caption_keyframes: int = Field(default=12, ge=1, description="Keyframes sampled per incident when captioning")
    caption_sheet_columns: int = Field(default=4, ge=1, description="Tiles per row in a contact sheet")
    caption_tile_size: int = Field(
        default=256, ge=64,
        description="Tile edge (px) inside a contact sheet. 256 keeps a 12-tile sheet under ~1.1k prompt tokens",
    )
    caption_image_size: int = Field(
        default=512, ge=56,
        description="Longest edge (px) of each decoded keyframe before it is tiled or sent on its own",
    )
    caption_image_detail: str = Field(
        default="low",
        description="OpenAI image_url 'detail' hint: low keeps keyframe vision tokens cheap; auto/high lets the model look closer",
    )
    caption_max_tokens: int = Field(
        default=700, ge=16,
        description=(
            "Max tokens generated per caption. This endpoint treats it as a hard cap: 400 truncated a "
            "12-tile request with finish_reason=length"
        ),
    )
    caption_temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature for captioning")
    caption_disable_thinking: bool = Field(
        default=True,
        description=(
            "Send chat_template_kwargs={'enable_thinking': false}. Required on this endpoint: with "
            "thinking on, reasoning eats max_tokens and content comes back null"
        ),
    )
    caption_use_response_format: bool = Field(
        default=False,
        description=(
            "Send response_format={'type':'json_object'}. Measured on this gateway it is unreliable "
            "(HTTP 502 'structured output was incomplete', or a 500 after ~300s), so the JSON contract "
            "is enforced in the prompt and in parse_caption_payload instead"
        ),
    )
    caption_event_iou_min: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Below this IoU between the VLM's event span and the label-derived span, a row is marked needs_review",
    )
    caption_max_chars: int = Field(default=200, ge=20, description="Captions longer than this are rejected as not one-line")

    # ------------------------------------------------------- incident clips
    clip_dataset_dir_override: Path | None = Field(
        default=None, description="Override where concatenated incident videos/manifests are written"
    )
    clip_val_frac: float = Field(default=0.1, ge=0.0, lt=1.0, description="Fraction of incidents held out (split by incident, never by clip)")
    event_pad_seconds: float = Field(default=0.0, ge=0.0, description="Pad added to each end of a label-derived event span")
    event_merge_gap_seconds: float = Field(default=0.0, ge=0.0, description="Merge two positive runs closer together than this")
    clip_num_frames: int = Field(default=8, ge=1, description="Frames sampled per concatenated incident video during clip training")
    clip_max_seq_length: int = Field(
        default=1024, ge=128,
        description="Token budget for clip samples. 8 frames x 64 vision tokens/frame needs >=1024",
    )
    clip_run_name: str = Field(default="smolvlm2-dcsass-clips-lora", description="MLflow run name for the clip (caption) task")
    include_needs_review: bool = Field(default=False, description="Include needs_review manifest rows in the training split")

    # ------------------------------------------------------------- derived
    @property
    def default_vlm_base_url(self) -> str:
        """OpenAI-compatible base URL of the gateway hosted by the tracking server."""
        return f"{self.mlflow_tracking_uri.rstrip('/')}/{GATEWAY_PATH}"

    @model_validator(mode="after")
    def _default_vlm_base_url_to_gateway(self) -> Settings:
        """When no explicit gateway URL is configured, use the tracking server's own.

        ``model_post_init`` is unreliable for pydantic-settings (env values are
        applied after it), so the fallback is resolved here instead: an unset
        ``vlm_base_url`` is filled from ``mlflow_tracking_uri`` and stays
        overridable through ``SMOLVLM_FT_VLM_BASE_URL``.
        """
        if not self.vlm_base_url:
            self.vlm_base_url = self.default_vlm_base_url
        return self

    @property
    def clip_dataset_dir(self) -> Path:
        if self.clip_dataset_dir_override is not None:
            return self.clip_dataset_dir_override
        return self.dataset_dir.parent / "dcsass-clips"

    @property
    def clip_index_path(self) -> Path:
        return self.clip_dataset_dir / "clip_index.jsonl"

    @property
    def captions_path(self) -> Path:
        return self.clip_dataset_dir / "captions.jsonl"

    @property
    def manifest_jsonl_path(self) -> Path:
        return self.clip_dataset_dir / "manifest.jsonl"

    @property
    def manifest_parquet_path(self) -> Path:
        return self.clip_dataset_dir / "manifest.parquet"

    @property
    def qc_report_path(self) -> Path:
        return self.clip_dataset_dir / "qc_report.md"

    @property
    def train_clips_parquet(self) -> Path:
        return self.dataset_dir / "train_clips.parquet"

    @property
    def val_clips_parquet(self) -> Path:
        return self.dataset_dir / "val_clips.parquet"

    @property
    def clip_run_output_dir(self) -> Path:
        return self.output_dir / self.clip_run_name

    @property
    def clip_merged_output_dir(self) -> Path:
        return self.output_dir / f"{self.clip_run_name}-merged"

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
