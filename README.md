# SmolVLM2-500M-Video-Instruct → DCSASS fine-tuning

LoRA fine-tuning of [`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)
on the [DCSASS violence-detection dataset](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset)
(16.5k short video clips, 13 classes), with all metrics logged to a self-hosted
MLflow server.

Task: given a clip, answer *"Is `<class>` shown in this video?"* with **Yes/No +
one-sentence justification** (chat SFT, loss only on the assistant answer).

## Architecture & memory design

This build is engineered to run in **~3 GB of free VRAM** (e.g. a 6 GB card with
~3 GB already occupied):

- **LoRA** (rank 16) on the LLM linear layers → ~9.6 M trainable params
  (1.85 % of 517 M). Vision tower and embeddings stay frozen.
- **Frozen-vision backward skip** (`freeze_vision_backward`): the vision tower
  is frozen, so its features are computed under `no_grad` and handed to the LLM
  as constants. The vision encoder never appears in the backward graph, saving
  its activation memory.
- **`bfloat16`** training, SDPA attention, gradient checkpointing.
- **Frames are decoded with decord** (not the default torchvision/torchcodec
  backends, which break on modern torchvision) and resized to a fixed **512×512**
  square. This size is required, not optional: the SmolVLM merger asserts the
  `<image>` token count is divisible by the per-frame vision token count, which
  only holds at the model's native `image_size=512` (64 tokens/frame).

Measured peak (512×512, 4 frames, 1 batch, gradient checkpointing, with the
frozen-vision skip): **~2.6 GB** → fits in 3 GB. Without the frozen-vision skip
the vision backward OOMs even at 1 frame.

> On a GPU with more free VRAM you can raise `per_device_batch_size` and
> `num_frames` to go faster.

## Setup

```bash
cd masters_project/smolvlm
uv venv .venv --python 3.12
uv pip install -e .
source .venv/bin/activate
```

### 1. Get the dataset

```bash
mkdir -p data && cd data
curl -sSL -o dcsass.zip "https://www.kaggle.com/api/v1/datasets/download/mateohervas/dcsass-dataset"
unzip -q dcsass.zip
cd ..
```

### 2. Build the train/val parquet

```bash
smolvlm-ft build-data
# → data/dataset/train.parquet (14,923 rows), data/dataset/val.parquet (1,658 rows)
```

### 3. Configure MLflow

```bash
cp .env.example .env
# SMOLVLM_FT_MLFLOW_TRACKING_URI=http://mlflow-server-1:8001
```

### 4. Train

```bash
smolvlm-ft train
```

Watch it live at `http://mlflow-server-1:8001` → experiment `smolvlm2-dcsass`.

## CLI

```
smolvlm-ft build-data   # scan raw DCSASS → parquet (idempotent)
smolvlm-ft train        # fine-tune + log to MLflow + save merged model
```

Every hyper-parameter is an env var (`SMOLVLM_FT_*`) or a field in
`src/smolvlm_ft/config.py` (`Settings`). Useful overrides:

```bash
SMOLVLM_FT_MAX_STEPS=500 \
SMOLVLM_FT_NUM_FRAMES=4 \
SMOLVLM_FT_PER_DEVICE_BATCH_SIZE=1 \
SMOLVLM_FT_GRADIENT_ACCUMULATION_STEPS=4 \
smolvlm-ft train
```

## Outputs

- `outputs/<run_name>/` — LoRA adapter (`adapter_model.safetensors`) + checkpoints
- `outputs/<run_name>-merged/` — LoRA merged into the base model, ready to serve

## Tests

```bash
uv pip install -e ".[dev]"
pytest
```

8 tests, including a real-clip collator test (decord decode → batch tensors →
correct label masking). No GPU needed.

## Verifying on a low-VRAM box (CPU smoke test)

To prove the whole pipeline (data → LoRA → collator → trainer → MLflow →
save/merge) without a big GPU, run a short CPU job:

```bash
SMOLVLM_FT_DEVICE=cpu SMOLVLM_FT_MAX_STEPS=2 SMOLVLM_FT_NUM_FRAMES=1 \
SMOLVLM_FT_EVAL_STEPS=999 smolvlm-ft train
```

(CPU is fp32; on GPU the run is bf16. ~1–2 s/step on CPU for 1 frame.)

## Notes

- Dataset clips are 320×240, 1–5 s. They are upscaled to 512×512 (the model's
  native size); 4 frames @ ~2 fps covers the whole clip.
- The dataset's `Labels/*.csv` map 1:1 to the `.mp4` files (each row is a clip).
  A clip is positive when its flag is `1`.
- If you hit OOM on the target server, lower `SMOLVLM_FT_NUM_FRAMES` first
  (2 → 1), then `SMOLVLM_FT_PER_DEVICE_BATCH_SIZE=1`.