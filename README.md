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

`build-clips` concatenates videos with the **ffmpeg binary** (`apt install ffmpeg`
/ `brew install ffmpeg`). If it is missing, the code falls back to re-encoding
through PyAV (`av`, already a dependency): slower, but the same output format.

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
# SMOLVLM_FT_MLFLOW_TRACKING_URI=https://mlflow.home
```

### 4. Train

```bash
smolvlm-ft train
```

Watch it live at `https://mlflow.home` → experiment `smolvlm2-dcsass`.

## CLI

```
smolvlm-ft build-data      # scan raw DCSASS → parquet (idempotent)
smolvlm-ft train           # fine-tune + log to MLflow + save merged model
smolvlm-ft evaluate        # base vs fine-tuned on the val split
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

## Second task: incident videos + VLM captions

The raw dataset is 16,639 short clips grouped per source video. This second task
turns each group back into **one video** and derives **when the event happens**,
then has a VLM describe the event in one line. The plan behind it is in
[`docs/dcsass-clip-dataset-plan.md`](docs/dcsass-clip-dataset-plan.md).

Nothing here changes the Yes/No dataset or its split.

```bash
smolvlm-ft build-clips        # index, measure durations, concat each incident
smolvlm-ft caption-clips      # ask the gateway VLM for a caption + event span
smolvlm-ft build-clip-data    # manifest → train_clips/val_clips.parquet
smolvlm-ft train --task clips # fine-tune on the caption/timing task
smolvlm-ft evaluate --task clips
```

**The dataset stores no timestamps.** What it stores is `<Class>NNN_x264_<k>.mp4`
segments plus a per-clip `0|1` flag. So the timing is *derived*: every clip's
duration is measured with decord, accumulated in clip-index order (never
lexicographic — `_10` sorts before `_2`), and a run of consecutive `1`s becomes
one event span in wall-clock seconds. Concatenation re-encodes to one uniform
h264 stream so "seconds in the output file" and "summed clip durations" are the
same number, and each output is re-opened and checked against that sum (±0.25 s)
before its timestamps are trusted.

### The VLM (MLflow AI Gateway)

Captioning goes through the OpenAI-compatible route the MLflow server itself
serves — no separate endpoint, no API key:

```
base_url = {SMOLVLM_FT_MLFLOW_TRACKING_URI}/gateway/mlflow/v1   # default, derived
model    = qwen3.8-flash-next-local                              # a gateway *endpoint name*
api_key  = ""                                                   # gateway auth is server-side
```

Before any real run, confirm the endpoint name actually exists (the gateway
resolves `model` to a *configured endpoint name*, not an upstream model id):

```bash
curl --cacert caddy-root-homelab.crt https://mlflow.home/gateway/mlflow/v1/models
uv run smolvlm-ft caption-clips --dry-run          # request/token estimate
uv run smolvlm-ft caption-clips --limit 5          # pilot, inspect captions.jsonl
uv run smolvlm-ft caption-clips                    # all 520 incidents
```

`caption-clips` calls `mlflow.autolog()` before `openai` is imported, so every
gateway call is recorded as a trace under a `caption-clips` parent span, in
experiment `smolvlm2-dcsass-caption`. Each manifest row keeps its `mlflow_trace_id`.
The model is **never told the ground-truth class**; the labels are used only
afterwards to score the answer (`event_iou`, `needs_review`).

Useful flags: `--limit N`, `--workers N`, `--mode contact_sheet|keyframes`,
`--keyframes N`, `--overwrite` (ignore the caption cache), `--dry-run`,
`--no-autolog`.

#### What this endpoint actually accepts (measured, not assumed)

Every keyframe is labelled with the time it was captured — without that, a request
to answer in seconds is unanswerable. *How* they are sent is configurable, because
what a serving stack tolerates is a property of that stack. These were measured
against `qwen3.8-flash-next-local` (upstream `qwen3.8-flash-next-iq3_xxs`):

| Request shape | Result |
|---|---|
| 1–2 separate `image_url` frames | HTTP 200 in ~7–9 s |
| **3+ separate frames** (256px, 384px, even 128px) | HTTP 500 after ~300 s |
| Thinking left ON (`enable_thinking: true`) | reasoning consumes `max_tokens` → `finish_reason=length`, `content: null` |
| `response_format={"type":"json_object"}` | HTTP 502 `structured output was incomplete`, or a 500 after ~300 s |
| `max_tokens=400` on a 12-tile sheet | truncated (`finish_reason=length`) |
| **12 tiles in ONE contact sheet** (4 cols × 256px) | HTTP 200 in 20–165 s, ~1.2k prompt tokens |

So the defaults are: `caption_mode=contact_sheet`, `caption_keyframes=12`,
`caption_sheet_columns=4`, `caption_tile_size=256`, `caption_max_tokens=700`,
`caption_disable_thinking=true`, `caption_use_response_format=false`,
`vlm_timeout=240`. `caption_mode=keyframes` stays selectable for a stack that can
handle it (keep `caption_keyframes<=2` there); `preflight` warns when that limit is
exceeded, and `--dry-run` prints the layout it would use.

A contact sheet is one JPEG tile-grid with `#7 t=41.2s` drawn under each tile, and
the same labels repeated as text in the prompt — so the timestamps survive whatever
the vision encoder does to the image. With thinking off and no `response_format`,
the JSON contract is enforced by the prompt plus `parse_caption_payload`'s fallbacks
(strict JSON → embedded object → text + time-range regex).


### Outputs

| Path | Contents |
|---|---|
| `data/dcsass-clips/<Class>/<incident_id>.mp4` | one concatenated video per incident |
| `data/dcsass-clips/clip_index.jsonl` | measured per-clip durations (cache; invalidated by size/mtime) |
| `data/dcsass-clips/captions.jsonl` | append-only caption cache, keyed by `incident_id + prompt_hash` |
| `data/dcsass-clips/manifest.jsonl` / `.parquet` | one auditable row per incident: spans, caption, IoU, trace id |
| `data/dcsass-clips/qc_report.md` | per-class counts, label anomalies, concat failures, span mismatches |
| `data/dataset/train_clips.parquet` / `val_clips.parquet` | chat rows for the caption/timing task |
| `outputs/<clip_run_name>[-merged]/` | LoRA adapter / merged model for the clip task |

The clip split is **by incident**, never by clip, so two windows of the same
source video can never appear on both sides of the split. Timing targets come
from the labels (ground truth); the VLM's own timestamps are only used for QC.

### Token budget (hard constraint, not a preference)

At the mandatory 512×512 the SmolVLM merger produces exactly 64 vision tokens per
frame, so `num_frames × 64` must fit inside `max_seq_length` or the collator
drops the sample:

| Task | frames | vision tokens | `max_seq_length` |
|---|---|---|---|
| Yes/No (short clips) | 4 | 256 | 512 |
| Clips (incidents) | 8 | 512 | 1024 |

For the clip task the collator samples most frames **inside the labelled event
window** plus a couple outside, because uniform sampling over a 90-second incident
would put every frame outside a 10-second event. It also appends the sampled
frames' capture times in seconds to the question, so a second-level answer is
learnable and verifiable from the prompt alone.

## Model outputs

- `outputs/<run_name>/` — LoRA adapter (`adapter_model.safetensors`) + checkpoints
- `outputs/<run_name>-merged/` — LoRA merged into the base model, ready to serve
- `outputs/<clip_run_name>/` and `outputs/<clip_run_name>-merged/` — the same for `--task clips`

## Tests

```bash
uv pip install -e ".[dev]"
pytest
```

No GPU needed. Covers: settings/CLI config, the stratified Yes/No split, a
real-clip collator test (decord decode → batch tensors → correct label masking),
incident indexing and **derived timestamps** (the hand-verified `Abuse001` 4–12 +
16–31, `Abuse007` 2–4 and `Shoplifting001` 9–15 runs, span padding, gap splitting),
the label-file quirks that must be *reported* rather than dropped, the derived
`vlm_base_url`, both caption request layouts (keyframes vs contact sheet), and
caption parsing/QC against canned gateway replies (including one that is not JSON at
all).

## Verifying on a low-VRAM box (CPU smoke test)

To prove the whole pipeline (data → LoRA → collator → trainer → MLflow →
save/merge) without a big GPU, run a short CPU job:

```bash
SMOLVLM_FT_DEVICE=cpu SMOLVLM_FT_MAX_STEPS=2 SMOLVLM_FT_NUM_FRAMES=1 \
SMOLVLM_FT_EVAL_STEPS=999 smolvlm-ft train
```

(CPU is fp32; on GPU the run is bf16. ~1–2 s/step on CPU for 1 frame.)

The clip task has the same smoke recipe, with its own frame/knob names:

```bash
SMOLVLM_FT_DEVICE=cpu SMOLVLM_FT_MAX_STEPS=2 SMOLVLM_FT_CLIP_NUM_FRAMES=1 \
SMOLVLM_FT_EVAL_STEPS=999 SMOLVLM_FT_CAPTION_KEYFRAMES=4 \
smolvlm-ft build-clips --limit 2 && \
smolvlm-ft caption-clips --limit 2 && \
smolvlm-ft build-clip-data && \
smolvlm-ft train --task clips
```

(`build-clips` and `build-clip-data` need no GPU and no network. `caption-clips`
needs the gateway; add `--no-autolog` to run it without tracing.)

A caption pilot of only a few incidents produces **no validation split** —
`clip_val_frac` rounds to zero per class. That is a supported state, not an error:
an empty split cannot be written as parquet at all (and cannot be read back if it
were), so `val_clips.parquet` is simply not written, training logs a warning and
disables evaluation, and the first checkpoint is saved at `first_checkpoint_step`
with no metric. `evaluate --task clips` refuses to run on an empty val split rather
than returning an empty report.

## Notes

- Dataset clips are 320×240, 1–5 s. They are upscaled to 512×512 (the model's
  native size); 4 frames @ ~2 fps covers the whole clip.
- The dataset's `Labels/*.csv` map 1:1 to the `.mp4` files (each row is a clip).
  A clip is positive when its flag is `1`.
- If you hit OOM on the target server, lower `SMOLVLM_FT_NUM_FRAMES` first
  (2 → 1), then `SMOLVLM_FT_PER_DEVICE_BATCH_SIZE=1`.