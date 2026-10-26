"""Batch collation for video SFT.

SmolVLM's processor turns one chat example into the tensors the model
consumes::

    input_ids (B, L), attention_mask (B, L),
    pixel_values (B, F, 3, H, W), pixel_attention_mask (B, F, H, W)

The collator below owns three things:

1. **Frame decoding** — decord (fast, reliable) instead of the default
   torchvision/torchcodec backends, which break on modern torchvision builds.
   We also build correct ``VideoMetadata`` so the per-frame timestamps the
   model embeds in the prompt are accurate.
2. **Chat templating** — the full conversation is templated once; the prompt
   half is templated separately to find where the assistant answer starts.
3. **Label masking** — loss is computed only on the assistant completion
   (user prompt + vision tokens are masked with -100).

Examples over ``max_seq_length`` are dropped (logged once) so a single long
clip can never blow up the batch.

For the caption/timing task (concatenated incident videos, several tens of
seconds long) the collator can additionally sample frames **inside the labelled
event window** — see :class:`ClipSamplingConfig`. Uniform sampling across a
90-second video would put nearly every frame outside the 10-second event, so the
model would have to answer "when" from frames that do not show the event.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import decord
import numpy as np
import torch
from PIL import Image
from transformers import ProcessorMixin

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoCollatorConfig:
    """Knobs for :class:`VideoSFTCollator`."""

    max_seq_length: int = 512
    num_frames: int = 4
    max_image_size: int = 448
    #: Set this to sample frames around the labelled event instead of uniformly,
    #: and to append the sampled frame times to the prompt (see ClipSamplingConfig).
    clip_sampling: ClipSamplingConfig | None = None
    #: Append "Sampled frame times: ..." to the question so the model can answer
    #: in seconds. The processor already labels each frame "Frame from MM:SS:",
    #: but MM:SS is not the format the answer is scored on.
    ground_timestamps: bool = True


@dataclass(frozen=True)
class ClipSamplingConfig:
    """Windowed frame sampling for long concatenated incident videos.

    Uniform sampling across a 90-second incident puts almost every frame outside
    the event, so the model would have to answer a timing question from frames
    that do not show the event at all. Instead most frames are drawn from inside
    the event spans and a few from outside, which teaches both "what happened"
    and "when" while keeping the frame count (and therefore the vision token
    count) fixed.
    """

    event_frames: int = 6
    context_frames: int = 2


def _sample_indices_in_range(total: int, low: float, high: float, fps: float, count: int) -> list[int]:
    """Frame indices uniformly spread over the ``[low, high)`` second window."""
    if count <= 0 or total <= 0:
        return []
    first = max(0, min(total - 1, int(low * fps)))
    last = max(first, min(total - 1, int(high * fps) - 1))
    if last <= first:
        return [first]
    step = (last - first) / max(1, count - 1)
    return [round(first + step * i) for i in range(count)]


def _allocate(count: int, weights: list[float]) -> list[int]:
    """Split ``count`` frames over windows proportionally, largest remainder first."""
    total_weight = sum(weights)
    if count <= 0 or not weights or total_weight <= 0:
        return [0] * len(weights)
    ideal = [count * w / total_weight for w in weights]
    taken = [min(int(floor), count) for floor in ideal]
    remaining = count - sum(taken)
    order = sorted(range(len(weights)), key=lambda i: ideal[i] - taken[i], reverse=True)
    index = 0
    while remaining > 0 and order:
        taken[order[index % len(order)]] += 1
        remaining -= 1
        index += 1
    return taken


def _windowed_frame_indices(
    total: int,
    fps: float,
    event_spans: list[dict[str, Any]],
    event_frames: int,
    context_frames: int,
) -> list[int]:
    """Frame indices concentrated inside ``event_spans`` plus context frames outside.

    Spans are wall-clock seconds in the concatenated video. Frames are spread
    over each span proportionally to its length, then the remaining
    ``context_frames`` are spread over the parts of the video outside the event,
    so the model also sees what the video looks like when nothing is happening.
    """
    duration = total / fps if fps else 0.0
    if duration <= 0:
        return []

    windows: list[tuple[float, float]] = []
    for span in event_spans or []:
        try:
            low, high = float(span["start"]), float(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        low, high = max(0.0, min(low, duration)), max(0.0, min(high, duration))
        if high > low:
            windows.append((low, high))
    if not windows:
        return []

    indices: set[int] = set()
    weights = [high - low for low, high in windows]
    for (low, high), count in zip(windows, _allocate(event_frames, weights)):
        indices.update(_sample_indices_in_range(total, low, high, fps, count))

    outside: list[tuple[float, float]] = []
    cursor = 0.0
    for low, high in windows:
        if low > cursor:
            outside.append((cursor, low))
        cursor = max(cursor, high)
    if cursor < duration:
        outside.append((cursor, duration))
    outside_weights = [high - low for low, high in outside]
    for (low, high), count in zip(outside, _allocate(context_frames, outside_weights)):
        indices.update(_sample_indices_in_range(total, low, high, fps, count))

    if not indices:
        return []
    # Keep the frame count exactly as configured: drop from the context frames
    # first (they are the least informative for a timing question), then from the
    # end of the event frames.
    ordered = sorted(indices)
    if len(ordered) > event_frames + context_frames:
        keep = event_frames + context_frames
        ordered = ordered[:keep]
    return ordered


def sample_video_frames_windowed(
    video_path: str,
    event_spans: list[dict[str, Any]],
    num_frames: int,
    frame_size: int,
    event_frames: int = 6,
    context_frames: int = 2,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Decode frames concentrated inside the event, with correct metadata.

    Falls back to plain uniform sampling when the row carries no event spans, so
    the same collator serves both the Yes/No task and the caption/timing task.
    Metadata keeps the *original* frame indices plus the real fps, which is what
    makes SmolVLM write a truthful ``Frame from MM:SS:`` label for every frame —
    the whole point of the timing task.
    """
    reader = decord.VideoReader(video_path, num_threads=1)
    total = len(reader)
    if total == 0:
        raise ValueError(f"Video has no frames: {video_path}")
    fps = float(reader.get_avg_fps()) or 30.0

    indices = _windowed_frame_indices(total, fps, event_spans, event_frames, context_frames)
    if not indices:
        return sample_video_frames(video_path, num_frames, frame_size)
    indices = sorted(set(indices))[:num_frames] if len(indices) > num_frames else sorted(set(indices))

    batch = reader.get_batch(indices).asnumpy()  # (F, H, W, 3) uint8
    frames = [
        np.array(Image.fromarray(batch[i]).resize((frame_size, frame_size), Image.BILINEAR))
        for i in range(batch.shape[0])
    ]
    metadata = {
        "total_num_frames": total,
        "fps": fps,
        "width": frame_size,
        "height": frame_size,
        "duration": total / fps,
        "video_backend": "decord",
        "frames_indices": [int(i) for i in indices],
    }
    return frames, metadata


def frame_times_note(meta: dict[str, Any]) -> str:
    """The sampled frames' capture times in seconds, as prompt text.

    SmolVLM labels every frame ``Frame from MM:SS:`` internally, but the answer
    we train and score is in seconds. Stating the same times in seconds removes a
    unit conversion the model would otherwise have to learn, and it makes a
    second-level answer verifiable from the prompt alone.
    """
    fps = meta.get("fps") or 0.0
    indices = meta.get("frames_indices") or []
    if not fps or not indices:
        return ""
    times = ", ".join(f"{int(i) / fps:.1f}" for i in indices)
    return (
        f"\nThe {len(indices)} frames above were captured at these times in seconds: "
        f"{times}. Answer in the same unit."
    )



def sample_video_frames(
    video_path: str, num_frames: int, frame_size: int
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Decode ``num_frames`` uniformly spaced frames with decord and resize them.

    Frames are resized to a fixed ``frame_size x frame_size`` square (the model's
    native ``image_size``). This is required: the SmolVLM merger asserts that the
    number of ``<image>`` tokens is divisible by the per-frame vision token count,
    which only holds when every frame has the model's native spatial size. Returns
    ``(H, W, 3)`` uint8 arrays plus the metadata dict the processor uses for the
    per-frame timestamps it writes into the prompt.
    """
    reader = decord.VideoReader(video_path, num_threads=1)
    total = len(reader)
    if total == 0:
        raise ValueError(f"Video has no frames: {video_path}")
    fps = float(reader.get_avg_fps()) or 30.0
    indices = np.linspace(0, total - 1, min(num_frames, total)).astype(int)
    batch = reader.get_batch(indices).asnumpy()  # (F, H, W, 3) uint8
    frames = [
        np.array(Image.fromarray(batch[i]).resize((frame_size, frame_size), Image.BILINEAR))
        for i in range(batch.shape[0])
    ]
    metadata = {
        "total_num_frames": total,
        "fps": fps,
        "width": frame_size,
        "height": frame_size,
        "duration": total / fps,
        "video_backend": "decord",
        "frames_indices": [int(i) for i in indices],
    }
    return frames, metadata


def _extract_video_and_messages(example: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Pull the video path and chat messages out of a raw dataset row."""
    messages = example["messages"]
    video_path = None
    for item in messages[0]["content"]:
        if item.get("type") == "video":
            video_path = item.get("path") or item.get("url")
            break
    if video_path is None:
        raise ValueError(f"No video item found in example: {example}")
    return video_path, messages


class VideoSFTCollator:
    """Turns raw chat rows into a model-ready batch dict with masked labels."""

    def __init__(self, processor: ProcessorMixin, config: VideoCollatorConfig) -> None:
        self.processor = processor
        self.config = config
        self._dropped_logged = False

    def _decode(self, video_path: str, example: dict[str, Any]):
        """Uniform sampling by default; event-window sampling for clip rows."""
        clip = self.config.clip_sampling
        spans = list(example.get("event_spans") or [])
        if clip is not None and spans:
            return sample_video_frames_windowed(
                video_path,
                spans,
                self.config.num_frames,
                self.config.max_image_size,
                event_frames=clip.event_frames,
                context_frames=clip.context_frames,
            )
        return sample_video_frames(video_path, self.config.num_frames, self.config.max_image_size)

    def _ground_timestamps(self, messages: list[dict[str, Any]], meta: dict[str, Any]) -> list[dict[str, Any]]:
        """Append the sampled frame times to the question, when asked for.

        Applied to both the full conversation and the prompt-only conversation, so
        the token where the answer starts is still found correctly.
        """
        if not self.config.ground_timestamps:
            return messages
        note = frame_times_note(meta)
        if not note:
            return messages
        grounded = [dict(message) for message in messages]
        content = list(grounded[0]["content"])
        for item in content:
            if item.get("type") == "text":
                item["text"] = str(item.get("text", "")) + note
                break
        grounded[0]["content"] = content
        return grounded

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        processed: list[dict[str, torch.Tensor]] = []
        videos: list[list[np.ndarray]] = []
        metadata: list[dict[str, Any]] = []

        for example in examples:
            video_path, messages = _extract_video_and_messages(example)
            try:
                frames, meta = self._decode(video_path, example)
            except Exception as exc:  # corrupt clip, missing file, ...
                logger.warning("Skipping undecodable video %s: %s", video_path, exc)
                continue

            messages = self._ground_timestamps(messages, meta)
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_text = self.processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )

            encoded = self.processor(
                text=[full_text],
                videos=[[frames]],
                video_metadata=[meta],
                padding=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"][0]
            if input_ids.shape[0] > self.config.max_seq_length:
                if not self._dropped_logged:
                    logger.warning(
                        "Dropping examples longer than max_seq_length=%d (first: %d tokens). "
                        "Raise max_seq_length or lower num_frames to keep them.",
                        self.config.max_seq_length, input_ids.shape[0],
                    )
                    self._dropped_logged = True
                continue

            # The processor expands the single <video> token into ~256 vision
            # tokens *inside the prompt*, so the raw prompt-text length is the
            # wrong boundary. Run the processor on the prompt-only conversation
            # (same frames -> same expansion); its token length is the exact
            # index where the assistant answer begins. Loss is masked before it.
            prompt_encoded = self.processor(
                text=[prompt_text],
                videos=[[frames]],
                video_metadata=[meta],
                padding=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_len = prompt_encoded["input_ids"].shape[1]

            labels = input_ids.clone()
            labels[:prompt_len] = -100

            processed.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": encoded["attention_mask"][0],
                    "pixel_values": encoded["pixel_values"][0],
                    "pixel_attention_mask": encoded["pixel_attention_mask"][0],
                    "labels": labels,
                }
            )
            videos.append(frames)
            metadata.append(meta)

        if not processed:
            raise RuntimeError("Collator received no processable examples (all dropped or undecodable).")

        return self._pad(processed)

    def _pad(self, items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Right-pad the text tensors; stack (uniform) vision tensors."""
        pad_id = self.processor.tokenizer.pad_token_id
        max_len = max(item["input_ids"].shape[0] for item in items)

        input_ids = torch.full((len(items), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(items), max_len), dtype=torch.long)
        labels = torch.full((len(items), max_len), -100, dtype=torch.long)
        for row, item in enumerate(items):
            length = item["input_ids"].shape[0]
            input_ids[row, :length] = item["input_ids"]
            attention_mask[row, :length] = item["attention_mask"]
            labels[row, :length] = item["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.stack([item["pixel_values"] for item in items]),
            "pixel_attention_mask": torch.stack([item["pixel_attention_mask"] for item in items]),
        }