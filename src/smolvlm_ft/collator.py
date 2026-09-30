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

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        processed: list[dict[str, torch.Tensor]] = []
        videos: list[list[np.ndarray]] = []
        metadata: list[dict[str, Any]] = []

        for example in examples:
            video_path, messages = _extract_video_and_messages(example)
            try:
                frames, meta = sample_video_frames(
                    video_path, self.config.num_frames, self.config.max_image_size
                )
            except Exception as exc:  # corrupt clip, missing file, ...
                logger.warning("Skipping undecodable video %s: %s", video_path, exc)
                continue

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