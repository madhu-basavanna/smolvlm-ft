"""Model construction: base SmolVLM2 + LoRA adapters, frozen vision tower.

The vision tower is large and expensive to backprop through. Because LoRA only
trains the LLM linear layers, the vision tower is fully frozen, so its
features can be computed once under ``no_grad`` and handed to the LLM as
constants. :func:`freeze_vision_backward` patches the inner model's
``get_image_features`` to run without building an autograd graph. The LLM
(with LoRA) then backprops as usual, but the vision encoder never appears in
the backward graph. This skips the vision backward pass entirely and saves
roughly the vision tower's activation memory, which is what makes a 500M
video model trainable in ~3 GB of free VRAM.
"""

from __future__ import annotations

import logging

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from .config import Settings

logger = logging.getLogger(__name__)


def resolve_device(settings: Settings) -> torch.device:
    """Pick the training device: CUDA when available, else CPU."""
    if settings.device.value == "cpu":
        return torch.device("cpu")
    if settings.device.value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_processor(settings: Settings):
    """Load the processor and pin the frame-sampling knobs used at train time.

    The collator decodes and resizes frames itself (decord) to a fixed square
    ``max_image_size``; these settings are belt-and-braces for the processor's
    own decoding path.
    """
    processor = AutoProcessor.from_pretrained(settings.model_name)
    if hasattr(processor, "video_processor"):
        processor.video_processor.num_frames = settings.num_frames
        processor.video_processor.fps = 2
        processor.video_processor.max_image_size = {"longest_edge": settings.max_image_size}
    return processor


def _find_vision_model(model) -> object:
    """Locate the inner ``SmolVLMModel`` (the one that exposes ``get_image_features``)."""
    for m in model.modules():
        if hasattr(m, "get_image_features"):
            return m
    raise RuntimeError("Could not locate SmolVLMModel.get_image_features in the model tree.")


def freeze_vision_backward(model) -> None:
    """Run the frozen vision encoder under ``no_grad``.

    Replaces ``SmolVLMModel.get_image_features`` with a wrapper that computes
    the same features but does not record them in the autograd graph. The
    vision tower's weights are frozen, so no gradients are lost.
    """
    vision_model = _find_vision_model(model)
    original = vision_model.get_image_features

    def no_grad_get_image_features(pixel_values, pixel_attention_mask=None):
        with torch.no_grad():
            return original(pixel_values, pixel_attention_mask)

    vision_model.get_image_features = no_grad_get_image_features
    logger.info("Vision tower now runs under no_grad (features are constants to the LLM).")


def build_lora_model(settings: Settings, device: torch.device):
    """Load the base model, attach LoRA to the LLM layers, freeze the vision tower.

    Returns the PEFT model (on ``device`` when CUDA).
    """
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    logger.info("Loading %s (dtype=%s, attn=%s)", settings.model_name, dtype, settings.attn_implementation)
    # Pin the whole model to a single device: "auto" would shard even this small
    # model across multiple GPUs when per-GPU VRAM is scarce, which breaks
    # gradient checkpointing (tensors end up split across cuda:0/cuda:1).
    device_map = {"": device.index or 0} if device.type == "cuda" else {"": "cpu"}
    model = AutoModelForImageTextToText.from_pretrained(
        settings.model_name,
        torch_dtype=dtype,
        attn_implementation=settings.attn_implementation,
        device_map=device_map,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=settings.lora_targets,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # PEFT already freezes non-LoRA params; be explicit that the vision tower
    # stays frozen or memory blows up.
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad_(False)

    freeze_vision_backward(model)

    if device.type == "cuda":
        model = model.to(device)
    model.print_trainable_parameters()
    return model
