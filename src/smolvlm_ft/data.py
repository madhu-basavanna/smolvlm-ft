"""DCSASS dataset discovery and conversion to chat-format parquet files.

The raw Kaggle dataset is organised as::

    DCSASS Dataset/
        <Class>/<clip_group>.mp4/<clip_id>.mp4    # one short clip per file
        Labels/<Class>.csv                        # "clip_id,Class,0|1"

Each CSV row maps 1:1 to one video file whose basename is ``<clip_id>.mp4``
(the rows are the clips themselves; the parent folder groups them per source
video). The label column is the ground truth for *that clip*. This module
scans the layout, assigns a question template and a Yes/No answer, and writes
a stratified train/val split as parquet. Parquet rows carry the absolute video
path plus the chat messages, so the training script only loads parquet and
never touches the raw folder layout again.

The second task in this file builds the **incident** dataset produced by
:mod:`smolvlm_ft.build_dataset`: one concatenated video per source video, a
VLM-written one-line caption, and the label-derived moment when the event
happens. Those rows use the same chat format as the Yes/No rows, but the
question asks for a time span plus a description. Their split is by
**incident**, never by clip, so two windows of the same source video can never
appear on both sides of the split.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)

CLASS_ORDER = [
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]

#: One shared question for all classes (the class name is embedded in the
#: answer). Varying the question would force the model to learn the class
#: identity from the video; we only want Yes/No detection.
QUESTION = "Is {class_name} shown in this video? Answer with Yes or No, then give a one-sentence justification."

CLIP_QUESTION = (
    "At what time does the main event occur in this video, and what happens? "
    "Answer with the start and end time in seconds, then one line describing the event."
)


def format_clip_answer(event_start: float, event_end: float, caption: str) -> str:
    """Target answer for a caption row: an explicit span plus the one-line caption."""
    return f"{event_start:.1f}s-{event_end:.1f}s. {caption.strip().rstrip('.')}"


_TIME_SPAN_RE = re.compile(
    r"(?P<start>\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?\s*(?:-|\u2013|\u2014|~|\bto\b|\bthrough\b)\s*"
    r"(?P<end>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def parse_time_span(text: str) -> tuple[float, float] | None:
    """Read a ``12.4s-27.8s`` / ``12.4 to 27.8 seconds`` span out of model output.

    Used both when building training targets and when scoring generated answers.
    Falls back to the first two numbers in the text, which is what a model that
    ignores the requested format usually produces.
    """
    match = _TIME_SPAN_RE.search(text or "")
    if match is not None:
        start, end = float(match.group("start")), float(match.group("end"))
    else:
        numbers = _NUMBER_RE.findall(text or "")
        if len(numbers) < 2:
            return None
        start, end = float(numbers[0]), float(numbers[1])
    if end < start:
        start, end = end, start
    return start, end


@dataclass(frozen=True)
class ClipRecord:
    """One fine-tuning example: a video clip plus its derived conversation.

    ``event_spans`` is empty for the Yes/No task and carries the label-derived
    event moments for the caption/timing task, so the collator can concentrate its
    frames inside the event.
    """

    video_path: str
    class_name: str
    positive: bool
    question: str
    answer: str
    event_spans: list[dict] = field(default_factory=list)


@dataclass
class DcsassDataset:
    """In-memory view of the raw DCSASS layout, built once from disk."""

    root: Path
    clips: list[ClipRecord] = field(default_factory=list)

    @classmethod
    def from_disk(cls, root: Path) -> DcsassDataset:
        labels_dir = root / "Labels"
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Labels directory not found under {root}. Is DCSASS_ROOT correct?")

        # Index every video file by basename; CSV rows reference them 1:1.
        videos_by_name = {p.name: p for p in root.glob("*/*/*.mp4")}
        if not videos_by_name:
            raise RuntimeError(f"No video files found under {root}; nothing to train on.")

        ds = cls(root=root)
        skipped_no_file = 0
        skipped_bad_label = 0
        for csv_path in sorted(labels_dir.glob("*.csv")):
            class_name = csv_path.stem
            with csv_path.open(newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                next(reader, None)  # header row
                for row in reader:
                    if len(row) < 3:
                        continue
                    clip_id, _, flag = row[0].strip(), row[1].strip(), row[2].strip()
                    video_path = videos_by_name.get(f"{clip_id}.mp4")
                    if video_path is None:
                        skipped_no_file += 1
                        continue
                    if flag not in ("0", "1"):
                        skipped_bad_label += 1
                        continue
                    positive = flag == "1"
                    ds.clips.append(
                        ClipRecord(
                            # Stored relative to dcsass_root so parquet files stay
                            # portable across machines/checkouts with different roots.
                            video_path=str(video_path.relative_to(root)),
                            class_name=class_name,
                            positive=positive,
                            question=QUESTION.format(class_name=class_name.lower()),
                            answer=(
                                f"Yes. The video shows {class_name.lower()} activity."
                                if positive
                                else f"No. The video does not show {class_name.lower()} activity."
                            ),
                        )
                    )

        if not ds.clips:
            raise RuntimeError(f"No clips found under {root}; nothing to train on.")
        logger.info(
            "Discovered %d clips under %s (skipped %d without file, %d with bad label)",
            len(ds.clips), root, skipped_no_file, skipped_bad_label,
        )
        return ds

    def split(self, val_frac: float, seed: int) -> tuple[list[ClipRecord], list[ClipRecord]]:
        """Stratified split so each class keeps the same positive ratio in both sets."""
        rng = random.Random(seed)
        train: list[ClipRecord] = []
        val: list[ClipRecord] = []
        for class_name in CLASS_ORDER:
            by_class = [c for c in self.clips if c.class_name == class_name]
            for positive in (True, False):
                bucket = [c for c in by_class if c.positive == positive]
                rng.shuffle(bucket)
                n_val = round(len(bucket) * val_frac)
                val.extend(bucket[:n_val])
                train.extend(bucket[n_val:])
        rng.shuffle(train)
        rng.shuffle(val)
        logger.info("Split: %d train / %d val", len(train), len(val))
        return train, val


# ------------------------------------------------- incident (caption) dataset

def load_clip_manifest(settings: Settings) -> list[dict[str, Any]]:
    """Read ``manifest.jsonl`` written by ``smolvlm-ft build-clip-data``.

    Prefers the JSONL because it is the auditable source; the parquet mirror exists
    only for fast loading by other tools.
    """
    path = settings.manifest_jsonl_path
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found at {path}. Run `smolvlm-ft build-clips`, `caption-clips` "
            "and `build-clip-data` first."
        )
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Ignoring corrupt manifest line in %s", path)
    if not rows:
        raise RuntimeError(f"Manifest {path} is empty.")
    logger.info("Loaded %d manifest rows from %s", len(rows), path)
    return rows


def manifest_to_clip_record(row: dict[str, Any], settings: Settings) -> ClipRecord | None:
    """Turn one manifest row into a training example, or skip it.

    The **timing** target comes from the label-derived spans (ground truth), not
    from the VLM's own timestamps, which are only used for QC. The **description**
    target is the VLM caption. That combination is what makes the task trainable:
    honest timing supervision plus free-text descriptions.
    """
    if not row.get("caption_ok") or not row.get("caption"):
        return None
    if row.get("needs_review") and not settings.include_needs_review:
        return None

    spans = row.get("event_spans") or []
    if spans:
        start = min(float(s["start"]) for s in spans)
        end = max(float(s["end"]) for s in spans)
    elif row.get("event_start") is not None and row.get("event_end") is not None:
        # No positive label: keep the model's own (usually empty) span so the
        # model can also learn to say "no event".
        start, end = float(row["event_start"]), float(row["event_end"])
    else:
        start, end = 0.0, 0.0
    if end <= start and spans:
        return None

    caption = str(row["caption"]).strip()
    if end > start:
        answer = format_clip_answer(start, end, caption)
    else:
        answer = f"No event detected. {caption}"

    return ClipRecord(
        # Relative to clip_dataset_dir; resolve_clip_video_path makes it absolute.
        video_path=str(row["video_path"]),
        class_name=str(row["class_name"]),
        positive=bool(spans),
        question=CLIP_QUESTION,
        answer=answer,
        # The collator uses these to sample frames inside the event.
        event_spans=[dict(s) for s in spans],
    )


def split_by_incident(
    records: list[ClipRecord], val_frac: float, seed: int
) -> tuple[list[ClipRecord], list[ClipRecord]]:
    """Split by incident id, stratified per class.

    ``DcsassDataset.split`` cannot be reused here: it splits per clip, and two
    clips of the *same* source video would then land on both sides of the split.
    The unit here is the whole incident video, so the id is the incident id taken
    from the video path (``Robbery017_x264.mp4``).
    """
    rng = random.Random(seed)
    by_class: dict[str, dict[str, ClipRecord]] = {}
    for record in records:
        incident_id = Path(record.video_path).stem
        by_class.setdefault(record.class_name, {})[incident_id] = record

    train: list[ClipRecord] = []
    val: list[ClipRecord] = []
    for class_name in sorted(by_class):
        ids = sorted(by_class[class_name])
        rng.shuffle(ids)
        n_val = round(len(ids) * val_frac)
        val.extend(by_class[class_name][incident_id] for incident_id in ids[:n_val])
        train.extend(by_class[class_name][incident_id] for incident_id in ids[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    logger.info("Incident split: %d train / %d val", len(train), len(val))
    return train, val


def resolve_clip_video_path(example: dict, clip_dir: Path) -> dict:
    """Make a manifest-relative incident video path absolute for this machine."""
    abs_path = str((clip_dir / example["video_path"]).resolve())
    example["video_path"] = abs_path
    for item in example["messages"][0]["content"]:
        if item.get("type") == "video":
            item["path"] = abs_path
    return example


def build_parquets(settings: Settings) -> tuple[Path, Path]:
    """Scan the raw dataset, write train/val parquet, return their paths."""
    settings.dataset_dir.mkdir(parents=True, exist_ok=True)
    ds = DcsassDataset.from_disk(settings.dcsass_root)
    train_clips, val_clips = ds.split(settings.val_frac, settings.seed)

    # Sanity: both splits must contain every class, otherwise stratification failed.
    for name, clips in (("train", train_clips), ("val", val_clips)):
        classes = {c.class_name for c in clips}
        missing = set(CLASS_ORDER) - classes
        if missing:
            raise RuntimeError(f"{name} split is missing classes: {sorted(missing)}")

    from datasets import Dataset  # imported here to keep module import light

    train_path = settings.train_parquet
    val_path = settings.val_parquet
    Dataset.from_list([clip_to_row(c) for c in train_clips]).to_parquet(train_path)
    Dataset.from_list([clip_to_row(c) for c in val_clips]).to_parquet(val_path)

    pos = sum(c.positive for c in train_clips)
    logger.info(
        "Wrote %s (%d rows, %d positive) and %s (%d rows)",
        train_path, len(train_clips), pos, val_path, len(val_clips),
    )
    return train_path, val_path


def _write_split_parquet(rows: list[dict], path: Path) -> bool:
    """Write one split to parquet. Returns False when there is nothing to write.

    An empty split is *not* written, and a stale file from an earlier run is
    removed. Two reasons: ``Dataset.from_list([])`` produces a parquet with no
    columns at all, and a columnless parquet can never be read back
    (``SchemaInferenceError`` / ``ArrowInvalid: BatchSize must be greater than 0``).
    A small ``clip_val_frac`` over a small captioned set easily produces an empty
    validation split, so the empty case has to be a supported state rather than a
    crash. ``load_clip_dataset`` reads it back as an empty dataset, and training
    turns evaluation off when there is none.
    """
    from datasets import Dataset

    if not rows:
        if path.is_file():
            path.unlink()
        return False
    Dataset.from_list(rows).to_parquet(path)
    return True


def build_clip_parquets(settings: Settings) -> tuple[Path, Path]:
    """Manifest -> train/val parquet for the caption/timing task."""
    rows = load_clip_manifest(settings)
    records: list[ClipRecord] = []
    skipped = 0
    for row in rows:
        record = manifest_to_clip_record(row, settings)
        if record is None:
            skipped += 1
            continue
        records.append(record)
    if not records:
        raise RuntimeError(
            "No usable manifest rows (all failed the caption check or are flagged "
            "needs_review). Set SMOLVLM_FT_INCLUDE_NEEDS_REVIEW=true to include them."
        )

    train, val = split_by_incident(records, settings.clip_val_frac, settings.seed)
    classes = {r.class_name for r in records}
    missing = set(CLASS_ORDER) - classes
    if missing:
        logger.warning("clip dataset is missing classes (they may have no captions): %s", sorted(missing))

    settings.dataset_dir.mkdir(parents=True, exist_ok=True)
    train_path = settings.train_clips_parquet
    val_path = settings.val_clips_parquet
    wrote_train = _write_split_parquet([clip_to_row(c) for c in train], train_path)
    wrote_val = _write_split_parquet([clip_to_row(c) for c in val], val_path)

    if not wrote_train:
        raise RuntimeError(
            f"Every incident landed in the validation split ({len(records)} incidents, "
            f"clip_val_frac={settings.clip_val_frac}). Lower SMOLVLM_FT_CLIP_VAL_FRAC."
        )
    if not wrote_val:
        logger.warning(
            "Validation split is empty (%d incidents x clip_val_frac=%g rounds to 0 per class): "
            "%s was not written. Training will run without evaluation; caption more incidents "
            "or raise SMOLVLM_FT_CLIP_VAL_FRAC to get one.",
            len(records), settings.clip_val_frac, val_path,
        )

    logger.info(
        "Wrote %s (%d incidents) and %s (%d incidents); skipped %d unusable manifest rows",
        train_path, len(train), val_path, len(val), skipped,
    )
    return train_path, val_path




def clip_to_row(clip: ClipRecord) -> dict:
    """Convert a clip to a HF chat-format row (works with datasets parquet).

    ``event_spans`` is empty for the Yes/No task and filled for the caption/timing
    task, where the collator uses it to concentrate frames inside the event.
    """
    return {
        "video_path": clip.video_path,
        "class_name": clip.class_name,
        "positive": clip.positive,
        "event_spans": list(clip.event_spans),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": clip.video_path},
                    {"type": "text", "text": clip.question},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": clip.answer}]},
        ],
    }


def resolve_video_path(example: dict, dcsass_root: Path) -> dict:
    """Rewrite a parquet-relative video path to an absolute one for this machine."""
    abs_path = str((dcsass_root / example["video_path"]).resolve())
    example["video_path"] = abs_path
    for item in example["messages"][0]["content"]:
        if item.get("type") == "video":
            item["path"] = abs_path
    return example


def load_clip_dataset(settings: Settings, split: str = "validation"):
    """Load one clip-task split and make its incident video paths absolute.

    Same shape as the Yes/No splits, but the paths are relative to
    ``clip_dataset_dir`` (the concatenated videos) rather than to the raw DCSASS
    root, so a separate resolver is used.

    The validation split is optional: ``build_clip_parquets`` deliberately does not
    write an empty one, so a missing validation file loads as an empty dataset
    instead of failing. A missing *train* file is a real error.
    """
    from datasets import Dataset, load_dataset

    validation = split == "validation"
    path = settings.val_clips_parquet if validation else settings.train_clips_parquet
    if not path.is_file():
        if validation:
            return Dataset.from_list([])
        raise FileNotFoundError(
            f"Missing clip dataset file: {path}. Run `smolvlm-ft build-clips`, "
            "`caption-clips` and `build-clip-data` first."
        )
    data = load_dataset("parquet", data_files={split: str(path)})[split]
    if len(data) == 0:
        return data
    return data.map(lambda ex: resolve_clip_video_path(ex, settings.clip_dataset_dir))
