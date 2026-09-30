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
"""

from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass(frozen=True)
class ClipRecord:
    """One fine-tuning example: a video clip plus its derived conversation."""

    video_path: str
    class_name: str
    positive: bool
    question: str
    answer: str


@dataclass
class DcsassDataset:
    """In-memory view of the raw DCSASS layout, built once from disk."""

    root: Path
    clips: list[ClipRecord] = field(default_factory=list)

    @classmethod
    def from_disk(cls, root: Path) -> "DcsassDataset":
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
                n_val = int(round(len(bucket) * val_frac))
                val.extend(bucket[:n_val])
                train.extend(bucket[n_val:])
        rng.shuffle(train)
        rng.shuffle(val)
        logger.info("Split: %d train / %d val", len(train), len(val))
        return train, val


def clip_to_row(clip: ClipRecord) -> dict:
    """Convert a clip to a HF chat-format row (works with datasets parquet)."""
    return {
        "video_path": clip.video_path,
        "class_name": clip.class_name,
        "positive": clip.positive,
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