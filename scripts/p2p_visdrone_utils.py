from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Sample:
    image: Path
    annotation: Path
    sequence: str
    frame_index: int


def _natural_key(path: Path) -> list[object]:
    return [
        int(value) if value.isdigit() else value.lower()
        for value in re.split(r"(\d+)", path.name)
    ]


def _annotation_candidates(image: Path, root: Path) -> list[Path]:
    stem = image.stem
    sequence_name = image.parent.name
    candidates = [
        image.with_suffix(".txt"),
        root / "annotations" / f"{sequence_name}.txt",
        image.parent.parent / "annotations" / f"{stem}.txt",
        root / "annotations" / image.parent.name / f"{stem}.txt",
    ]
    try:
        relative = image.relative_to(root)
        candidate = root / str(relative).replace("sequences", "annotations")
        candidates.append(candidate.with_suffix(".txt"))
    except ValueError:
        pass
    return candidates


def discover_samples(root: str | Path) -> list[Sample]:
    root = Path(root)
    images = sorted(
        (path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: (str(path.parent), _natural_key(path)),
    )
    samples: list[Sample] = []
    for image in images:
        annotation = next(
            (
                candidate
                for candidate in _annotation_candidates(image, root)
                if candidate.exists()
            ),
            None,
        )
        if annotation is None:
            continue
        sequence = str(image.parent.relative_to(root))
        frame_match = re.search(r"\d+", image.stem)
        frame_index = (
            int(frame_match.group()) if frame_match is not None else len(samples) + 1
        )
        samples.append(Sample(image, annotation, sequence, frame_index))
    return samples


@lru_cache(maxsize=256)
def _load_annotation_rows(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = [value for value in re.split(r"[,\s]+", line.strip()) if value]
        try:
            rows.append([float(value) for value in fields])
        except ValueError:
            continue
    width = max((len(row) for row in rows), default=0)
    return np.asarray(
        [row for row in rows if len(row) == width],
        dtype=np.float32,
    )


def load_points(path: Path, frame_index: int | None = None) -> np.ndarray:
    rows = _load_annotation_rows(path)
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if rows.shape[1] >= 3 and frame_index is not None:
        rows = rows[rows[:, 0].astype(int) == frame_index]
        return rows[:, 1:3].copy()
    return rows[:, :2].copy()


def sequence_folds(samples: list[Sample], folds: int, seed: int) -> dict[str, int]:
    sequences = sorted({sample.sequence for sample in samples})
    generator = np.random.default_rng(seed)
    generator.shuffle(sequences)
    return {
        sequence: index % folds
        for index, sequence in enumerate(sequences)
    }
