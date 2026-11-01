from __future__ import annotations

import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UAVDOT_ROOT = PROJECT_ROOT / "external" / "uav-dot"
sys.path.insert(0, str(UAVDOT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from src.datamodule.dataset.generic_dataset import GenericDataset  # noqa: E402
from src.datamodule.dataset.upcount_dataset import UpcountDataset  # noqa: E402
from scripts.p2p_visdrone_utils import (  # noqa: E402
    Sample,
    discover_samples,
    load_points,
    sequence_folds,
)

# UAV-Dot's DroneCrowd statistics (configs/dronecrowd.yaml), kept for every dataset so the
# trained model is normalised the same way as the released checkpoint.
DATA_MEAN = (0.38051389, 0.38728308, 0.37517634)
DATA_STD = (0.23425215, 0.2339074, 0.24194536)
IMAGE_SIZE = (1920, 1088)
MASK_SIZE = (960, 544)


class VisDroneDotDataset(GenericDataset):
    """VisDrone2020-CC frames in UAV-Dot's (image, keypoints) format, read from this repo's layout."""

    def __init__(self, samples: list[Sample], mosaic: float, transforms, normalize) -> None:
        super().__init__(None, None, IMAGE_SIZE, MASK_SIZE, mosaic, transforms, normalize)
        self.num_classes = 1
        self.samples = samples
        self.images_list = [sample.image for sample in samples]
        self.mask_generate = self.create_mask_generator(self.num_classes)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_data(self, index: int):
        sample = self.samples[index]
        frame = cv2.cvtColor(cv2.imread(str(sample.image), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        points = load_points(sample.annotation, sample.frame_index)
        points = np.clip(points, 0, [width - 1, height - 1]).astype(np.int64)
        labels = np.zeros((len(points), 5), dtype=np.int32)
        labels[:, 1:3] = points
        labels[:, 3:5] = 1
        return frame, labels, np.array([0.0], dtype=np.float32)


def train_augmentation(zoom_min: float, zoom_p: float) -> A.Compose:
    """UAV-Dot's training augmentation with a configurable zoom range.

    RandomResizedCrop keeps `scale` of the frame area and resizes it to the model input, so
    zoom_min=0.5 (upstream) enlarges people by at most 1.4x; zoom_min=0.16 allows 2.5x.
    """
    width, height = IMAGE_SIZE
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.OneOf([A.ISONoise(), A.GaussNoise()], p=0.5),
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.8, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
                    A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
                ],
                p=0.5,
            ),
            A.RandomGamma((40, 220), p=0.8),
            A.RandomResizedCrop(
                height=height, width=width, scale=(zoom_min, 1.0), ratio=(1.0, 1.0), p=zoom_p
            ),
            A.Resize(height=height, width=width, always_apply=True),
        ],
        keypoint_params=A.KeypointParams(
            format="xy", remove_invisible=True, label_fields=["labels", "shapes"]
        ),
    )


def eval_transforms() -> A.Compose:
    width, height = IMAGE_SIZE
    return A.Compose(
        [A.Resize(height=height, width=width, always_apply=True)],
        keypoint_params=A.KeypointParams(
            format="xy", remove_invisible=True, label_fields=["labels", "shapes"]
        ),
    )


def visdrone_fold(root: Path, fold: int, folds: int, seed: int) -> tuple[list[Sample], list[Sample]]:
    """Same sequence-level split as scripts/prepare_p2pnet_visdrone.py (annotated sequences only)."""
    samples = discover_samples(root)
    if not samples:
        raise RuntimeError(f"No annotated samples found below {root}")
    fold_map = sequence_folds(samples, folds, seed)
    train = [sample for sample in samples if fold_map[sample.sequence] != fold]
    val = [sample for sample in samples if fold_map[sample.sequence] == fold]
    return train, val


def upcount_split(root: Path, split: str) -> list[str]:
    sequences = (root / f"{split}.txt").read_text().split()
    return [str(path) for seq in sequences for path in sorted((root / "images" / seq).glob("*.jpg"))]


class UavDotDataModule(LightningDataModule):
    """Training on a VisDrone2020-CC fold (optionally plus UP-COUNT train); validation on the fold only."""

    def __init__(
        self,
        visdrone_root: Path,
        fold: int,
        folds: int,
        seed: int,
        upcount_root: Path | None,
        batch_size: int,
        workers: int,
        mosaic: float,
        zoom_min: float,
        zoom_p: float,
        upcount_share: float = 0.5,
        train_all: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        normalize = A.Normalize(mean=DATA_MEAN, std=DATA_STD, max_pixel_value=255.0)
        augmentation = train_augmentation(zoom_min, zoom_p)

        train, val = visdrone_fold(visdrone_root, fold, folds, seed)
        if train_all:
            # Final model: every annotated sequence trains, so the fold's val frames are seen and only a sanity log.
            train = train + val
        train_sets =[VisDroneDotDataset(train, mosaic, augmentation, normalize)]
        if upcount_root is not None:
            train_sets.append(
                UpcountDataset(
                    upcount_root,
                    upcount_split(upcount_root, "train"),
                    image_subdir="train",
                    image_size=IMAGE_SIZE,
                    mask_size=MASK_SIZE,
                    mosaic=mosaic,
                    transforms=augmentation,
                    normalize=normalize,
                )
            )
        self.train_dataset = ConcatDataset(train_sets)
        # An epoch is always one VisDrone fold's worth of samples, so runs with and without UP-COUNT share the
        # same steps, LR schedule and GPU time. With UP-COUNT, `upcount_share` of those samples come from it.
        self.epoch_samples = len(train)
        self.sampler = None
        if upcount_root is not None:
            n_visdrone, n_upcount = len(train), len(train_sets[1])
            weights = torch.cat([
                torch.full((n_visdrone,), (1.0 - upcount_share) / n_visdrone),
                torch.full((n_upcount,), upcount_share / n_upcount),
            ])
            self.sampler = WeightedRandomSampler(weights, self.epoch_samples, replacement=True)
        self.val_dataset = VisDroneDotDataset(val, 0.0, eval_transforms(), normalize)
        self.collate_fn = self.val_dataset.collate_fn
        self.summary = {
            "visdrone_train_frames": len(train),
            "visdrone_val_frames": len(val),
            "visdrone_val_sequences": sorted({sample.sequence for sample in val}),
            "upcount_train_frames": len(train_sets[1]) if upcount_root is not None else 0,
            "upcount_share": upcount_share if upcount_root is not None else 0.0,
            "epoch_samples": self.epoch_samples,
            "train_all": train_all,
        }

    def optimizer_steps_per_epoch(self, accumulate: int) -> int:
        return self.epoch_samples // (self.batch_size * accumulate)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=self.sampler is None,
            sampler=self.sampler,
            drop_last=True,
            persistent_workers=self.workers > 0,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            collate_fn=self.collate_fn,
        )
