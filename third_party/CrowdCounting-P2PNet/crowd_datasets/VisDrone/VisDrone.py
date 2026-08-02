from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def _read_points(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2:
            rows.append([float(fields[0]), float(fields[1])])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 2)


class VisDronePoints(Dataset):
    def __init__(
        self,
        data_root,
        transform,
        train=False,
        patch_size=512,
        num_patches=2,
        flip=True,
    ):
        self.root = Path(data_root)
        self.transform = transform
        self.train = train
        self.patch_size = int(patch_size)
        self.num_patches = int(num_patches)
        self.flip = flip
        list_path = self.root / ("train.list" if train else "val.list")
        self.samples = []
        for line in list_path.read_text().splitlines():
            fields = line.split()
            if len(fields) >= 2:
                self.samples.append((self.root / fields[0], self.root / fields[1]))

    def __len__(self):
        return len(self.samples)

    def _crop(self, image, points):
        _, height, width = image.shape
        size = self.patch_size
        images, point_sets = [], []
        for _ in range(self.num_patches):
            if len(points) and random.random() < 0.8:
                center_x, center_y = points[random.randrange(len(points))]
                jitter_x = random.uniform(-0.25, 0.25) * size
                jitter_y = random.uniform(-0.25, 0.25) * size
                left = int(round(center_x + jitter_x - size / 2))
                top = int(round(center_y + jitter_y - size / 2))
                left = min(max(0, left), width - size)
                top = min(max(0, top), height - size)
            else:
                left = random.randint(0, width - size)
                top = random.randint(0, height - size)
            right, bottom = left + size, top + size
            keep = (
                (points[:, 0] >= left)
                & (points[:, 0] < right)
                & (points[:, 1] >= top)
                & (points[:, 1] < bottom)
            )
            crop_points = points[keep].copy()
            crop_points[:, 0] -= left
            crop_points[:, 1] -= top
            images.append(image[:, top:bottom, left:right])
            point_sets.append(crop_points)
        return torch.stack(images), point_sets

    def __getitem__(self, index):
        image_path, points_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        points = _read_points(points_path)
        image = self.transform(image)

        if self.train:
            scale = random.uniform(0.8, 1.2)
            old_height, old_width = image.shape[1:]
            new_height = max(self.patch_size, int(round(old_height * scale)))
            new_width = max(self.patch_size, int(round(old_width * scale)))
            image = F.interpolate(
                image.unsqueeze(0),
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            points[:, 0] *= new_width / old_width
            points[:, 1] *= new_height / old_height
            if self.flip and random.random() < 0.5:
                image = torch.flip(image, dims=[2])
                points[:, 0] = new_width - 1 - points[:, 0]
            images, point_sets = self._crop(image, points)
        else:
            images, point_sets = image, [points]

        targets = []
        for points_for_image in point_sets:
            targets.append(
                {
                    "point": torch.as_tensor(points_for_image, dtype=torch.float32),
                    "image_id": torch.tensor([index], dtype=torch.long),
                    "labels": torch.ones(len(points_for_image), dtype=torch.long),
                }
            )
        return images, targets
