"""Standalone inference runtime for the P2PNet VisDrone checkpoint.

This module intentionally contains only the architecture and deployment path.
Training, matching, and dataset code remain in the main GitHub repository.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from torch import nn
from torchvision.models import vgg16_bn


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class VGGBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        features = list(vgg16_bn(weights=None).features.children())
        self.body1 = nn.Sequential(*features[:13])
        self.body2 = nn.Sequential(*features[13:23])
        self.body3 = nn.Sequential(*features[23:33])
        self.body4 = nn.Sequential(*features[33:43])

    def forward(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for layer in (self.body1, self.body2, self.body3, self.body4):
            tensor = layer(tensor)
            outputs.append(tensor)
        return outputs


class RegressionModel(nn.Module):
    def __init__(
        self,
        num_features_in: int,
        num_anchor_points: int = 4,
        feature_size: int = 256,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, 3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, 3, padding=1)
        self.act2 = nn.ReLU()
        # Retained for exact checkpoint compatibility with upstream P2PNet.
        self.conv3 = nn.Conv2d(feature_size, feature_size, 3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, 3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, 3, padding=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.act1(self.conv1(tensor))
        tensor = self.act2(self.conv2(tensor))
        tensor = self.output(tensor).permute(0, 2, 3, 1)
        return tensor.contiguous().view(tensor.shape[0], -1, 2)


class ClassificationModel(nn.Module):
    def __init__(
        self,
        num_features_in: int,
        num_anchor_points: int = 4,
        num_classes: int = 2,
        feature_size: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points
        self.conv1 = nn.Conv2d(num_features_in, feature_size, 3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, 3, padding=1)
        self.act2 = nn.ReLU()
        # Retained for exact checkpoint compatibility with upstream P2PNet.
        self.conv3 = nn.Conv2d(feature_size, feature_size, 3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, 3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(
            feature_size,
            num_anchor_points * num_classes,
            3,
            padding=1,
        )
        self.output_act = nn.Sigmoid()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.act1(self.conv1(tensor))
        tensor = self.act2(self.conv2(tensor))
        tensor = self.output(tensor).permute(0, 2, 3, 1)
        batch, height, width, _ = tensor.shape
        tensor = tensor.view(
            batch,
            height,
            width,
            self.num_anchor_points,
            self.num_classes,
        )
        return tensor.contiguous().view(batch, -1, self.num_classes)


class Decoder(nn.Module):
    def __init__(
        self,
        c3_size: int = 256,
        c4_size: int = 512,
        c5_size: int = 512,
        feature_size: int = 256,
    ) -> None:
        super().__init__()
        self.P5_1 = nn.Conv2d(c5_size, feature_size, 1)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P5_2 = nn.Conv2d(feature_size, feature_size, 3, padding=1)
        self.P4_1 = nn.Conv2d(c4_size, feature_size, 1)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P4_2 = nn.Conv2d(feature_size, feature_size, 3, padding=1)
        self.P3_1 = nn.Conv2d(c3_size, feature_size, 1)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P3_2 = nn.Conv2d(feature_size, feature_size, 3, padding=1)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        c3, c4, c5 = inputs
        p5 = self.P5_1(c5)
        p5_upsampled = self.P5_upsampled(p5)
        p5 = self.P5_2(p5)
        p4 = self.P4_1(c4) + p5_upsampled
        p4_upsampled = self.P4_upsampled(p4)
        p4 = self.P4_2(p4)
        p3 = self.P3_1(c3) + p4_upsampled
        p3 = self.P3_2(p3)
        return [p3, p4, p5]


def _anchor_points(
    image: torch.Tensor,
    stride: int = 8,
    rows: int = 2,
    columns: int = 2,
) -> torch.Tensor:
    height = (image.shape[-2] + stride - 1) // stride
    width = (image.shape[-1] + stride - 1) // stride
    dtype = image.dtype
    device = image.device

    base_x = (
        (torch.arange(columns, device=device, dtype=dtype) + 0.5)
        * (stride / columns)
        - stride / 2
    )
    base_y = (
        (torch.arange(rows, device=device, dtype=dtype) + 0.5)
        * (stride / rows)
        - stride / 2
    )
    base_y_grid, base_x_grid = torch.meshgrid(base_y, base_x, indexing="ij")
    base = torch.stack((base_x_grid.flatten(), base_y_grid.flatten()), dim=1)

    shift_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride
    shift_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride
    shift_y_grid, shift_x_grid = torch.meshgrid(shift_y, shift_x, indexing="ij")
    shifts = torch.stack((shift_x_grid.flatten(), shift_y_grid.flatten()), dim=1)
    return (shifts[:, None, :] + base[None, :, :]).reshape(1, -1, 2)


class P2PNet(nn.Module):
    def __init__(self, rows: int = 2, columns: int = 2) -> None:
        super().__init__()
        self.backbone = VGGBackbone()
        num_anchor_points = rows * columns
        self.regression = RegressionModel(256, num_anchor_points)
        self.classification = ClassificationModel(256, num_anchor_points, 2)
        self.anchor_points = nn.Identity()
        self.fpn = Decoder()
        self.rows = rows
        self.columns = columns

    def forward(self, samples: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(samples)
        features_fpn = self.fpn([features[1], features[2], features[3]])
        regression = self.regression(features_fpn[1]) * 100
        classification = self.classification(features_fpn[1])
        anchors = _anchor_points(samples, rows=self.rows, columns=self.columns)
        anchors = anchors.repeat(samples.shape[0], 1, 1)
        return {
            "pred_logits": classification,
            "pred_points": regression + anchors,
        }


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested not in {"auto", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.empty(1, device=device)
            return device
    except (RuntimeError, AssertionError):
        if requested == "cuda":
            raise
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cpu")


def load_model(
    weights_path: str | Path,
    device: str | torch.device = "auto",
) -> tuple[P2PNet, torch.device]:
    resolved = resolve_device(device) if isinstance(device, str) else device
    model = P2PNet()
    state_dict = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(resolved).eval()
    return model, resolved


def prepare_image(
    image: Image.Image,
    max_side: int = 1920,
) -> tuple[torch.Tensor, float, int, int]:
    image = image.convert("RGB")
    source_width, source_height = image.size
    scale = min(1.0, max_side / max(source_width, source_height))
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    if scale < 1.0:
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    padded_height = ((height + 127) // 128) * 128
    padded_width = ((width + 127) // 128) * 128
    tensor = F.pad(tensor, (0, padded_width - width, 0, padded_height - height))
    return tensor.unsqueeze(0), scale, width, height


def predict_points(
    model: P2PNet,
    image: Image.Image,
    device: torch.device,
    threshold: float = 0.5,
    max_side: int = 1920,
    use_half: bool = False,
) -> dict[str, object]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    tensor, scale, width, height = prepare_image(image, max_side=max_side)
    tensor = tensor.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_half and device.type == "cuda",
        ):
            outputs = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - started) * 1000.0

    scores = torch.softmax(outputs["pred_logits"].float(), dim=-1)[0, :, 1]
    points = outputs["pred_points"][0].float()
    keep = (
        (scores >= threshold)
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    selected_points = points[keep].cpu().numpy() / scale
    selected_scores = scores[keep].cpu().numpy()
    return {
        "points": selected_points,
        "scores": selected_scores,
        "count": int(len(selected_points)),
        "inference_ms": inference_ms,
        "device": str(device),
        "scale": scale,
    }


def annotate_image(
    image: Image.Image,
    points: np.ndarray,
    threshold: float,
) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for x, y in points:
        center_x = int(round(float(x)))
        center_y = int(round(float(y)))
        draw.ellipse(
            (center_x - 5, center_y - 5, center_x + 5, center_y + 5),
            fill=(0, 0, 0),
        )
        draw.ellipse(
            (center_x - 3, center_y - 3, center_x + 3, center_y + 3),
            fill=(255, 35, 35),
        )
    label = f"P2PNet predicted heads: {len(points)}   threshold={threshold:.2f}"
    font = ImageFont.load_default(size=22)
    text_box = draw.textbbox((0, 0), label, font=font)
    box_width = min(output.width, text_box[2] - text_box[0] + 28)
    box_height = text_box[3] - text_box[1] + 22
    draw.rectangle((0, 0, box_width, box_height), fill=(0, 0, 0))
    draw.text((14, 10), label, fill=(255, 255, 255), font=font)
    return output


def serializable_result(result: dict[str, object]) -> dict[str, object]:
    points = result["points"]
    scores = result["scores"]
    return {
        "predicted_head_count": result["count"],
        "inference_ms": round(float(result["inference_ms"]), 3),
        "device": result["device"],
        "input_scale": result["scale"],
        "points": [
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "confidence": float(score),
            }
            for point, score in zip(points, scores)
        ],
    }
