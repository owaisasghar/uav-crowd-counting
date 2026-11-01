"""Standalone inference runtime for the released UAV-Dot DOT + Pixel Distill DroneCrowd model.

The network definition (h_sigmoid ... DotModel) is adapted from UAV-Dot, src/model/dot_model.py at commit
a6d3664b50cc0cc6f1a08848eb294b71753b6600 (https://github.com/up-count/uav-dot), licensed under the Apache License 2.0
(LICENSE-uav-dot.txt). Changes: inference only, the 'pixel' input mode only, no ImageNet download, and the
segmentation-models-pytorch 0.5 `use_norm` argument in place of `use_batchnorm`. Pre- and postprocessing follow
UAV-Dot's DotRegressor and pt_pred.py.

The weights are not redistributed: download_checkpoint() fetches them from the authors' public mirror and checks
their SHA-256 before loading.
"""
from __future__ import annotations

import hashlib
import pickle
import time
import types
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.encoders import get_encoder

CHECKPOINT_NAME = "dot_pd_dronecrowd_51.00.ckpt"
CHECKPOINT_URL = f"https://chmura.put.poznan.pl/public.php/webdav/{CHECKPOINT_NAME}"
CHECKPOINT_SHARE = "SGlQ3OFfgRJr86g"  # public share token, sent as the WebDAV user name
CHECKPOINT_SHA256 = "8d37e4a09334d10915bf8b239409d6e25d9066283e4df9be464c88d776ef0e23"
IMAGE_SIZE = (1920, 1088)
MASK_SIZE = (960, 544)
DATA_MEAN = (0.38051389, 0.38728308, 0.37517634)
DATA_STD = (0.23425215, 0.2339074, 0.24194536)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = md.Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1,
                                   use_norm="batchnorm")
        self.attention1 = md.Attention(None, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_norm="batchnorm")
        self.attention2 = md.Attention(None, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.attention2(x)


class UnetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        # Drop the full-resolution skip and start from the encoder head.
        encoder_channels = encoder_channels[1:][::-1]
        in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        self.center = nn.Identity()
        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, decoder_channels)
        ])

    def forward(self, *features):
        features = features[1:][::-1]
        x = self.center(features[0])
        skips = features[1:]
        out = []
        for i, decoder_block in enumerate(self.blocks):
            x = decoder_block(x, skips[i] if i < len(skips) else None)
            out.append(x)
        return out[::-1]


class PixelDistillBlock(nn.Module):
    def __init__(self, in_channels, features_channels, scale_down=1):
        super(PixelDistillBlock, self).__init__()
        if scale_down > 1:
            self.down = nn.PixelUnshuffle(scale_down)
            in_channels = int(in_channels * scale_down**2)
        else:
            self.down = nn.Identity()
        self.conv = nn.Conv2d(in_channels, features_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(features_channels)
        self.relu = nn.ReLU()
        self.attn = CoordAtt(inp=features_channels, oup=features_channels)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(self.down(x))))
        return self.attn(x)


class PixelDistill(nn.Module):
    def __init__(self, in_channels, out_channels, features_channels, down_scale=1):
        super(PixelDistill, self).__init__()
        self.down_reduction = int(2**2)
        self.unshuffle = nn.PixelUnshuffle(2)
        self.blocks = nn.ModuleList([
            PixelDistillBlock(in_channels, features_channels // 2, scale_down=down_scale // 2)
            for _ in range(self.down_reduction)
        ])
        self.conv_1 = nn.Conv2d(int(self.down_reduction * (features_channels // 2)), features_channels, 3, 1, 1)
        self.bn_1 = nn.BatchNorm2d(features_channels)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(features_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.unshuffle(x)
        outs = [block(x[:, i::self.down_reduction]) for i, block in enumerate(self.blocks)]
        x = self.relu(self.bn_1(self.conv_1(torch.cat(outs, dim=1))))
        return self.conv_2(x)


class DotModel(nn.Module):
    def __init__(self, encoder_name="mit_b2", classes=1, image_size=IMAGE_SIZE, mask_size=MASK_SIZE):
        super().__init__()
        decoder_channels = (256, 128, 64, 32, 16)
        self.encoder = get_encoder(encoder_name, in_channels=3, depth=5, weights=None)
        self.decoder = UnetDecoder(self.encoder.out_channels, decoder_channels)
        self.segmentation_head = nn.ModuleList([
            SegmentationHead(in_channels=channel, out_channels=classes, activation=None, kernel_size=3)
            for channel in decoder_channels[::-1]
        ])
        self.pre = PixelDistill(3, 3, features_channels=32, down_scale=int(image_size[0] / mask_size[0]))

    def forward(self, x):
        decoder_output = self.decoder(*self.encoder(self.pre(x)))
        # Only the full-resolution head is used for points.
        return self.segmentation_head[0](decoder_output[0])


def download_checkpoint(directory: str | Path) -> Path:
    """Fetch the released checkpoint from the authors' mirror once, refusing any file with a different SHA-256."""
    path = Path(directory) / CHECKPOINT_NAME
    if path.exists() and _sha256(path) == CHECKPOINT_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, CHECKPOINT_URL, CHECKPOINT_SHARE, "")
    opener = urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(manager))
    with opener.open(CHECKPOINT_URL) as response, partial.open("wb") as out:
        while chunk := response.read(8 * 2**20):
            out.write(chunk)
    if _sha256(partial) != CHECKPOINT_SHA256:
        partial.unlink()
        raise RuntimeError(f"{CHECKPOINT_NAME} from {CHECKPOINT_URL} does not match the expected SHA-256")
    partial.rename(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 2**20):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "cpu" or not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cpu")
    return torch.device("cuda")


class _Inert:
    """Stands in for any non-tensor class in the checkpoint (Hydra configs, paths, callback state)."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        pass


class _WeightsUnpickler(pickle.Unpickler):
    """Builds tensors and plain containers only, so loading the weights needs no Lightning or Hydra classes."""

    SAFE_BUILTINS = {"set", "frozenset", "slice", "range", "complex", "bytearray"}

    def find_class(self, module: str, name: str):
        if module.startswith(("torch", "collections", "numpy", "_codecs", "copyreg")):
            return super().find_class(module, name)
        if module == "builtins" and name in self.SAFE_BUILTINS:
            return super().find_class(module, name)
        return _Inert


def load_model(checkpoint_path: str | Path, device: str | torch.device = "auto") -> tuple[DotModel, torch.device]:
    resolved = resolve_device(device) if isinstance(device, str) else device
    weights_pickle = types.SimpleNamespace(Unpickler=_WeightsUnpickler, load=pickle.load, __name__="pickle")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False, pickle_module=weights_pickle)
    state = checkpoint["state_dict"]
    network = {key.removeprefix("network."): value for key, value in state.items() if key.startswith("network.")}
    model = DotModel()
    model.load_state_dict(network, strict=True)
    return model.to(resolved).eval(), resolved


def predict_points(
    model: DotModel,
    image: Image.Image,
    device: torch.device,
    threshold: float = 0.2,
    use_half: bool = False,
) -> dict[str, object]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    image = image.convert("RGB")
    source_width, source_height = image.size
    half = use_half and device.type == "cuda"
    dtype = torch.float16 if half else torch.float32
    mean = torch.tensor(DATA_MEAN, device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor(DATA_STD, device=device).view(1, 3, 1, 1) * 255
    tensor = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None].float()
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        x = F.interpolate(tensor, size=(IMAGE_SIZE[1], IMAGE_SIZE[0]), mode="bilinear", align_corners=False)
        x = ((x - mean) / std).to(dtype)
        logits = model.to(dtype)(x).float()
        logits = F.interpolate(logits, size=(MASK_SIZE[1], MASK_SIZE[0]), mode="bilinear", align_corners=True)
        heat = torch.sigmoid(logits)[0, 0]
        peaks = heat * (F.max_pool2d(heat[None, None], 3, stride=1, padding=1)[0, 0] == heat)
        yx = torch.argwhere(peaks > threshold)
        scores = heat[yx[:, 0], yx[:, 1]]
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - started) * 1000.0
    # Mask cell -> model input pixel (as DotRegressor.postprocessing) -> source image pixel.
    points = yx[:, [1, 0]].float().cpu().numpy()
    points[:, 0] *= IMAGE_SIZE[0] / MASK_SIZE[0] * source_width / IMAGE_SIZE[0]
    points[:, 1] *= IMAGE_SIZE[1] / MASK_SIZE[1] * source_height / IMAGE_SIZE[1]
    return {
        "points": points,
        "scores": scores.cpu().numpy(),
        "count": int(len(points)),
        "inference_ms": inference_ms,
        "device": str(device),
        "scale": IMAGE_SIZE[0] / source_width,
    }


def annotate_image(image: Image.Image, points: np.ndarray, threshold: float) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for x, y in points:
        center_x, center_y = int(round(float(x))), int(round(float(y)))
        draw.ellipse((center_x - 5, center_y - 5, center_x + 5, center_y + 5), fill=(0, 0, 0))
        draw.ellipse((center_x - 3, center_y - 3, center_x + 3, center_y + 3), fill=(35, 200, 255))
    label = f"UAV-Dot predicted heads: {len(points)}   threshold={threshold:.2f}"
    font = ImageFont.load_default(size=22)
    text_box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, min(output.width, text_box[2] - text_box[0] + 28), text_box[3] - text_box[1] + 22),
                   fill=(0, 0, 0))
    draw.text((14, 10), label, fill=(255, 255, 255), font=font)
    return output


def serializable_result(result: dict[str, object]) -> dict[str, object]:
    return {
        "predicted_head_count": result["count"],
        "inference_ms": round(float(result["inference_ms"]), 3),
        "device": result["device"],
        "input_scale": result["scale"],
        "points": [
            {"x": float(point[0]), "y": float(point[1]), "confidence": float(score)}
            for point, score in zip(result["points"], result["scores"])
        ],
    }
