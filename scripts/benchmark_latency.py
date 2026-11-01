"""Measure per-frame GPU latency of UAV-Dot and P2PNet as deployed: fp16 weights, channels_last, GPU preprocessing.

Model latency covers resize/normalisation on the GPU, the forward pass and point extraction for one 1920x1080 test
frame already in GPU memory; end-to-end adds JPEG decoding and the upload. Medians over --frames test frames spread
across all test sequences, after --warmup frames. Runs in the UAV-Dot environment:

    external/uav-dot/.venv/bin/python scripts/benchmark_latency.py \
        uavdot=external/uav-dot/checkpoints/dot_pd_dronecrowd_51.00.ckpt p2pnet=huggingface/model.safetensors

Writes runs/eval/latency.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from evaluate_dronecrowd import DEFAULT_THRESHOLDS, frames_for
from uavdot_data import DATA_MEAN, DATA_STD, PROJECT_ROOT

from src.model.dot_regressor import DotRegressor  # noqa: E402  (path set up by uavdot_data)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normaliser(mean, std):
    mean = torch.tensor(mean, device="cuda").view(1, 3, 1, 1) * 255
    std = torch.tensor(std, device="cuda").view(1, 3, 1, 1) * 255
    return lambda x: (x - mean) / std


def uavdot_runner(checkpoint: Path):
    model = DotRegressor.load_from_checkpoint(str(checkpoint), map_location="cuda")
    model = model.eval().half().to(memory_format=torch.channels_last)
    width, height = model._image_size
    normalise = normaliser(DATA_MEAN, DATA_STD)

    def run(frame: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(frame.permute(2, 0, 1)[None].float(), size=(height, width), mode="bilinear",
                          align_corners=False)
        x = normalise(x).half().contiguous(memory_format=torch.channels_last)
        return model.postprocessing(model(x)[0].float(), thresh=DEFAULT_THRESHOLDS["uavdot"])[0]

    return run


def p2pnet_runner(weights: Path):
    sys.path.insert(0, str(PROJECT_ROOT / "huggingface"))
    from p2pnet_runtime import load_model

    model, _ = load_model(weights, "cuda")
    model = model.half().to(memory_format=torch.channels_last)
    normalise = normaliser(IMAGENET_MEAN, IMAGENET_STD)

    def run(frame: torch.Tensor) -> torch.Tensor:
        height, width = frame.shape[:2]
        # P2PNet needs both sides padded to a multiple of 128.
        x = F.pad(normalise(frame.permute(2, 0, 1)[None].float()), (0, -width % 128, 0, -height % 128))
        outputs = model(x.half().contiguous(memory_format=torch.channels_last))
        scores = torch.softmax(outputs["pred_logits"].float(), dim=-1)[0, :, 1]
        return outputs["pred_points"][0][scores >= DEFAULT_THRESHOLDS["p2pnet"]]

    return run


@torch.inference_mode()
def benchmark(run, paths: list[Path], warmup: int) -> dict[str, float]:
    model_ms, end_to_end_ms = [], []
    for i, path in enumerate(paths):
        torch.cuda.synchronize()
        started = time.perf_counter()
        frame = torch.from_numpy(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)).cuda()
        torch.cuda.synchronize()
        uploaded = time.perf_counter()
        run(frame)
        torch.cuda.synchronize()
        finished = time.perf_counter()
        if i >= warmup:
            model_ms.append((finished - uploaded) * 1000)
            end_to_end_ms.append((finished - started) * 1000)
    return {
        "model_ms": statistics.median(model_ms),
        "model_ms_p90": float(np.percentile(model_ms, 90)),
        "end_to_end_ms": statistics.median(end_to_end_ms),
        "frames": len(model_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", help="NAME=PATH (.ckpt: UAV-Dot, .safetensors: exported P2PNet)")
    parser.add_argument("--visdrone-root", type=Path, default=PROJECT_ROOT / "data/raw/VisDrone2020-CC")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "eval" / "latency.json")
    args = parser.parse_args()

    frames = frames_for("test", args.visdrone_root, None, 0, 5, 42)
    paths = [frames[i][2] for i in np.linspace(0, len(frames) - 1, args.warmup + args.frames).astype(int)]
    results = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "frame_size": "1920x1080",
    }
    for item in args.models:
        name, path = item.split("=", 1)
        runner = p2pnet_runner if path.endswith(".safetensors") else uavdot_runner
        results[name] = {"checkpoint": path, **benchmark(runner(Path(path)), paths, args.warmup)}
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print(f"{results['gpu']}, torch {results['torch']}, {args.frames} frames after {args.warmup} warm-up")
    print(f"{'name':<12}{'model ms':>10}{'p90':>8}{'end-to-end ms':>15}{'model FPS':>11}")
    for item in args.models:
        name = item.split("=", 1)[0]
        r = results[name]
        print(f"{name:<12}{r['model_ms']:10.1f}{r['model_ms_p90']:8.1f}{r['end_to_end_ms']:15.1f}{1000 / r['model_ms']:11.1f}")


if __name__ == "__main__":
    main()
