from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
P2PNET_ROOT = PROJECT_ROOT / "third_party" / "CrowdCounting-P2PNet"
sys.path.insert(0, str(P2PNET_ROOT))

from models import build_model  # noqa: E402


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.empty(1, device=device)
            return device
    except (RuntimeError, AssertionError) as error:
        if requested == "cuda":
            raise RuntimeError(f"CUDA was requested but initialization failed: {error}")
        print(f"CUDA initialization failed; using CPU: {error}", flush=True)
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cpu")


def _prepare_frame(
    frame: np.ndarray,
    max_side: int,
) -> tuple[torch.Tensor, float, int, int]:
    source_height, source_width = frame.shape[:2]
    scale = min(1.0, max_side / max(source_height, source_width))
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    tensor = (tensor - MEAN) / STD
    padded_height = ((resized_height + 127) // 128) * 128
    padded_width = ((resized_width + 127) // 128) * 128
    tensor = F.pad(
        tensor,
        (0, padded_width - resized_width, 0, padded_height - resized_height),
    )
    return tensor.unsqueeze(0), scale, resized_width, resized_height


def _predict(
    model: torch.nn.Module,
    frame: np.ndarray,
    device: torch.device,
    threshold: float,
    max_side: int,
    use_half: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    tensor, scale, resized_width, resized_height = _prepare_frame(frame, max_side)
    tensor = tensor.to(device, non_blocking=device.type == "cuda")
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
        & (points[:, 0] < resized_width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < resized_height)
    )
    predicted_points = points[keep].cpu().numpy() / scale
    predicted_scores = scores[keep].cpu().numpy()
    return predicted_points, predicted_scores, inference_ms


def _draw_points(
    frame: np.ndarray,
    points: np.ndarray,
    count: int,
    threshold: float,
    update_age_frames: int,
    fps: float,
) -> np.ndarray:
    result = frame.copy()
    for x, y in points:
        center = (int(round(x)), int(round(y)))
        cv2.circle(result, center, 5, (0, 0, 0), -1)
        cv2.circle(result, center, 3, (0, 0, 255), -1)
    age_seconds = update_age_frames / max(fps, 1e-6)
    cv2.rectangle(result, (0, 0), (720, 72), (0, 0, 0), -1)
    cv2.putText(
        result,
        f"P2PNet predicted heads: {count}",
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        f"threshold={threshold:.2f}  prediction age={age_seconds:.2f}s",
        (14, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run P2PNet head-point inference on a video."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--inference-fps",
        type=float,
        default=5.0,
        help="Fresh model updates per source-video second.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1920,
        help="Maximum model-input side; coordinates are scaled back to the video.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("The video has invalid FPS or dimensions")
    frame_step = max(1, int(round(fps / args.inference_fps)))

    device = _resolve_device(args.device)
    model_args = SimpleNamespace(
        backbone="vgg16_bn",
        row=2,
        line=2,
        pretrained_backbone=False,
    )
    model = build_model(model_args).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {args.output}")

    model_updates: list[dict[str, object]] = []
    latest_points = np.empty((0, 2), dtype=np.float32)
    latest_scores = np.empty((0,), dtype=np.float32)
    latest_update_frame = 0
    frame_index = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            if frame_index % frame_step == 0:
                latest_points, latest_scores, inference_ms = _predict(
                    model,
                    frame,
                    device,
                    args.threshold,
                    args.max_side,
                    args.half,
                )
                latest_update_frame = frame_index
                model_updates.append(
                    {
                        "frame": frame_index,
                        "time_seconds": frame_index / fps,
                        "predicted_head_count": int(len(latest_points)),
                        "inference_ms": inference_ms,
                        "points": [
                            {
                                "x": float(point[0]),
                                "y": float(point[1]),
                                "confidence": float(score),
                            }
                            for point, score in zip(latest_points, latest_scores)
                        ],
                    }
                )
            annotated = _draw_points(
                frame,
                latest_points,
                len(latest_points),
                args.threshold,
                frame_index - latest_update_frame,
                fps,
            )
            writer.write(annotated)
            frame_index += 1
            if frame_index % max(1, int(round(fps))) == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"frames={frame_index}/{total_frames} "
                    f"updates={len(model_updates)} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - started
    counts = [int(update["predicted_head_count"]) for update in model_updates]
    inference_times = [float(update["inference_ms"]) for update in model_updates]
    summary_path = args.summary or args.output.with_suffix(".json")
    summary = {
        "model": "P2PNet",
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "output": str(args.output),
        "device": str(device),
        "source_width": width,
        "source_height": height,
        "source_fps": fps,
        "source_frames": total_frames,
        "processed_frames": frame_index,
        "model_input_max_side": args.max_side,
        "confidence_threshold": args.threshold,
        "requested_inference_fps": args.inference_fps,
        "frame_step": frame_step,
        "model_updates": len(model_updates),
        "mean_predicted_count": float(np.mean(counts)) if counts else 0.0,
        "minimum_predicted_count": min(counts) if counts else 0,
        "maximum_predicted_count": max(counts) if counts else 0,
        "mean_model_inference_ms": (
            float(np.mean(inference_times)) if inference_times else 0.0
        ),
        "processing_seconds": elapsed,
        "processing_fps": frame_index / elapsed if elapsed else 0.0,
        "updates": model_updates,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "updates"}, indent=2))


if __name__ == "__main__":
    main()
