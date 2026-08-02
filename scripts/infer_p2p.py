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
import torchvision.transforms as transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
P2PNET_ROOT = PROJECT_ROOT / "third_party" / "CrowdCounting-P2PNet"
sys.path.insert(0, str(P2PNET_ROOT))

from models import build_model  # noqa: E402
from scripts.p2p_visdrone_utils import (  # noqa: E402
    discover_samples,
    load_points,
    sequence_folds,
)


def _draw_points(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    result = image.copy()
    for x, y in points:
        cv2.circle(result, (int(round(x)), int(round(y))), 4, (0, 0, 0), -1)
        cv2.circle(result, (int(round(x)), int(round(y))), 2, color, -1)
    cv2.rectangle(result, (0, 0), (min(result.shape[1], 620), 48), (0, 0, 0), -1)
    cv2.putText(
        result,
        f"{label}: {len(points)}",
        (12, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="True point-proposal inference using P2PNet."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=P2PNET_ROOT / "weights" / "SHTechA.pth",
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/VisDrone2020-CC"))
    parser.add_argument("--validation-index", type=int, default=122)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("inference/p2pnet"))
    args = parser.parse_args()

    samples = discover_samples(args.dataset_root)
    by_path = {sample.image.resolve(): sample for sample in samples}
    if args.input is None:
        fold_map = sequence_folds(samples, args.folds, args.seed)
        validation = [
            sample for sample in samples if fold_map[sample.sequence] == args.fold
        ]
        sample = validation[args.validation_index % len(validation)]
        image_path = sample.image
    else:
        image_path = args.input
        sample = by_path.get(image_path.resolve())

    raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw_bgr is None:
        raise RuntimeError(f"Could not read {image_path}")
    height, width = raw_bgr.shape[:2]
    pil_image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    tensor = transform(pil_image).unsqueeze(0)
    padded_height = ((height + 127) // 128) * 128
    padded_width = ((width + 127) // 128) * 128
    tensor = F.pad(tensor, (0, padded_width - width, 0, padded_height - height))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    tensor = tensor.to(device)

    with torch.inference_mode():
        for _ in range(3):
            outputs = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - start) * 1000.0

    scores = torch.softmax(outputs["pred_logits"], dim=-1)[0, :, 1]
    points = outputs["pred_points"][0]
    keep = (
        (scores >= args.threshold)
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    predicted_points = points[keep].cpu().numpy()
    predicted_scores = scores[keep].cpu().numpy()
    ground_truth = (
        load_points(sample.annotation, sample.frame_index)
        if sample is not None
        else np.empty((0, 2), dtype=np.float32)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{image_path.parent.name}_{image_path.stem}"
    prediction_path = args.output_dir / f"{stem}_predicted_points.jpg"
    comparison_path = args.output_dir / f"{stem}_comparison.jpg"
    json_path = args.output_dir / f"{stem}_points.json"
    prediction_image = _draw_points(
        raw_bgr, predicted_points, (0, 0, 255), "P2PNet predicted heads"
    )
    if not cv2.imwrite(str(prediction_path), prediction_image):
        raise RuntimeError(f"Could not write {prediction_path}")
    if sample is not None:
        ground_truth_image = _draw_points(
            raw_bgr, ground_truth, (0, 255, 0), "Ground-truth heads"
        )
        comparison = np.concatenate([prediction_image, ground_truth_image], axis=1)
        if not cv2.imwrite(str(comparison_path), comparison):
            raise RuntimeError(f"Could not write {comparison_path}")

    result = {
        "model": "P2PNet",
        "checkpoint": str(args.checkpoint),
        "checkpoint_training_domain": "ShanghaiTech Part A"
        if args.checkpoint.name == "SHTechA.pth"
        else "VisDrone fine-tuned",
        "input": str(image_path),
        "confidence_threshold": args.threshold,
        "predicted_head_count": int(len(predicted_points)),
        "ground_truth_head_count": int(len(ground_truth)) if sample is not None else None,
        "absolute_count_error": (
            abs(int(len(predicted_points)) - int(len(ground_truth)))
            if sample is not None
            else None
        ),
        "inference_ms": inference_ms,
        "device": str(device),
        "prediction_image": str(prediction_path),
        "comparison_image": str(comparison_path) if sample is not None else None,
        "points": [
            {"x": float(point[0]), "y": float(point[1]), "confidence": float(score)}
            for point, score in zip(predicted_points, predicted_scores)
        ],
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "points"}, indent=2))


if __name__ == "__main__":
    main()
