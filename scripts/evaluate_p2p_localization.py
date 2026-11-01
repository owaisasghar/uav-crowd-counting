from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
P2PNET_ROOT = PROJECT_ROOT / "third_party" / "CrowdCounting-P2PNet"
sys.path.insert(0, str(P2PNET_ROOT))

from models import build_model  # noqa: E402
from util.misc import nested_tensor_from_tensor_list  # noqa: E402

from scripts.p2p_metrics import (  # noqa: E402
    count_summary,
    interpolated_average_precision,
    localization_summary,
)


def _read_points(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2:
            rows.append([float(fields[0]), float(fields[1])])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 2)


def _read_manifest(root: Path, name: str) -> list[tuple[Path, Path]]:
    samples = []
    for line in (root / name).read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2:
            samples.append((root / fields[0], root / fields[1]))
    return samples


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        try:
            torch.empty(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError:
            if requested == "cuda":
                raise
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate P2PNet count and point-localization metrics."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/p2pnet_visdrone"),
    )
    parser.add_argument("--manifest", default="val.list")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--proposal-min-score", type=float, default=0.001)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--radii", type=float, nargs="+", default=[8, 16, 32])
    args = parser.parse_args()

    thresholds = sorted(set(float(value) for value in args.thresholds))
    radii = sorted(set(float(value) for value in args.radii))
    if min(thresholds) < args.proposal_min_score:
        raise ValueError("proposal-min-score cannot exceed the smallest threshold")

    samples = _read_manifest(args.data_root, args.manifest)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError("No evaluation samples were found")

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
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    cached_predictions: list[dict[str, object]] = []
    targets: list[np.ndarray] = []
    started = time.perf_counter()
    for index, (image_path, points_path) in enumerate(samples):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        tensor = transform(image)
        tensor = nested_tensor_from_tensor_list([tensor]).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - inference_started) * 1000.0

        scores = torch.softmax(outputs["pred_logits"].float(), dim=-1)[0, :, 1]
        points = outputs["pred_points"][0].float()
        retained = scores > args.proposal_min_score
        scores = scores[retained].cpu().numpy()
        points = points[retained].cpu().numpy()
        in_bounds = (
            (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )
        cached_predictions.append(
            {
                "image": str(image_path),
                "points": points,
                "scores": scores,
                "in_bounds": in_bounds,
                "inference_ms": inference_ms,
            }
        )
        targets.append(_read_points(points_path))
        if (index + 1) % 25 == 0 or index + 1 == len(samples):
            elapsed = time.perf_counter() - started
            print(
                f"evaluated={index + 1}/{len(samples)} elapsed={elapsed:.1f}s",
                flush=True,
            )

    target_counts = [len(points) for points in targets]
    threshold_results = []
    for threshold in thresholds:
        bounded_points = []
        bounded_counts = []
        legacy_counts = []
        per_image = []
        for prediction, target in zip(cached_predictions, targets):
            scores = np.asarray(prediction["scores"])
            points = np.asarray(prediction["points"])
            in_bounds = np.asarray(prediction["in_bounds"])
            above = scores > threshold
            selected = points[above & in_bounds]
            bounded_points.append(selected)
            bounded_counts.append(len(selected))
            legacy_counts.append(int(np.count_nonzero(above)))
            per_image.append(
                {
                    "image": prediction["image"],
                    "predicted_count": len(selected),
                    "target_count": len(target),
                    "absolute_count_error": abs(len(selected) - len(target)),
                }
            )
        localization = {
            str(int(radius) if radius.is_integer() else radius): localization_summary(
                bounded_points,
                targets,
                radius,
            )
            for radius in radii
        }
        threshold_results.append(
            {
                "threshold": threshold,
                "bounded_count": count_summary(bounded_counts, target_counts),
                "legacy_padded_count": count_summary(legacy_counts, target_counts),
                "localization": localization,
                "per_image": per_image,
            }
        )

    scored_predictions = []
    for image_index, prediction in enumerate(cached_predictions):
        points = np.asarray(prediction["points"])
        scores = np.asarray(prediction["scores"])
        in_bounds = np.asarray(prediction["in_bounds"])
        scored_predictions.extend(
            (image_index, float(score), point)
            for point, score in zip(points[in_bounds], scores[in_bounds])
        )
    ranked_localization = {
        str(int(radius) if radius.is_integer() else radius): {
            "radius_pixels": radius,
            "minimum_proposal_score": args.proposal_min_score,
            **interpolated_average_precision(
                scored_predictions,
                targets,
                radius,
            ),
        }
        for radius in radii
    }

    best_count = min(
        threshold_results,
        key=lambda result: result["bounded_count"]["mae"],
    )
    best_f1 = {}
    for radius in radii:
        key = str(int(radius) if radius.is_integer() else radius)
        candidate = max(
            threshold_results,
            key=lambda result: result["localization"][key]["f1"],
        )
        best_f1[key] = {
            "threshold": candidate["threshold"],
            **candidate["localization"][key],
        }

    result = {
        "model": "P2PNet VisDrone fold 0",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch_zero_based": checkpoint.get("epoch"),
        "manifest": str(args.data_root / args.manifest),
        "device": str(device),
        "images": len(samples),
        "ground_truth_points": int(sum(target_counts)),
        "proposal_minimum_score": args.proposal_min_score,
        "matching": (
            "one-to-one Hungarian matching; prediction is correct when Euclidean "
            "distance is at most the stated pixel radius"
        ),
        "count_protocols": {
            "bounded_count": "deployment behavior; predictions outside the source image are removed",
            "legacy_padded_count": "training-log behavior; predictions in padded regions are counted",
        },
        "calibration_warning": (
            "Thresholds are evaluated on the same validation fold used for checkpoint "
            "selection and must not be treated as independent-test estimates."
        ),
        "mean_inference_ms": float(
            np.mean([prediction["inference_ms"] for prediction in cached_predictions])
        ),
        "evaluation_seconds": time.perf_counter() - started,
        "threshold_results": threshold_results,
        "ranked_localization": ranked_localization,
        "best_bounded_count_threshold": {
            "threshold": best_count["threshold"],
            **best_count["bounded_count"],
        },
        "best_f1_threshold_by_radius": best_f1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    compact = {key: value for key, value in result.items() if key != "threshold_results"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
