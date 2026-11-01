"""Evaluate UAV-Dot and P2PNet on VisDrone2020-CC frames with the official DroneCrowd localisation protocol.

Per model: count MAE/RMSE, F1 at 16 px (one-to-one Hungarian matching) and L-mAP (DroneCrowd VID toolkit,
AP averaged over 1..25 px) on the evaluated frames. The model type follows the file: a .ckpt is a UAV-Dot
checkpoint, run as in UAV-Dot's pt_pred.py (frame resized to the model input, points above 0.2 with sigmoid
confidence, fp16 + channels_last); a .safetensors file is the exported P2PNet (huggingface/p2pnet_runtime.py,
fp16, points above 0.5). --threshold overrides both deployment thresholds. UAV-Dot inputs are normalised with the
DroneCrowd statistics every model here was trained with; --uavdot-stats upcount suits the released UP-COUNT model.
Ground truth is the toolkit's head boxes (our frame j is DroneCrowd frame 10*(j-1)); on the fold-0 val frames it
matches data/raw/VisDrone2020-CC/annotations exactly.

--split upcount-val scores the UP-COUNT val frames (4K, 28-101 m altitude) instead, also per altitude band, to check
scale coverage without touching the VisDrone test set. Its points and the predictions are scaled to 1920x1080, so
pixel radii mean the same as on VisDrone.

Runs in the UAV-Dot environment:

    external/uav-dot/.venv/bin/python scripts/evaluate_dronecrowd.py --split test \
        uavdot=external/uav-dot/checkpoints/dot_pd_dronecrowd_51.00.ckpt \
        p2pnet=huggingface/model.safetensors

Writes runs/eval/<split>_metrics.json (one entry per name) and runs/eval/<split>/<name>.npz
({"<seq>/<frame>": [N, 3] x, y, confidence in 1920x1080 frame coordinates}).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import io
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from uavdot_data import DATA_MEAN, DATA_STD, PROJECT_ROOT, UAVDOT_ROOT, visdrone_fold

from src.model.dot_regressor import DotRegressor  # noqa: E402  (path set up by uavdot_data)

TOOLKIT = UAVDOT_ROOT / "eval_tool"
FRAME_SIZE = (1920, 1080)
ALTITUDE_BANDS = {"<40m": (0, 40), "40-80m": (40, 80), ">=80m": (80, 1000)}
DEFAULT_THRESHOLDS = {"uavdot": 0.2, "p2pnet": 0.5}
# UAV-Dot input statistics (mean, std) from its configs/dronecrowd.yaml and configs/upcount.yaml.
UAVDOT_STATS = {
    "dronecrowd": (DATA_MEAN, DATA_STD),
    "upcount": ((0.44356189, 0.44594052, 0.41271554), (0.21697577, 0.20454077, 0.21697334)),
}


def load_toolkit_map():
    # The toolkit's package is also called `src`, so load map.py by path instead of importing it.
    spec = importlib.util.spec_from_file_location("dronecrowd_map", TOOLKIT / "src" / "map.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mAP


def frames_for(split: str, root: Path, upcount_root: Path, fold: int, folds: int, seed: int) -> list[tuple[str, int, Path]]:
    """(sequence, frame number, image path) for fold-0 val, the official test sequences or UP-COUNT val."""
    if split == "val":
        _, val = visdrone_fold(root, fold, folds, seed)
        return [(s.image.parent.name, s.frame_index, s.image) for s in val]
    if split == "upcount-val":
        sequences = (upcount_root / "val.txt").read_text().split()
        # <seq>__<frame>__<altitude>.jpg
        return [(seq, int(path.stem.split("__")[1]), path)
                for seq in sequences for path in sorted((upcount_root / "images" / seq).glob("*.jpg"))]
    sequences = (root / "testlist.txt").read_text().split()
    return [(seq, int(path.stem), path) for seq in sequences for path in sorted((root / "sequences" / seq).glob("*.jpg"))]


def altitude(path: Path) -> float:
    return float(path.stem.split("__")[-1])


def ground_truth(split: str, frames: list[tuple[str, int, Path]]) -> dict[tuple[str, int], np.ndarray]:
    """Toolkit annotation rows [id, x1, y1, x2, y2] for each evaluated frame."""
    gt, cache = {}, {}
    if split == "upcount-val":
        # Point labels as zero-size boxes (the toolkit matches box centres), scaled to FRAME_SIZE.
        for seq, j, path in frames:
            label = Path(str(path).replace("images", "labels")).with_suffix(".txt")
            points = np.loadtxt(label, dtype=np.float64, ndmin=2).reshape(-1, 2)
            points *= np.divide(FRAME_SIZE, Image.open(path).size)
            gt[(seq, j)] = np.column_stack([np.arange(len(points)), points, points])
        return gt
    for seq, j, _ in frames:
        if seq not in cache:
            cache[seq] = io.loadmat(str(TOOLKIT / "gt" / "dronecrowd" / f"{int(seq):05d}.mat"))["anno"]
        anno = cache[seq]
        gt[(seq, j)] = anno[anno[:, 0] == 10 * (j - 1), 1:6].astype(np.float64)
    return gt


@torch.inference_mode()
def predict_uavdot(checkpoint: Path, frames: list[tuple[str, int, Path]], threshold: float,
                   stats: str = "dronecrowd") -> dict[tuple[str, int], np.ndarray]:
    model = DotRegressor.load_from_checkpoint(str(checkpoint), map_location="cuda")
    model = model.eval().half().to(memory_format=torch.channels_last)
    width, height = model._image_size
    mean = torch.tensor(UAVDOT_STATS[stats][0], device="cuda").view(1, 3, 1, 1) * 255
    std = torch.tensor(UAVDOT_STATS[stats][1], device="cuda").view(1, 3, 1, 1) * 255
    predictions = {}
    for seq, j, path in frames:
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float()
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        x = ((x - mean) / std).half().contiguous(memory_format=torch.channels_last)
        points = model.postprocessing(model(x)[0].float(), thresh=threshold)[0]
        predictions[(seq, j)] = torch.stack([
            points[:, 1] * FRAME_SIZE[0] / width,
            points[:, 2] * FRAME_SIZE[1] / height,
            torch.sigmoid(points[:, 3]),
        ], 1).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return predictions


@torch.inference_mode()
def predict_p2pnet(weights: Path, frames: list[tuple[str, int, Path]], threshold: float) -> dict[tuple[str, int], np.ndarray]:
    sys.path.insert(0, str(PROJECT_ROOT / "huggingface"))
    from p2pnet_runtime import load_model, predict_points

    model, device = load_model(weights, "cuda")
    predictions = {}
    for seq, j, path in frames:
        image = Image.open(path)
        result = predict_points(model, image, device, threshold=threshold, max_side=FRAME_SIZE[0], use_half=True)
        points = result["points"].reshape(-1, 2) * np.divide(FRAME_SIZE, image.size)
        predictions[(seq, j)] = np.column_stack([points, result["scores"]])
    del model
    torch.cuda.empty_cache()
    return predictions


def score(predictions, gt, map_fn, radius: float) -> dict[str, float]:
    errors, tp, n_pred, n_gt = [], 0, 0, 0
    all_gt, all_det = {}, {}
    # Per sequence: sums from which any group of sequences gets its exact MAE, precision, recall and F1.
    by_sequence = {}
    for (seq, j), rows in gt.items():
        pred = predictions[(seq, j)]
        centres = np.stack([(rows[:, 1] + rows[:, 3]) / 2, (rows[:, 2] + rows[:, 4]) / 2], 1)
        errors.append(len(pred) - len(centres))
        n_pred += len(pred)
        n_gt += len(centres)
        matched = 0
        if len(pred) and len(centres):
            distances = cdist(pred[:, :2], centres)
            # Pairs beyond the radius get a prohibitive cost, so the assignment maximises matches within it
            # (plain distances would trade close matches for a lower total over forced far pairs).
            r, c = linear_sum_assignment(np.where(distances <= radius, distances, 1e6))
            matched = int((distances[r, c] <= radius).sum())
        tp += matched
        sums = by_sequence.setdefault(seq, {"frames": 0, "people": 0, "predicted": 0, "matched": 0, "abs_error": 0})
        sums["frames"] += 1
        sums["people"] += len(centres)
        sums["predicted"] += len(pred)
        sums["matched"] += matched
        sums["abs_error"] += abs(len(pred) - len(centres))
        # Toolkit layout (src/save_anno_res.py): gt [frame, id, x1, y1, x2, y2, ignore], det [frame, -1, x-10, y-10, 20, 20, conf]
        all_gt.setdefault(seq, []).append(np.column_stack([np.full(len(rows), j), rows, np.zeros(len(rows))]))
        all_det.setdefault(seq, []).append(np.column_stack([
            np.full(len(pred), j), np.full(len(pred), -1), pred[:, 0] - 10, pred[:, 1] - 10,
            np.full(len(pred), 20), np.full(len(pred), 20), pred[:, 2],
        ]))
    all_gt = {int(seq): np.concatenate(v) for seq, v in all_gt.items()}
    all_det = {int(seq): np.concatenate(v) for seq, v in all_det.items()}
    thresholds = np.arange(1, 26, 1)
    ap = map_fn(all_gt, all_det, thresh_range=thresholds, threads=8)
    errors = np.array(errors, dtype=np.float64)
    precision, recall = tp / max(n_pred, 1), tp / max(n_gt, 1)
    return {
        "frames": len(errors),
        "people_per_frame": n_gt / len(errors),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "bias": float(errors.mean()),
        f"f1@{radius:g}": 2 * precision * recall / max(precision + recall, 1e-9),
        "precision": precision,
        "recall": recall,
        "l_map": float(ap.get(thresholds)),
        "ap@10": float(ap.get([10])),
        "by_sequence": by_sequence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="+",
                        help="NAME=PATH (.ckpt: UAV-Dot, .safetensors: exported P2PNet, .npz: saved predictions)")
    parser.add_argument("--split", choices=["val", "test", "upcount-val"], default="val")
    parser.add_argument("--visdrone-root", type=Path, default=PROJECT_ROOT / "data/raw/VisDrone2020-CC")
    parser.add_argument("--upcount-root", type=Path, default=PROJECT_ROOT / "data/raw/UP-COUNT")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uavdot-stats", choices=sorted(UAVDOT_STATS), default="dronecrowd",
                        help="input normalisation for UAV-Dot checkpoints")
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"confidence threshold for every model; default per model {DEFAULT_THRESHOLDS}")
    parser.add_argument("--radius", type=float, default=16.0, help="match distance for F1, in pixels")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "eval")
    args = parser.parse_args()

    frames = frames_for(args.split, args.visdrone_root, args.upcount_root, args.fold, args.folds, args.seed)
    gt = ground_truth(args.split, frames)
    map_fn = load_toolkit_map()
    (args.out / args.split).mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / f"{args.split}_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    for item in args.checkpoints:
        name, path = item.split("=", 1)
        if path.endswith(".npz"):
            # Rescore predictions this script saved earlier (already thresholded), keeping the entry's model details.
            saved = np.load(path)
            predictions = {(seq, j): saved[f"{seq}/{j:05d}"] for seq, j, _ in frames}
            details = {key: metrics.get(name, {}).get(key) for key in ("checkpoint", "model", "threshold")}
        else:
            kind = "p2pnet" if path.endswith(".safetensors") else "uavdot"
            threshold = DEFAULT_THRESHOLDS[kind] if args.threshold is None else args.threshold
            if kind == "p2pnet":
                predictions = predict_p2pnet(Path(path), frames, threshold)
            else:
                predictions = predict_uavdot(Path(path), frames, threshold, args.uavdot_stats)
            np.savez_compressed(args.out / args.split / f"{name}.npz",
                                **{f"{seq}/{j:05d}": p.astype(np.float32) for (seq, j), p in predictions.items()})
            details = {"checkpoint": path, "model": kind, "threshold": threshold}
            if kind == "uavdot":
                details["normalisation"] = args.uavdot_stats
        metrics[name] = {**details, **score(predictions, gt, map_fn, args.radius)}
        if args.split == "upcount-val":
            paths = {(seq, j): path for seq, j, path in frames}
            metrics[name]["by_altitude"] = {
                band: score(predictions, {k: v for k, v in gt.items() if low <= altitude(paths[k]) < high}, map_fn, args.radius)
                for band, (low, high) in ALTITUDE_BANDS.items()
            }
        metrics_path.write_text(json.dumps(metrics, indent=2))

    f1_key = f"f1@{args.radius:g}"
    print(f"{args.split}: {len(frames)} frames, {next(iter(metrics.values()))['people_per_frame']:.1f} people per frame")
    print(f"{'name':<32}{'MAE':>8}{'RMSE':>8}{'bias':>8}{'F1@' + format(args.radius, 'g'):>8}{'P':>7}{'R':>7}{'L-mAP':>8}")
    for name, m in metrics.items():
        rows = [(name, m)] + [(f"  {band} ({b['frames']} frames)", b) for band, b in m.get("by_altitude", {}).items()]
        for label, r in rows:
            print(f"{label:<32}{r['mae']:8.2f}{r['rmse']:8.2f}{r['bias']:8.2f}{r[f1_key]:8.3f}"
                  f"{r['precision']:7.3f}{r['recall']:7.3f}{r['l_map']:8.2f}")


if __name__ == "__main__":
    main()
