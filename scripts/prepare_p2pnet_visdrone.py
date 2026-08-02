from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.p2p_visdrone_utils import discover_samples, load_points, sequence_folds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VisDrone sequence annotations to P2PNet point files."
    )
    parser.add_argument("--root", type=Path, default=Path("data/raw/VisDrone2020-CC"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/p2pnet_visdrone")
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = discover_samples(args.root)
    if not samples:
        raise RuntimeError(f"No annotated samples found below {args.root}")
    fold_map = sequence_folds(samples, args.folds, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    list_rows = {"train": [], "val": []}
    point_totals = {"train": 0, "val": 0}

    for sample in samples:
        points = load_points(sample.annotation, sample.frame_index)
        point_path = (
            args.output
            / "points"
            / Path(sample.sequence).name
            / f"{sample.image.stem}.txt"
        )
        point_path.parent.mkdir(parents=True, exist_ok=True)
        point_path.write_text(
            "".join(f"{float(x):.4f} {float(y):.4f}\n" for x, y in points)
        )
        split = "val" if fold_map[sample.sequence] == args.fold else "train"
        image_relative = os.path.relpath(sample.image.resolve(), args.output.resolve())
        point_relative = os.path.relpath(point_path.resolve(), args.output.resolve())
        list_rows[split].append(f"{image_relative} {point_relative}\n")
        point_totals[split] += len(points)

    for split, rows in list_rows.items():
        (args.output / f"{split}.list").write_text("".join(rows))
    summary = {
        "source": str(args.root),
        "fold": args.fold,
        "folds": args.folds,
        "seed": args.seed,
        "train_frames": len(list_rows["train"]),
        "validation_frames": len(list_rows["val"]),
        "train_points": point_totals["train"],
        "validation_points": point_totals["val"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
