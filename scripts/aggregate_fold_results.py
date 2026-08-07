from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate fixed-protocol P2PNet fold results.")
    parser.add_argument("evaluations", type=Path, nargs="+")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--radius", type=float, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    documents = [json.loads(path.read_text()) for path in args.evaluations]
    radius_key = str(int(args.radius) if args.radius.is_integer() else args.radius)
    selected = []
    for path, document in zip(args.evaluations, documents):
        threshold_result = next(
            (
                row
                for row in document["threshold_results"]
                if np.isclose(row["threshold"], args.threshold)
            ),
            None,
        )
        if threshold_result is None:
            raise ValueError(f"{path} does not contain threshold {args.threshold}")
        selected.append(
            {
                "path": str(path),
                "images": document["images"],
                "count": threshold_result["bounded_count"],
                "localization": threshold_result["localization"][radius_key],
                "ranked_localization": document["ranked_localization"][radius_key],
            }
        )

    result = {
        "folds": len(selected),
        "fixed_confidence_threshold": args.threshold,
        "localization_radius_pixels": args.radius,
        "count_mae": _mean_std([row["count"]["mae"] for row in selected]),
        "count_rmse": _mean_std([row["count"]["rmse"] for row in selected]),
        "localization_precision": _mean_std(
            [row["localization"]["precision"] for row in selected]
        ),
        "localization_recall": _mean_std(
            [row["localization"]["recall"] for row in selected]
        ),
        "localization_f1": _mean_std([row["localization"]["f1"] for row in selected]),
        "localization_ap": _mean_std(
            [row["ranked_localization"]["average_precision"] for row in selected]
        ),
        "per_fold": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
