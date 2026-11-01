from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


COLORS = [
    (31, 119, 180),
    (214, 39, 40),
    (44, 160, 44),
    (148, 103, 189),
]


def _draw_plot(
    title: str,
    x_values: list[float],
    series: list[tuple[str, list[float]]],
    width: int = 900,
    height: int = 560,
    x_label: str = "Epoch",
) -> np.ndarray:
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    left, right, top, bottom = 90, width - 35, 75, height - 75
    cv2.putText(canvas, title, (left, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (25, 25, 25), 2)
    all_y = [value for _, values in series for value in values if np.isfinite(value)]
    y_min = min(all_y) if all_y else 0.0
    y_max = max(all_y) if all_y else 1.0
    padding = max((y_max - y_min) * 0.08, 1e-9)
    y_min -= padding
    y_max += padding
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_max += 1

    for tick in range(6):
        fraction = tick / 5
        y = int(round(bottom - fraction * (bottom - top)))
        value = y_min + fraction * (y_max - y_min)
        cv2.line(canvas, (left, y), (right, y), (225, 225, 225), 1)
        cv2.putText(
            canvas,
            f"{value:.3g}",
            (10, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (70, 70, 70),
            1,
        )
    cv2.line(canvas, (left, top), (left, bottom), (50, 50, 50), 2)
    cv2.line(canvas, (left, bottom), (right, bottom), (50, 50, 50), 2)
    for tick in range(6):
        fraction = tick / 5
        x = int(round(left + fraction * (right - left)))
        value = x_min + fraction * (x_max - x_min)
        cv2.putText(
            canvas,
            f"{value:.2g}",
            (x - 14, bottom + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (70, 70, 70),
            1,
        )
    cv2.putText(
        canvas,
        x_label,
        ((left + right) // 2 - 24, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (50, 50, 50),
        1,
    )

    legend_x = left
    for series_index, (name, values) in enumerate(series):
        color = COLORS[series_index % len(COLORS)]
        pixels = []
        for x_value, y_value in zip(x_values, values):
            x = int(round(left + (x_value - x_min) / (x_max - x_min) * (right - left)))
            y = int(round(bottom - (y_value - y_min) / (y_max - y_min) * (bottom - top)))
            pixels.append((x, y))
        if len(pixels) >= 2:
            cv2.polylines(canvas, [np.asarray(pixels, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        for pixel in pixels:
            cv2.circle(canvas, pixel, 2, color, -1)
        cv2.line(canvas, (legend_x, 58), (legend_x + 22, 58), color, 3)
        cv2.putText(
            canvas,
            name,
            (legend_x + 28, 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (45, 45, 45),
            1,
        )
        legend_x += 45 + len(name) * 9
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate portfolio metric plots.")
    parser.add_argument(
        "--training-metrics",
        type=Path,
        default=Path("runs/p2pnet_visdrone_fold0/log/metrics.jsonl"),
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("reports/fold0_localization.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    args = parser.parse_args()

    training = [
        json.loads(line)
        for line in args.training_metrics.read_text().splitlines()
        if line.strip()
    ]
    evaluation = json.loads(args.evaluation.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [float(row["epoch"]) for row in training]
    loss_plot = _draw_plot(
        "P2PNet training losses",
        epochs,
        [
            ("total", [row["train_loss"] for row in training]),
            ("classification", [row["train_classification_loss"] for row in training]),
            ("point", [row["train_point_loss"] for row in training]),
        ],
    )
    validation = [row for row in training if "validation_mae" in row]
    validation_plot = _draw_plot(
        "Validation count error (lower is better)",
        [float(row["epoch"]) for row in validation],
        [
            ("MAE", [row["validation_mae"] for row in validation]),
            ("RMSE", [row["validation_rmse"] for row in validation]),
        ],
    )
    training_canvas = np.concatenate([loss_plot, validation_plot], axis=1)
    cv2.imwrite(str(args.output_dir / "training_curves.png"), training_canvas)

    threshold_results = evaluation["threshold_results"]
    thresholds = [float(row["threshold"]) for row in threshold_results]
    count_plot = _draw_plot(
        "Count error vs confidence",
        thresholds,
        [
            ("bounded MAE", [row["bounded_count"]["mae"] for row in threshold_results]),
            ("bounded RMSE", [row["bounded_count"]["rmse"] for row in threshold_results]),
        ],
        x_label="Confidence threshold",
    )
    radius_keys = sorted(
        threshold_results[0]["localization"],
        key=lambda value: float(value),
    )
    f1_plot = _draw_plot(
        "Localization F1 vs confidence",
        thresholds,
        [
            (
                f"F1@{radius}px",
                [row["localization"][radius]["f1"] for row in threshold_results],
            )
            for radius in radius_keys
        ],
        x_label="Confidence threshold",
    )
    threshold_canvas = np.concatenate([count_plot, f1_plot], axis=1)
    cv2.imwrite(str(args.output_dir / "threshold_sweep.png"), threshold_canvas)
    print(args.output_dir / "training_curves.png")
    print(args.output_dir / "threshold_sweep.png")


if __name__ == "__main__":
    main()
