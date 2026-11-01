"""Draw the README charts in assets/ (a light and a dark version of each) from the evaluation outputs.

Reads runs/eval/test_metrics.json, runs/eval/upcount-val_metrics.json (evaluate_dronecrowd.py) and
runs/eval/latency.json (benchmark_latency.py):

    .venv/bin/python scripts/generate_portfolio_charts.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Validated categorical palette (light and dark steps of the same three hues) and chart ink.
THEMES = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "primary": "#f0efec", "secondary": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70"]},
}
# Colour follows the model in every chart.
MODELS = {
    "released_dronecrowd": ("UAV-Dot (released weights)", 0),
    "final": ("UAV-Dot + UP-COUNT fine-tune", 1),
    "p2pnet": ("P2PNet (trained here)", 2),
}
SCENE_GROUPS = {
    "Low altitude\n(00017, 00018)": ["00017", "00018"],
    "Night\n(00069, 00070, 00074)": ["00069", "00070", "00074"],
}


def style_axes(ax, theme: dict, value_axis: str) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
    ax.tick_params(colors=theme["secondary"], labelsize=9, length=0)
    ax.grid(axis=value_axis, color=theme["grid"], linewidth=1)
    ax.set_axisbelow(True)


def group_f1(by_sequence: dict, sequences: list[str]) -> float:
    matched = sum(by_sequence[s]["matched"] for s in sequences)
    predicted = sum(by_sequence[s]["predicted"] for s in sequences)
    people = sum(by_sequence[s]["people"] for s in sequences)
    return 2 * matched / max(predicted + people, 1)


def speed_accuracy(theme: dict, test: dict, latency: dict, path: Path) -> None:
    rows = [("released_dronecrowd", "uavdot"), ("p2pnet", "p2pnet")]
    panels = [
        ("Latency per 1080p frame (ms)\nlower is better", [latency[k]["model_ms"] for _, k in rows], "{:.0f} ms"),
        ("Count error, MAE (people)\nlower is better", [test[m]["mae"] for m, _ in rows], "{:.1f}"),
        ("Localisation, L-mAP\nhigher is better", [test[m]["l_map"] for m, _ in rows], "{:.1f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.4), facecolor=theme["surface"])
    labels = [MODELS[m][0] for m, _ in rows]
    colours = [theme["series"][MODELS[m][1]] for m, _ in rows]
    for i, (ax, (title, values, fmt)) in enumerate(zip(axes, panels)):
        style_axes(ax, theme, "x")
        y = np.arange(len(rows))[::-1]
        ax.barh(y, values, height=0.42, color=colours)
        ax.set_yticks(y, labels if i == 0 else [""] * len(rows), color=theme["primary"], fontsize=9.5)
        ax.set_xlim(0, max(values) * 1.3)
        for yi, value in zip(y, values):
            ax.text(value + max(values) * 0.03, yi, fmt.format(value), va="center", color=theme["primary"], fontsize=9.5)
        ax.set_title(title, color=theme["secondary"], fontsize=9.5, loc="left")
    fig.suptitle("VisDrone2020-CC test set (900 frames), RTX 4060 Laptop GPU, fp16", color=theme["primary"],
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=theme["surface"])
    plt.close(fig)


def upcount_experiment(theme: dict, test: dict, upcount: dict, path: Path) -> None:
    models = list(MODELS)
    bands = list(upcount["released_dronecrowd"]["by_altitude"])
    left = {m: [upcount[m]["by_altitude"][b]["l_map"] for b in bands] for m in models}
    all_test = sorted(test["released_dronecrowd"]["by_sequence"])
    groups = dict(SCENE_GROUPS)
    groups["Other 25 test\nsequences"] = [s for s in all_test if s not in sum(SCENE_GROUPS.values(), [])]
    right = {m: [group_f1(test[m]["by_sequence"], seqs) for seqs in groups.values()] for m in models}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), facecolor=theme["surface"])
    panels = [
        (axes[0], left, [f"{b.replace('>=', '≥')}" for b in bands], 80, "UP-COUNT val: L-mAP by flying height"),
        (axes[1], right, list(groups), 1.0, "VisDrone test: F1@16px by scene type"),
    ]
    width = 0.24
    for ax, values, ticks, top, title in panels:
        style_axes(ax, theme, "y")
        x = np.arange(len(ticks))
        for i, m in enumerate(models):
            ax.bar(x + (i - 1) * width, values[m], width=width, color=theme["series"][MODELS[m][1]],
                   edgecolor=theme["surface"], linewidth=1.5, label=MODELS[m][0])
        ax.set_xticks(x, ticks, color=theme["primary"], fontsize=9)
        ax.set_ylim(0, top)
        ax.set_title(title, color=theme["primary"], fontsize=10.5, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=3, frameon=False, fontsize=9, labelcolor=theme["secondary"],
               bbox_to_anchor=(0.005, 1.0), handlelength=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=160, facecolor=theme["surface"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval", type=Path, default=PROJECT_ROOT / "runs" / "eval")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "assets")
    args = parser.parse_args()

    test = json.loads((args.eval / "test_metrics.json").read_text())
    upcount = json.loads((args.eval / "upcount-val_metrics.json").read_text())
    latency = json.loads((args.eval / "latency.json").read_text())
    for mode, theme in THEMES.items():
        suffix = "" if mode == "light" else "_dark"
        speed_accuracy(theme, test, latency, args.out / f"speed_accuracy{suffix}.png")
        upcount_experiment(theme, test, upcount, args.out / f"upcount_experiment{suffix}.png")
    print("wrote", ", ".join(sorted(p.name for p in args.out.glob("*experiment*.png")) +
                             sorted(p.name for p in args.out.glob("speed_accuracy*.png"))))


if __name__ == "__main__":
    main()
