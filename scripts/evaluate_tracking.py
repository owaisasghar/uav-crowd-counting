"""Score tracker runs on UP-COUNT-TRACK with TrackEval (HOTA, CLEAR, Identity) using the uav-dot-track settings.

A run is runs/tracking/MOT17-test/<name>/data/<video>.txt from track_video.py. Only videos present in every compared
run are scored, against data/raw/UP-COUNT-TRACK/mot_style_test_labels. As in the authors' eval/track_eval.py:
20x20 px boxes around each point, no preprocessing, similarity threshold 0. Also reports the unique-person count
error (predicted vs ground-truth identities per video; the paper's Tr-MAE / Tr-nMAE) and the pipeline timings.
Runs in the UAV-Dot environment:

    external/uav-dot/.venv/bin/python scripts/evaluate_tracking.py baseline cmc cmc+alt cmc+alt+cls full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "uav-dot-track" / "eval" / "TrackEval"))
import trackeval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trackers", nargs="+", help="run names under <runs>/MOT17-test/")
    parser.add_argument("--runs", type=Path, default=PROJECT_ROOT / "runs" / "tracking")
    parser.add_argument("--videos", nargs="*", default=None, help="score only these videos (default: all common ones)")
    parser.add_argument("--out", type=Path, default=None, help="metrics JSON (default: <runs>/metrics.json)")
    parser.add_argument("--gt", type=Path, default=PROJECT_ROOT / "data/raw/UP-COUNT-TRACK/mot_style_test_labels/gt")
    args = parser.parse_args()

    videos = sorted(set.intersection(*[
        {path.stem for path in (args.runs / "MOT17-test" / name / "data").glob("*.txt")} for name in args.trackers
    ]))
    if args.videos:
        videos = [v for v in videos if v in args.videos]
    seqmap = args.runs / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(videos) + "\n")

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(DISPLAY_LESS_PROGRESS=True, PRINT_RESULTS=False, OUTPUT_SUMMARY=False, OUTPUT_EMPTY_CLASSES=False,
                       OUTPUT_DETAILED=False, PLOT_CURVES=False, PRINT_CONFIG=False, TIME_PROGRESS=False)
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(GT_FOLDER=str(args.gt), TRACKERS_FOLDER=str(args.runs), BENCHMARK="MOT17",
                          SPLIT_TO_EVAL="test", TRACKERS_TO_EVAL=args.trackers, SEQMAP_FILE=str(seqmap),
                          DO_PREPROC=False, PRINT_CONFIG=False)
    metrics_config = {"THRESHOLD": 0.0, "PRINT_CONFIG": False}
    metrics = [trackeval.metrics.HOTA(metrics_config), trackeval.metrics.CLEAR(metrics_config),
               trackeval.metrics.Identity(metrics_config)]
    results, _ = trackeval.Evaluator(eval_config).evaluate([trackeval.datasets.MotChallenge2DBox(dataset_config)], metrics)

    summary = {"videos": videos}
    for name in args.trackers:
        per_video = results["MotChallenge2DBox"][name]
        combined = per_video["COMBINED_SEQ"]["pedestrian"]
        count_errors = [(per_video[v]["pedestrian"]["Count"]["IDs"], per_video[v]["pedestrian"]["Count"]["GT_IDs"])
                        for v in videos]
        # Runs made outside track_video.py (e.g. the authors' main.py) have no timing files.
        timing_files = [args.runs / "timing" / name / f"{v}.json" for v in videos]
        timings = [json.loads(path.read_text()) for path in timing_files if path.exists()]
        summary[name] = {
            "hota": float(np.mean(combined["HOTA"]["HOTA"])),
            "mota": float(combined["CLEAR"]["MOTA"]),
            "idf1": float(combined["Identity"]["IDF1"]),
            "id_switches": int(combined["CLEAR"]["IDSW"]),
            "track_count_mae": float(np.mean([abs(p - g) for p, g in count_errors])),
            "track_count_nmae": float(np.mean([abs(p - g) / g for p, g in count_errors])),
            "identities": {v: {"predicted": p, "ground_truth": g} for v, (p, g) in zip(videos, count_errors)},
            "detect_ms": float(np.mean([t["median_detect_ms"] for t in timings])) if timings else None,
            "track_ms": float(np.mean([t["median_track_ms"] for t in timings])) if timings else None,
        }
    (args.out or args.runs / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"{len(videos)} videos: {' '.join(videos)}")
    print(f"{'tracker':<16}{'HOTA':>7}{'MOTA':>7}{'IDF1':>7}{'ID-SW':>7}{'Tr-MAE':>8}{'Tr-nMAE':>9}"
          f"{'det ms':>8}{'trk ms':>8}{'FPS':>6}")
    for name in args.trackers:
        s = summary[name]
        timing = (f"{s['detect_ms']:8.1f}{s['track_ms']:8.1f}{1000 / (s['detect_ms'] + s['track_ms']):6.1f}"
                  if s["detect_ms"] is not None else f"{'-':>8}{'-':>8}{'-':>6}")
        print(f"{name:<16}{s['hota']:7.3f}{s['mota']:7.3f}{s['idf1']:7.3f}{s['id_switches']:7d}{s['track_count_mae']:8.1f}"
              f"{s['track_count_nmae']:9.2f}{timing}")


if __name__ == "__main__":
    main()
