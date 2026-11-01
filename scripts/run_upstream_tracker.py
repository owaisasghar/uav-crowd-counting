"""Run the uav-dot-track authors' own main.py (fp32 detector, CPU preprocessing) to check track_video.py against it.

Uses the same MXNet stub as track_video.py and moves the tracks to <runs>/MOT17-test/upstream_<name>/data/, so
evaluate_tracking.py scores both pipelines on the same videos. Options after the video go to main.py unchanged:

    external/uav-dot/.venv/bin/python scripts/run_upstream_tracker.py \
        data/raw/UP-COUNT-TRACK/test_videos/0035.mp4 full --track-cmc-flow --track-use-alt \
        --track-use-add-cls --track-use-pointflow
"""
from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path

from track_video import PROJECT_ROOT, TRACK_ROOT  # also installs the MXNet stub and the tracker's import paths


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    video, name, options = Path(sys.argv[1]).absolute(), sys.argv[2], sys.argv[3:]
    os.chdir(TRACK_ROOT)
    sys.argv = ["main.py", "--name", name, "--dataset", "upcount", "--video", str(video), "--task", "pred",
                "--device", "cuda", *options]
    try:
        runpy.run_path("main.py", run_name="__main__")
    except SystemExit as exit_code:  # click exits after the command
        if exit_code.code:
            raise
    target = PROJECT_ROOT / "runs" / "tracking" / "MOT17-test" / f"upstream_{name}" / "data"
    target.mkdir(parents=True, exist_ok=True)
    shutil.move(TRACK_ROOT / "outputs" / "preds" / "upcount" / "MOT17-test" / name / "data" / f"{video.stem}.txt",
                target / f"{video.stem}.txt")


if __name__ == "__main__":
    main()
