#!/usr/bin/env bash
# Tracker ablation on the downloaded UP-COUNT-TRACK test videos: the uav-dot-track variants from the paper's table,
# each adding one option, fed by the fp16 UAV-Dot detector. Then scores them with TrackEval.
# Usage: scripts/run_tracking_ablation.sh [video ...]   (default: every video in data/raw/UP-COUNT-TRACK/test_videos)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=external/uav-dot/.venv/bin/python
videos=("$@")
[ ${#videos[@]} -gt 0 ] || videos=(data/raw/UP-COUNT-TRACK/test_videos/*.mp4)

declare -A variants=(
    [baseline]=""
    [cmc]="--cmc"
    [cmc+alt]="--cmc --alt"
    [cmc+alt+cls]="--cmc --alt --cls"
    [full]="--cmc --alt --cls --ddcf"
)
order=(baseline cmc cmc+alt cmc+alt+cls full)

for video in "${videos[@]}"; do
    for name in "${order[@]}"; do
        # shellcheck disable=SC2086  # variant options are deliberately word-split
        $PY scripts/track_video.py "$video" --dataset upcount ${variants[$name]} --name "$name" 2>&1 \
            | grep -E '^\{' | $PY -c "import json, sys; s = json.load(sys.stdin); print(f\"{s['video'].split('/')[-1]} {s['tracker']:<12} {s['frames']} frames, {s['unique_tracks']} tracks, detect {s['median_detect_ms']:.1f} ms, track {s['median_track_ms']:.1f} ms\")"
    done
done
$PY scripts/evaluate_tracking.py "${order[@]}" 2>&1 | grep -v -i warn
