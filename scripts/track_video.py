"""Track people through a drone video: fast UAV-Dot detections fed to the uav-dot-track tracker.

The tracker is the authors' code (external/uav-dot-track, Ptak & Kraft, IEEE Access 2026, Apache-2.0); its options
are used unchanged: --cmc (camera-motion compensation by optical flow), --alt (altitude-aware gating), --cls (extra
head classifier for new tracks) and --ddcf (correlation filters on UAV-Dot decoder features to recover lost tracks).
Detection runs as in benchmark_latency.py (fp16, channels_last, GPU preprocessing) instead of the authors' fp32 CPU
path, and decoder features leave the GPU only when DDCF needs them. fp16 leaves tracking unchanged on typical scenes;
on ~1,000-person frames its tiny numeric differences cascade into other identity assignments, so --precision fp32
reproduces the authors' tracks there.

Writes MOT-format tracks (<out>/MOT17-test/<name>/data/<video>.txt, for evaluate_tracking.py), per-stage timings
(<out>/timing/<name>/<video>.json) and optionally an annotated 1080p video. Runs in the UAV-Dot environment:

    external/uav-dot/.venv/bin/python scripts/track_video.py data/raw/UP-COUNT-TRACK/test_videos/0035.mp4 \
        --dataset upcount --cmc --alt --cls --ddcf --name full --video-out runs/tracking/0035_full.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACK_ROOT = PROJECT_ROOT / "external" / "uav-dot-track"
CHECKPOINTS = {
    "dronecrowd": PROJECT_ROOT / "external/uav-dot/checkpoints/dot_pd_dronecrowd_51.00.ckpt",
    "upcount": PROJECT_ROOT / "external/uav-dot/checkpoints/dot_pd_upcount_66.49.ckpt",
}


def stub_mxnet() -> None:
    """pyECO imports MXNet for a ResNet feature that DDCF never uses (it tracks on UAV-Dot decoder features)."""
    names = ["mxnet", "mxnet.gluon", "mxnet.gluon.model_zoo", "mxnet.gluon.model_zoo.vision"]
    modules = {name: types.ModuleType(name) for name in names}
    for parent, child in zip(names, names[1:]):
        setattr(modules[parent], child.rsplit(".", 1)[1], modules[child])
    sys.modules.update(modules)


# The tracker and its UAV-Dot fork (which also returns decoder features) both use a top-level `src` namespace.
sys.path[:0] = [str(TRACK_ROOT), str(TRACK_ROOT / "uavdot")]
stub_mxnet()
from src.model.dot_regressor import DotRegressor  # noqa: E402
from src.tracking.detection_result import from_numpy_to_detection_results  # noqa: E402
from src.tracking.tracker import Tracker  # noqa: E402
from src.utils.csv_writer import CSVWriter  # noqa: E402
from src.utils.video_reader import VideoAltitudeReader  # noqa: E402


class Detector:
    """UAV-Dot with GPU preprocessing; points in video pixels, plus decoder features on request."""

    def __init__(self, checkpoint: Path, normalisation: str, precision: str = "fp16") -> None:
        config = yaml.safe_load((TRACK_ROOT / "uavdot" / "configs" / f"{normalisation}.yaml").read_text())
        model = DotRegressor.load_from_checkpoint(str(checkpoint), map_location="cuda")
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.model = model.eval().to(self.dtype).to(memory_format=torch.channels_last)
        self.width, self.height = model._image_size
        self.mean = torch.tensor(config["data_mean"], device="cuda").view(1, 3, 1, 1) * 255
        self.std = torch.tensor(config["data_std"], device="cuda").view(1, 3, 1, 1) * 255

    @torch.inference_mode()
    def __call__(self, frame: np.ndarray, features: bool) -> tuple[np.ndarray, np.ndarray | None]:
        height, width = frame.shape[:2]
        x = torch.from_numpy(frame).cuda().flip(-1).permute(2, 0, 1)[None].float()  # BGR -> RGB
        x = F.interpolate(x, size=(self.height, self.width), mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).to(self.dtype).contiguous(memory_format=torch.channels_last)
        heatmap, _, _, decoder_output = self.model(x)
        points = self.model.postprocessing(heatmap.float(), thresh=0.2)[0]
        detections = torch.stack([
            points[:, 1] * width / self.width,
            points[:, 2] * height / self.height,
            torch.sigmoid(points[:, 3]),
        ], 1).cpu().numpy()
        return detections, decoder_output.float().cpu().numpy() if features else None


def draw_overlay(frame: np.ndarray, frame_index: int, people: int, unique: int, fps: float) -> None:
    lines = [f"people in frame: {people}", f"unique people so far: {unique}", f"pipeline: {fps:.1f} FPS"]
    scale = frame.shape[1] / 1920
    cv2.rectangle(frame, (0, 0), (int(560 * scale), int(150 * scale)), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (int(16 * scale), int((44 + 44 * i) * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1 * scale, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--dataset", choices=["dronecrowd", "upcount"], required=True,
                        help="tracker settings (classifier, flow scale, altitude source) and default detector")
    parser.add_argument("--checkpoint", type=Path, default=None, help="UAV-Dot checkpoint; default the released one")
    parser.add_argument("--normalisation", choices=["dronecrowd", "upcount"], default=None,
                        help="detector input statistics; default --dataset's (our trained models use dronecrowd)")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16",
                        help="detector precision; fp32 matches the authors' tracks on very dense crowds (REPORT §15)")
    parser.add_argument("--cmc", action="store_true")
    parser.add_argument("--alt", action="store_true")
    parser.add_argument("--cls", action="store_true")
    parser.add_argument("--ddcf", action="store_true")
    parser.add_argument("--max-age", type=int, default=60, help="frames a lost track survives (authors' default)")
    parser.add_argument("--min-hits", type=int, default=30, help="frames before a track is confirmed")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0: whole video)")
    parser.add_argument("--name", default="tracker")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "tracking")
    parser.add_argument("--video-out", type=Path, default=None, help="annotated 1080p video")
    args = parser.parse_args()

    video = args.video.absolute()  # not resolve(): the altitude log is found from the path as given
    out = args.out.resolve()
    video_out = args.video_out.resolve() if args.video_out else None
    detector = Detector((args.checkpoint or CHECKPOINTS[args.dataset]).resolve(), args.normalisation or args.dataset,
                        args.precision)
    os.chdir(TRACK_ROOT)  # the tracker loads its classifier weights by relative path

    reader = VideoAltitudeReader(str(video), dataset=args.dataset, use_alt=args.alt)
    tracker = Tracker(dataset=args.dataset, max_age=args.max_age, min_hits=args.min_hits, use_flow=args.cmc,
                      use_add_cls=args.cls, use_pointflow=args.ddcf, track_use_cutoff=False,
                      flow_scale_factor=2.0 if args.dataset == "dronecrowd" else 4.0)
    # TrackEval's MOT layout: <trackers folder>/MOT17-test/<tracker>/data/<sequence>.txt
    writer = CSVWriter(str(out / "MOT17-test" / args.name / "data"), reader.file_name)
    video_writer = None
    if video_out:
        video_out.parent.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), reader.fps, (1920, 1080))

    timings = {"decode_ms": [], "detect_ms": [], "track_ms": []}
    unique_ids, detections_per_frame = set(), []
    frames = iter(reader)
    for i in range(len(reader) if not args.max_frames else min(args.max_frames, len(reader))):
        started = time.perf_counter()
        try:
            frame, altitude = next(frames)
        except StopIteration:
            break
        decoded = time.perf_counter()
        detections, features = detector(frame, features=args.ddcf)
        detected = time.perf_counter()
        tracks = tracker(frame, features, i, from_numpy_to_detection_results(detections, altitude, frame_shape=frame.shape))
        tracked = time.perf_counter()
        for track in tracks:
            writer.update(frame_id=i, track=track)
            unique_ids.add(track.track_id)
        timings["decode_ms"].append((decoded - started) * 1000)
        timings["detect_ms"].append((detected - decoded) * 1000)
        timings["track_ms"].append((tracked - detected) * 1000)
        detections_per_frame.append(len(detections))
        if video_writer:
            for track in tracks:
                track.draw(frame, history_limit=100, pointsize=0.5 if args.dataset == "dronecrowd" else 1.0)
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            draw_overlay(frame, i, len(tracks), len(unique_ids),
                         1000 / (timings["detect_ms"][-1] + timings["track_ms"][-1]))
            video_writer.write(frame)
    writer.release()
    if video_writer:
        video_writer.release()

    summary = {
        "video": str(video), "tracker": args.name, "frames": len(detections_per_frame), "fps": reader.fps,
        "options": {k: getattr(args, k) for k in ("precision", "cmc", "alt", "cls", "ddcf", "max_age", "min_hits")},
        "checkpoint": str(args.checkpoint or CHECKPOINTS[args.dataset]),
        "detections_per_frame": float(np.mean(detections_per_frame)),
        "unique_tracks": len(unique_ids),
        **{f"median_{k}": statistics.median(v[1:]) for k, v in timings.items()},
    }
    summary["median_pipeline_ms"] = summary["median_detect_ms"] + summary["median_track_ms"]
    timing_path = out / "timing" / args.name / f"{reader.file_name}.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
