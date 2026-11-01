"""Render the showcase video (LinkedIn) from saved predictions and tracks; no model runs here.

Clips: UP-COUNT-TRACK test video 0016 (about 1,000 people, tracked), 0086 (crosswalk, unique-person count) and
VisDrone2020-CC test sequence 00112 (UAV-Dot and P2PNet split screen), between title, results and closing cards.
Inputs come from evaluate_dronecrowd.py (runs/eval) and track_video.py (runs/tracking). Runs in the UAV-Dot
environment and needs ffmpeg with libx264:

    external/uav-dot/.venv/bin/python scripts/make_showcase_video.py --out runs/demo/showcase.mp4
"""
from __future__ import annotations

import argparse
import colorsys
import json
import subprocess
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_dronecrowd import frames_for, ground_truth
from uavdot_data import PROJECT_ROOT

W, H, FPS = 1920, 1080, 30
FADE = 12
FONTS = Path("/usr/share/fonts/truetype/noto")
REPO = "github.com/owaisasghar/uav-crowd-counting"
SPACE = "huggingface.co/spaces/uwaisasghar/uav-crowd-counting"
TRACK_DATA = PROJECT_ROOT / "data/raw/UP-COUNT-TRACK"
UAVDOT_BGR, P2PNET_BGR = (255, 160, 40), (60, 90, 255)


@lru_cache(maxsize=None)
def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / f"NotoSans-{weight}.ttf"), size)


def draw_text_block(frame: np.ndarray, x: int, y: int, lines: list[tuple[str, int, str, tuple]], pad: int = 22,
                    min_width: int = 0, alpha: int = 175) -> np.ndarray:
    """Lines of (text, size, weight, rgb) on a translucent rounded card with its top-left corner at (x, y)."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    heights = [font(s, w).getbbox(t)[3] + 10 for t, s, w, _ in lines]
    width = max([min_width] + [int(draw.textlength(t, font=font(s, w))) for t, s, w, _ in lines]) + 2 * pad
    draw.rounded_rectangle((x, y, x + width, y + sum(heights) + 2 * pad - 10), radius=16, fill=(12, 14, 18, alpha))
    cy = y + pad
    for (text, size, weight, rgb), height in zip(lines, heights):
        draw.text((x + pad, cy), text, font=font(size, weight), fill=rgb + (255,))
        cy += height
    return cv2.cvtColor(np.array(Image.alpha_composite(image, overlay).convert("RGB")), cv2.COLOR_RGB2BGR)


def footnote(frame: np.ndarray, text: str) -> np.ndarray:
    return draw_text_block(frame, 24, H - 66, [(text, 20, "Regular", (205, 205, 205))], pad=12, alpha=140)


def track_colour(track_id: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((track_id * 0.61803) % 1.0, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def read_tracks(path: Path) -> dict[int, list[tuple[int, float, float]]]:
    """MOT rows (frame, id, left, top, 20, 20, ...) -> {frame: [(id, centre x, centre y)]}, frames from 0."""
    by_frame = defaultdict(list)
    for row in np.loadtxt(path, delimiter=","):
        by_frame[int(row[0]) - 1].append((int(row[1]), row[2] + 10, row[3] + 10))
    return by_frame


def gt_counts(video: str) -> dict[int, int]:
    rows = np.loadtxt(TRACK_DATA / "mot_style_test_labels/gt/MOT17-test" / video / "gt/gt.txt", delimiter=",")
    frames, counts = np.unique(rows[:, 0].astype(int), return_counts=True)
    return {f - 1: c for f, c in zip(frames, counts)}


def video_frames(video: str, start: int, stop: int):
    capture = cv2.VideoCapture(str(TRACK_DATA / "test_videos" / f"{video}.mp4"))
    for index in range(stop):
        ok, frame = capture.read()
        if not ok:
            return
        if index >= start:
            yield index, frame
    capture.release()


def tracked_clip(video: str, run: str, start: int, stop: int, trail: int, dot: int, card, note: str, hold: int = 0):
    """Video frames with each tracked person's dot and recent trail, plus a card built by
    card(frame, index, tracked now, unique so far, final); the last frame is held for `hold` frames."""
    tracks = read_tracks(PROJECT_ROOT / "runs/tracking/MOT17-test" / run / "data" / f"{video}.txt")
    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail))
    seen: set[int] = set()
    for index in range(start):  # build trails and the unique-ID count up to the first shown frame
        for track_id, x, y in tracks.get(index, []):
            history[track_id].append((x, y))
            seen.add(track_id)
    for index, frame in video_frames(video, start, stop):
        scale = W / frame.shape[1]
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        current = tracks.get(index, [])
        for track_id, x, y in current:
            history[track_id].append((x, y))
            seen.add(track_id)
        for track_id, x, y in current:
            points = (np.array(history[track_id]) * scale).astype(np.int32)
            if len(points) > 1:
                cv2.polylines(frame, [points], False, track_colour(track_id), max(1, dot // 3), cv2.LINE_AA)
        for track_id, x, y in current:
            centre = (int(x * scale), int(y * scale))
            cv2.circle(frame, centre, dot + 1, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(frame, centre, dot, track_colour(track_id), -1, cv2.LINE_AA)
        drawn = frame
        yield footnote(card(drawn, index, len(current), len(seen), False), note)
    final = footnote(card(drawn, index, len(current), len(seen), True), note)
    for _ in range(hold):
        yield final


def split_clip(sequence: str, eval_dir: Path, latency: dict, hold: int):
    """VisDrone test frames: UAV-Dot points on the left half, P2PNet on the right, counts for the whole frame."""
    frames = [f for f in frames_for("test", PROJECT_ROOT / "data/raw/VisDrone2020-CC", None, 0, 5, 42) if f[0] == sequence]
    truth = ground_truth("test", frames)
    predictions = {name: np.load(eval_dir / "test" / f"{name}.npz") for name in ("released_dronecrowd", "p2pnet")}
    for seq, j, path in frames:
        image = cv2.imread(str(path))
        key = f"{seq}/{j:05d}"
        uavdot, p2pnet = predictions["released_dronecrowd"][key], predictions["p2pnet"][key]
        for points, colour, keep in ((uavdot, UAVDOT_BGR, lambda x: x < W / 2), (p2pnet, P2PNET_BGR, lambda x: x >= W / 2)):
            for x, y, _ in points:
                if keep(x):
                    cv2.circle(image, (int(x), int(y)), 5, (20, 20, 20), -1, cv2.LINE_AA)
                    cv2.circle(image, (int(x), int(y)), 4, colour, -1, cv2.LINE_AA)
        cv2.line(image, (W // 2, 0), (W // 2, H), (255, 255, 255), 3)
        true = len(truth[(seq, j)])
        image = draw_text_block(image, 24, 24, [
            ("UAV-Dot", 40, "Bold", (120, 190, 255)),
            (f"{latency['uavdot']['model_ms']:.0f} ms per frame", 28, "SemiBold", (255, 255, 255)),
            (f"Count {len(uavdot)}  ·  true {true}", 26, "Regular", (230, 230, 230)),
        ], min_width=330)
        image = draw_text_block(image, W // 2 + 24, 24, [
            ("P2PNet (trained by me)", 40, "Bold", (255, 140, 110)),
            (f"{latency['p2pnet']['model_ms']:.0f} ms per frame", 28, "SemiBold", (255, 255, 255)),
            (f"Count {len(p2pnet)}  ·  true {true}", 26, "Regular", (230, 230, 230)),
        ], min_width=330)
        image = footnote(image, f"VisDrone2020-CC test sequence {sequence}  ·  counts cover the whole frame  ·  "
                                "speed: RTX 4060 Laptop GPU, fp16")
        for _ in range(hold):
            yield image


def card_frames(background: np.ndarray, blocks: list, seconds: float):
    frame = background.copy()
    for x, y, lines, min_width in blocks:
        frame = draw_text_block(frame, x, y, lines, min_width=min_width, alpha=0)
    for _ in range(int(seconds * FPS)):
        yield frame


def backdrop(video: str, index: int) -> np.ndarray:
    _, frame = next(video_frames(video, index, index + 1))
    frame = cv2.GaussianBlur(cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA), (0, 0), 14)
    return (frame * 0.32).astype(np.uint8)


def faded(frames):
    """Fade a clip in from and out to black, holding back only the last FADE frames."""
    tail: deque = deque()
    for i, frame in enumerate(frames):
        alpha = min(1.0, (i + 1) / FADE)
        tail.append((frame * alpha).astype(np.uint8) if alpha < 1 else frame)
        if len(tail) > FADE:
            yield tail.popleft()
    for i, frame in enumerate(tail):
        yield (frame * ((len(tail) - i) / (FADE + 1))).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "demo" / "showcase.mp4")
    parser.add_argument("--eval", type=Path, default=PROJECT_ROOT / "runs" / "eval")
    args = parser.parse_args()

    test = json.loads((args.eval / "test_metrics.json").read_text())
    latency = json.loads((args.eval / "latency.json").read_text())
    tracking = json.loads((PROJECT_ROOT / "runs/tracking/metrics_typical.json").read_text())
    white, grey, blue = (255, 255, 255), (200, 200, 200), (120, 190, 255)
    speedup = latency["p2pnet"]["model_ms"] / latency["uavdot"]["model_ms"]
    dense_truth, crosswalk_truth = gt_counts("0016"), gt_counts("0086")
    crosswalk_total = len(np.unique(np.loadtxt(
        TRACK_DATA / "mot_style_test_labels/gt/MOT17-test/0086/gt/gt.txt", delimiter=",")[:, 1]))
    base, typical = tracking["baseline"], tracking["full"]

    def dense_card(frame, index, current, unique, final):
        return draw_text_block(frame, 24, 24, [
            ("Mass event, 76 m up", 40, "Bold", white),
            (f"Tracked now: {current:,}", 30, "SemiBold", blue),
            (f"Ground truth: {dense_truth.get(index, 0):,}", 30, "SemiBold", grey),
        ], min_width=380)

    def crosswalk_card(frame, index, current, unique, final):
        lines = [("Every person keeps one ID", 40, "Bold", white),
                 (f"Unique people so far: {unique}", 30, "SemiBold", blue)]
        if final:
            lines.append((f"Ground truth: {crosswalk_total} unique people", 30, "SemiBold", (140, 230, 150)))
        return draw_text_block(frame, 24, 24, lines, min_width=470)

    title_bg, results_bg = backdrop("0016", 200), backdrop("0086", 200)
    clips = [
        card_frames(title_bg, [(150, 330, [
            ("Counting and tracking people from a drone", 72, "Bold", white),
            ("Every head becomes a point  ·  29 ms per 1080p frame  ·  identities kept across video", 34, "Regular", grey),
            ("", 20, "Regular", grey),
            ("Awais Asghar", 32, "SemiBold", blue),
        ], 0)], 3.5),
        tracked_clip("0016", "full_fp32", 40, 280, trail=10, dot=3, card=dense_card,
                     note="Footage: UP-COUNT-TRACK test video (CC BY-NC 4.0, PUT Vision Lab)  ·  "
                          "UAV-Dot + uav-dot-track, rendered offline"),
        tracked_clip("0086", "full", 30, 361, trail=60, dot=6, card=crosswalk_card,
                     note="Footage: UP-COUNT-TRACK test video (CC BY-NC 4.0, PUT Vision Lab)  ·  "
                          "each trail is one tracked person", hold=55),
        split_clip("00112", args.eval, latency, hold=6),
        card_frames(results_bg, [(130, 120, [
            ("Results on held-out test data", 60, "Bold", white),
            ("", 14, "Regular", grey),
            ("Detection  ·  VisDrone2020-CC test, 900 frames", 32, "SemiBold", blue),
            (f"UAV-Dot: count error {test['released_dronecrowd']['mae']:.1f}  ·  F1 "
             f"{test['released_dronecrowd']['f1@16']:.2f}  ·  {latency['uavdot']['model_ms']:.0f} ms per frame "
             f"({speedup:.1f}x faster)", 30, "Regular", white),
            (f"P2PNet (trained by me): count error {test['p2pnet']['mae']:.1f}  ·  F1 {test['p2pnet']['f1@16']:.2f}  ·  "
             f"{latency['p2pnet']['model_ms']:.0f} ms per frame", 30, "Regular", white),
            ("", 14, "Regular", grey),
            ("Tracking  ·  UP-COUNT-TRACK, five test videos", 32, "SemiBold", blue),
            (f"ID switches cut from {base['id_switches']} to {typical['id_switches']}  ·  unique-person count error "
             f"from {100 * base['track_count_nmae']:.0f}% to {100 * typical['track_count_nmae']:.0f}%", 30, "Regular", white),
            ("", 14, "Regular", grey),
            ("An honest negative result", 32, "SemiBold", blue),
            ("Extra training data fixed low-altitude scenes but hurt night scenes,", 30, "Regular", white),
            ("so a rule set before testing kept the original model.", 30, "Regular", white),
        ], 0)], 7.0),
        card_frames(title_bg, [(150, 360, [
            ("Code, evaluation and full report", 34, "SemiBold", grey),
            (REPO, 46, "Bold", white),
            ("", 16, "Regular", grey),
            ("Try it: side-by-side demo", 34, "SemiBold", grey),
            (SPACE, 40, "Bold", white),
            ("", 16, "Regular", grey),
            ("Built on P2PNet (Tencent), UAV-Dot and uav-dot-track (PUT Vision Lab)", 26, "Regular", grey),
        ], 0)], 4.5),
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(args.out)],
        stdin=subprocess.PIPE,
    )
    written = 0
    for clip in clips:
        for frame in faded(clip):
            encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
            written += 1
    encoder.stdin.close()
    encoder.wait()
    print(f"wrote {args.out} ({written / FPS:.1f} s, {args.out.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
