---
title: UAV Crowd Counting - UAV-Dot vs P2PNet
emoji: 🚁
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: other
models:
  - owaisasghar/p2pnet-visdrone-head-counting
---

# UAV crowd counting: UAV-Dot vs P2PNet

Upload an aerial RGB image and compare two head-point detectors side by side.
Each model returns individually marked head points, the predicted count,
confidence scores and inference timing.

| VisDrone2020-CC test set (900 frames) | Count MAE | F1 @ 16 px | L-mAP | fp16 latency, RTX 4060 Laptop |
|---|---:|---:|---:|---:|
| UAV-Dot (released weights) | 25.0 | 0.726 | 51.0 | 29 ms |
| P2PNet (trained in the project) | 21.9 | 0.725 | 52.5 | 115 ms |

## Where the models come from

- **P2PNet:** loaded from the model repository set by `P2PNET_MODEL_ID`
  (default `owaisasghar/p2pnet-visdrone-head-counting`).
- **UAV-Dot:** the released DOT + Pixel Distill DroneCrowd weights are **not
  re-hosted**. On first use, `uavdot_runtime.py` downloads them from the UAV-Dot
  authors' public mirror and refuses any file whose SHA-256 differs. The cache
  directory is `UAVDOT_CHECKPOINT_DIR` (default `~/.cache/uavdot`).

## Licensing and attribution

- **UAV-Dot network:** the definition in `uavdot_runtime.py` is adapted from
  [up-count/uav-dot](https://github.com/up-count/uav-dot) (commit `a6d3664`)
  under the Apache License 2.0; see `LICENSE-uav-dot.txt` and the file header
  for the changes.
- **P2PNet:** this part is restricted to academic research by its upstream
  license.

## Notes

- **Resolution:** the published metrics use complete 1920×1080 frames. UAV-Dot
  always runs at 1920×1088. P2PNet defaults to a 1280-pixel maximum side, so its
  count may differ from the full-resolution protocol.
- **Validation:** this academic research demo is not validated for
  safety-critical deployment, identity recognition, night footage or new
  drone and camera domains.
