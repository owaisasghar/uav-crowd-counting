---
title: UAV P2PNet Head-Point Demo
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

# UAV P2PNet head-point demo

Upload an aerial RGB image to obtain individually marked head-point proposals,
the predicted count, confidence scores, and inference timing.

The Space expects `model.safetensors` and `p2pnet_runtime.py` in the model
repository configured by `P2PNET_MODEL_ID`. The default is
`owaisasghar/p2pnet-visdrone-head-counting`.

Published validation metrics use complete 1,920×1,080 frames. The interactive
demo defaults to a 1,280-pixel maximum side, so its count may differ from the
full-resolution evaluation protocol.

This academic research demo is not validated for safety-critical deployment,
identity recognition, or new drone/camera domains.
