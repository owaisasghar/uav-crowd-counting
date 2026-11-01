# Release notes

## Unreleased: test-set evaluation, UAV-Dot, UP-COUNT experiment, tracking

- Added an official-protocol test evaluation for both detectors
  (`scripts/evaluate_dronecrowd.py`). On the test set P2PNet scores MAE 21.86 and
  L-mAP 52.49; the released UAV-Dot weights score MAE 25.04 and L-mAP 51.00,
  reproducing the published figure.
- Added a latency benchmark (`scripts/benchmark_latency.py`): 29.3 ms per 1080p
  frame for UAV-Dot and 114.7 ms for P2PNet, fp16 on an RTX 4060 Laptop GPU.
- Added a UAV-Dot training pipeline for VisDrone folds with UP-COUNT mixing,
  warm starts and all-sequence training (`scripts/train_uavdot.py`).
- The UP-COUNT fine-tune raised UP-COUNT validation L-mAP from 21.9 to 61.6 but
  lowered test L-mAP to 48.4, so the released weights remain the deployed
  detector (REPORT §14).
- Added uav-dot-track tracking on the fp16 detector (`scripts/track_video.py`),
  a TrackEval ablation on UP-COUNT-TRACK (`scripts/run_tracking_ablation.sh`,
  `scripts/evaluate_tracking.py`) and a range-request downloader for its videos.
- Added a pinned setup for the external Apache-2.0 code
  (`scripts/setup_uavdot.sh`, `requirements-uavdot.txt`, checkpoint checksums)
  and README charts (`scripts/generate_portfolio_charts.py`).

## P2PNet VisDrone fold-0 model v1.0.0

### Local evaluation additions (not published)

- Added one-to-one localization precision, recall, F1, and fixed-radius AP.
- Added threshold calibration, plots, synthetic metric tests, and fold
  aggregation tooling.
- Fixed anchor tensors to follow the input device for explicit CPU/GPU use.
- Fixed-threshold result: F1@16px 0.8148 and ranked AP@16px 0.8558.
- Added a SafeTensors export, Hugging Face model-card package, standalone
  runtime, Gradio Space scaffold, CI workflow, citation file, and root license.
- Added a conceptual portfolio banner; it is not presented as model output.
- Verified the 86,362,656-byte inference export against the original runtime;
  its SHA-256 is
  `acb2f7196538fff3187b2e02fd9629802a9f7d5faec3d8bdbe9c47cd6ed003c9`.

This private release contains the best validation checkpoint from the retained
VisDrone2020-CC fold-0 training run.

### Asset

```text
best_mae.pth
230,238,136 bytes
SHA-256: 8add58d38e1e0bb2d0de96500073eda9171d1af796488b5faf6e7b218df53233
```

### Validation result

- Best epoch: 49
- MAE: 10.025490196078431
- RMSE: 13.083217808906726
- Validation frames: 510
- Confidence threshold: 0.5

Use this checkpoint rather than `latest.pth`; validation error worsened after
epoch 49 while training loss continued to decrease.

### Download

```bash
mkdir -p runs/p2pnet_visdrone_fold0/checkpoints
gh release download v1.0.0 \
  --repo owaisasghar/uav-p2pnet-head-counting \
  --pattern best_mae.pth \
  --dir runs/p2pnet_visdrone_fold0/checkpoints
```

The upstream P2PNet license restricts use to academic research. Review
`MODEL_CARD.md` and `third_party/CrowdCounting-P2PNet/LICENSE` before use.
