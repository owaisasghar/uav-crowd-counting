# P2PNet VisDrone fold-0 model v1.0.0

This private release contains the best validation checkpoint from the retained
VisDrone2020-CC fold-0 training run.

## Asset

```text
best_mae.pth
230,238,136 bytes
SHA-256: 8add58d38e1e0bb2d0de96500073eda9171d1af796488b5faf6e7b218df53233
```

## Validation result

- Best epoch: 49
- MAE: 10.025490196078431
- RMSE: 13.083217808906726
- Validation frames: 510
- Confidence threshold: 0.5

Use this checkpoint rather than `latest.pth`; validation error worsened after
epoch 49 while training loss continued to decrease.

## Download

```bash
mkdir -p runs/p2pnet_visdrone_fold0/checkpoints
gh release download v1.0.0 \
  --repo owaisasghar/uav-p2pnet-head-counting \
  --pattern best_mae.pth \
  --dir runs/p2pnet_visdrone_fold0/checkpoints
```

The upstream P2PNet license restricts use to academic research. Review
`MODEL_CARD.md` and `third_party/CrowdCounting-P2PNet/LICENSE` before use.
