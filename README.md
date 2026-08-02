# UAV P2PNet head detection and crowd counting

This private research repository adapts
[P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet) for
individual head-point localization and crowd counting in VisDrone2020-CC
aerial imagery. Unlike density-map visualization, every retained prediction is
an explicit `(x, y, confidence)` head proposal.

The repository contains the VisDrone data adapter, sequence-level split
preparation, corrected P2PNet training code, checkpoint management, early
stopping, metrics logging, and full-resolution point inference. The trained
checkpoint is distributed separately as a private GitHub release asset because
it exceeds GitHub's normal source-file size limit.

> **License restriction:** the vendored P2PNet source is licensed for academic
> research only. Read
> [`third_party/CrowdCounting-P2PNet/LICENSE`](third_party/CrowdCounting-P2PNet/LICENSE)
> before using or redistributing this project.

## Result summary

| Item | Result |
|---|---:|
| Dataset | VisDrone2020-CC |
| Split | Sequence-level fold 0 of 5, seed 42 |
| Training frames | 1,950 |
| Validation frames | 510 |
| Training head points | 290,556 |
| Validation head points | 72,171 |
| Best epoch | 49 |
| Best validation MAE | **10.0255** |
| Best validation RMSE | **13.0832** |
| Confidence threshold | 0.5 |
| Stopping epoch | 79 of 100, early stopping |

Lower MAE and RMSE are better. These values measure count error on the retained
fold; they are not classification accuracy percentages and do not measure
point-localization precision or recall.

Detailed methodology and limitations are documented in [`REPORT.md`](REPORT.md).
Checkpoint-specific details are in [`MODEL_CARD.md`](MODEL_CARD.md).

## Repository layout

```text
.
├── MODEL_CARD.md
├── REPORT.md
├── RELEASE_NOTES.md
├── requirements.txt
├── scripts/
│   ├── infer_p2p.py
│   ├── p2p_visdrone_utils.py
│   └── prepare_p2pnet_visdrone.py
└── third_party/CrowdCounting-P2PNet/
    ├── crowd_datasets/VisDrone/
    ├── models/
    ├── util/
    ├── LICENSE
    └── train.py
```

The dataset, checkpoints, training logs, TensorBoard events, and generated
inference images are deliberately ignored. This keeps the Git repository small
and avoids redistributing VisDrone imagery.

## Requirements

- Linux
- Python 3.13 for the reproduced environment
- NVIDIA CUDA-capable GPU recommended
- GitHub CLI for downloading the private model release
- Sufficient RAM/VRAM for full-resolution inference

The completed experiment used an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB
VRAM. CPU inference works but is much slower.

## Clone and install

Because the repository is private, authenticate GitHub CLI first:

```bash
gh auth login
gh repo clone owaisasghar/uav-p2pnet-head-counting
cd uav-p2pnet-head-counting
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## Download the trained model

The best checkpoint is attached to the private `v1.0.0` release:

```bash
mkdir -p runs/p2pnet_visdrone_fold0/checkpoints
gh release download v1.0.0 \
  --repo owaisasghar/uav-p2pnet-head-counting \
  --pattern best_mae.pth \
  --dir runs/p2pnet_visdrone_fold0/checkpoints
sha256sum runs/p2pnet_visdrone_fold0/checkpoints/best_mae.pth
```

Expected SHA-256:

```text
8add58d38e1e0bb2d0de96500073eda9171d1af796488b5faf6e7b218df53233
```

## Prepare VisDrone2020-CC

Obtain VisDrone2020-CC from its authorized distributor. Do not commit the
archive or extracted images. The expected source layout is:

```text
data/raw/VisDrone2020-CC/
├── annotations/
│   ├── 00001.txt
│   └── ...
└── sequences/
    ├── 00001/
    │   ├── 00001.jpg
    │   └── ...
    └── ...
```

Convert sequence annotations into per-frame point files and deterministic
train/validation lists:

```bash
PYTHONPATH=. .venv/bin/python scripts/prepare_p2pnet_visdrone.py \
  --root data/raw/VisDrone2020-CC \
  --output data/processed/p2pnet_visdrone \
  --fold 0 \
  --folds 5 \
  --seed 42
```

Expected summary:

```json
{
  "train_frames": 1950,
  "validation_frames": 510,
  "train_points": 290556,
  "validation_points": 72171
}
```

## Train

To reproduce the initialization used for this run, place the official
ShanghaiTech Part A P2PNet checkpoint at:

```text
third_party/CrowdCounting-P2PNet/weights/SHTechA.pth
```

It can be downloaded directly from the upstream repository:

```bash
mkdir -p third_party/CrowdCounting-P2PNet/weights
curl -L \
  https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/raw/main/weights/SHTechA.pth \
  -o third_party/CrowdCounting-P2PNet/weights/SHTechA.pth
```

Then run:

```bash
PYTHONPATH=third_party/CrowdCounting-P2PNet \
  .venv/bin/python third_party/CrowdCounting-P2PNet/train.py \
  --dataset_file VisDrone \
  --data_root data/processed/p2pnet_visdrone \
  --no_pretrained_backbone \
  --resume third_party/CrowdCounting-P2PNet/weights/SHTechA.pth \
  --epochs 100 \
  --lr 0.0001 \
  --lr_backbone 0.00001 \
  --lr_drop 70 \
  --eval_freq 2 \
  --patience 15 \
  --batch_size 2 \
  --patch_size 512 \
  --num_patches 2 \
  --num_workers 4 \
  --output_dir runs/p2pnet_visdrone_fold0/log \
  --checkpoints_dir runs/p2pnet_visdrone_fold0/checkpoints \
  --tensorboard_dir runs/p2pnet_visdrone_fold0/tensorboard \
  --gpu_id 0
```

Training writes:

- `best_mae.pth`: checkpoint with the lowest validation MAE;
- `latest.pth`: most recent checkpoint including optimizer and scheduler state;
- `metrics.jsonl`: per-epoch loss, learning rate, MAE, and RMSE;
- TensorBoard events under the configured TensorBoard directory.

Resume an interrupted training run by changing `--resume` to `latest.pth`. The
saved epoch, optimizer, and scheduler are restored automatically.

## Inference

Run on a deterministic held-out frame:

```bash
PYTHONPATH=. .venv/bin/python scripts/infer_p2p.py \
  --checkpoint runs/p2pnet_visdrone_fold0/checkpoints/best_mae.pth \
  --validation-index 122 \
  --threshold 0.5 \
  --output-dir inference/p2pnet_visdrone_best
```

Run on an external aerial image:

```bash
PYTHONPATH=. .venv/bin/python scripts/infer_p2p.py \
  --checkpoint runs/p2pnet_visdrone_fold0/checkpoints/best_mae.pth \
  --input path/to/image.jpg \
  --threshold 0.5 \
  --output-dir inference/p2pnet_external
```

Each inference run writes:

- `*_predicted_points.jpg`: red predicted head points;
- `*_comparison.jpg`: prediction and green ground truth side by side when an
  annotation exists;
- `*_points.json`: count, latency, device, and every predicted
  `(x, y, confidence)` tuple.

The script pads the image to a multiple of 128 without rescaling the original
coordinates. Predictions outside the original image boundary are removed.

## Important implementation changes

This repository vendors upstream P2PNet commit
`5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455` and adds:

- VisDrone2020-CC sequence parsing and a deterministic five-fold split;
- crowd-aware 512×512 training crops, scaling, color jitter, and flipping;
- configurable pretrained-backbone loading;
- portable VGG weight loading;
- checkpoint resume with optimizer, scheduler, and epoch state;
- JSONL metrics, best-checkpoint selection, and early stopping;
- modern PyTorch/torchvision compatibility;
- direct point-inference images and JSON output;
- a correction from `loss_points` to `loss_point`, ensuring the localization
  loss returned by the criterion is actually included in optimization.

## Known limitations

- Only one deterministic sequence-level fold was trained; the result is not a
  five-fold mean.
- MAE/RMSE evaluate counting only. Point precision, recall, F1, and nAP have not
  yet been reported.
- The fixed confidence threshold is 0.5 and was not calibrated on an
  independent test set.
- Performance has not been established on a different drone, camera, country,
  altitude distribution, or weather domain.
- Full-resolution VGG16-BN P2PNet inference is not real-time on the reproduced
  laptop GPU without further optimization.
- The model may produce false positives on objects that visually resemble
  people and may miss small, occluded, or low-contrast heads.

## Responsible use

This model was prepared for academic crowd-analysis research. Crowd monitoring
can affect privacy and civil rights. Obtain authorization, comply with local
law, minimize retention of identifiable imagery, validate performance for the
actual operating domain, and keep human oversight for safety-critical use.

## Attribution

The implementation is based on:

> Qingyu Song, Changan Wang, Zhengkai Jiang, Yabiao Wang, Ying Tai,
> Chengjie Wang, Jilin Li, Feiyue Huang, and Yang Wu,
> “Rethinking Counting and Localization in Crowds: A Purely Point-Based
> Framework,” ICCV 2021.

See the upstream license and [`NOTICE.md`](NOTICE.md) for source provenance and
modification details.
