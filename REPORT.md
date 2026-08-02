# Technical report: P2PNet on VisDrone2020-CC

## 1. Objective

The objective was to build an aerial crowd-counting system that returns genuine
individual head coordinates rather than approximate peaks extracted from a
density map. The retained solution fine-tunes P2PNet on VisDrone2020-CC and
produces a confidence score and image coordinate for each detected head.

## 2. Method

P2PNet formulates crowd localization as direct point prediction. A VGG16-BN
backbone produces image features, a point proposal head regresses coordinates,
and a classification head distinguishes head proposals from background.
Hungarian one-to-one matching assigns predictions to annotated points during
training.

At inference, the model returns `pred_points` and `pred_logits`. The deployment
script applies softmax to the head class, keeps predictions at or above a 0.5
confidence threshold, removes points outside the original image boundary, and
uses the number of retained points as the predicted crowd count.

## 3. Dataset and split

The experiment used the annotated portion of VisDrone2020-CC:

| Split statistic | Value |
|---|---:|
| Annotated frames discovered | 2,460 |
| Sequences | 82 |
| Cross-validation folds | 5 |
| Selected validation fold | 0 |
| Random seed | 42 |
| Training frames | 1,950 |
| Validation frames | 510 |
| Training head points | 290,556 |
| Validation head points | 72,171 |

The split is made at sequence level. Frames from a sequence cannot appear in
both training and validation, reducing temporal leakage between adjacent video
frames.

The converter accepts the VisDrone2020-CC format where one annotation file per
sequence stores `frame,x,y` rows. It produces one two-column point file per
image plus `train.list` and `val.list` manifests.

## 4. Data pipeline

Training uses:

- ImageNet normalization;
- color jitter;
- random scale from 0.8 to 1.2;
- random horizontal flip;
- two 512×512 patches per source image;
- crowd-aware crop centers near an annotated point 80% of the time.

Validation uses complete normalized frames without crop augmentation. The data
loader returns the image tensor and a variable-length list of point targets.

## 5. Implementation corrections and extensions

The starting point was Tencent's P2PNet repository at commit
`5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455`.

The main changes are:

1. Added a VisDrone dataset implementation and loader selection.
2. Added patch size and patch count training arguments.
3. Added deterministic data-limiting options for smoke tests.
4. Made ImageNet backbone initialization configurable.
5. Replaced a private VGG checkpoint path with portable weight loading.
6. Corrected torchvision version parsing for modern releases.
7. Stored epoch, optimizer, and scheduler state in checkpoints.
8. Added `metrics.jsonl`, TensorBoard fallback, best-MAE checkpointing, and
   validation-based early stopping.
9. Added full-resolution point inference with visual and JSON outputs.
10. Corrected the loss weight key from `loss_points` to `loss_point`. The
    criterion emits the singular key, so the upstream mismatch excluded the
    localization loss from the weighted training objective.

## 6. Training configuration

| Parameter | Value |
|---|---:|
| Initialization | Official ShanghaiTech Part A checkpoint |
| Maximum epochs | 100 |
| Batch size | 2 source images |
| Patches per source image | 2 |
| Patch size | 512×512 |
| Main learning rate | 1×10⁻⁴ |
| Backbone learning rate | 1×10⁻⁵ |
| LR step epoch | 70 |
| Evaluation frequency | Every 2 epochs |
| Early-stopping patience | 15 validation evaluations |
| Workers | 4 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB |

The completed run stopped after epoch 79 because validation MAE had not
improved for 15 evaluations. Total recorded training time was 9:47:12.

## 7. Counting results

| Checkpoint | Epoch | Train loss | Validation MAE | Validation RMSE |
|---|---:|---:|---:|---:|
| Best MAE | **49** | 0.004503 | **10.0255** | **13.0832** |
| Final/early stop | 79 | 0.003693 | 13.1451 | 17.0126 |

Training loss continued to decrease after epoch 49 while validation results
were unstable and did not improve. This is evidence of overfitting, which is
why downstream inference must use `best_mae.pth`, not `latest.pth`.

MAE is the mean absolute difference between predicted and annotated counts per
image. RMSE is the square root of the mean squared count error. Neither metric
measures whether individual predicted points are spatially matched to the
correct annotated heads.

## 8. Qualitative inference checks

The best checkpoint was exercised on held-out images using a 0.5 threshold:

| Sequence/frame | Predicted | Ground truth | Absolute error |
|---|---:|---:|---:|
| 00026/00003 | 150 | 145 | 5 |
| 00006/00001 | 370 | 393 | 23 |
| 00010/00021 | 105 | 109 | 4 |
| 00040/00021 | 151 | 134 | 17 |
| 00078/00021 | 42 | 45 | 3 |
| 00109/00030 | 195 | 165 | 30 |

The mean absolute error over these six selected examples is 13.67. This number
is only a small qualitative sample and must not replace the full 510-frame
validation MAE.

## 9. Runtime observations

After warm-up, complete-resolution GPU forward passes were approximately
236–239 ms on selected 1,920×1,080 frames, or about 4.2 model forward passes per
second. The timing excludes model loading and most file I/O. A later process
encountered a CUDA initialization error and fell back to CPU, requiring about
3.92 seconds for the forward pass. CUDA availability must therefore be checked
before using runtime results.

This unoptimized VGG16-BN model is suitable as an accuracy baseline, not yet as
a real-time deployment. Candidate optimizations include fixed lower resolution,
FP16, compilation/TensorRT, a lighter backbone, scheduled inference, and
tracking or temporal smoothing between detector frames. Each optimization must
be evaluated for tiny-head recall.

## 10. Reproducibility boundaries

This report describes one deterministic sequence-level fold, not a full
five-fold experiment. The selected confidence threshold was not calibrated on
an independent test set. Hardware, CUDA, camera resolution, preprocessing, and
domain shift can all change both accuracy and latency.

The model was not evaluated with a localization matching metric. A complete
point-localization evaluation should report precision, recall, F1, and a
distance-normalized metric such as nAP in addition to count MAE/RMSE.

## 11. Vehicle-detection extension

For future 100 m vehicle detection, the most defensible baseline is a separate
vehicle bounding-box detector trained on aerial imagery and fine-tuned on the
target drone/camera. A later one-model system can share a lightweight backbone
between:

- the P2P head-point branch; and
- a vehicle class-and-box branch.

A multi-class point-only P2PNet is also possible if vehicle centers and counts
are sufficient. It is not appropriate when vehicle dimensions or oriented
boxes are required. Any joint model should be compared against the two
specialized baselines for count error, localization, vehicle mAP, latency, and
VRAM usage.

## 12. Conclusion

The retained system successfully converts the original density-oriented work
into genuine individual head-point prediction. The best fold-0 checkpoint
achieves a validation MAE of 10.0255 and produces inspectable head coordinates.
The next technical priorities are spatial localization evaluation, CUDA/runtime
optimization, cross-domain testing, and a separately validated aerial vehicle
detector before attempting a shared multi-task model.
