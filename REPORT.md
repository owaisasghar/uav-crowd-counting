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

### Relationship to the referenced UAV study

This implementation is an adaptation of P2PNet to the same aerial crowd
domain, not a reproduction of the network in Ptak et al., *Remote Sensing*
2022. The referenced study formulates counting as density regression and
evaluates optical-flow-augmented five-channel inputs, multiple architectures,
multiple resolutions, five folds, and embedded hardware. This repository
instead prioritizes explicit individual head coordinates and evaluates one
sequence-level fold. A paper-reproduction claim would require implementing and
testing the study's full density-regression protocol.

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
11. Added device-correct anchor generation instead of selecting CUDA from
    global availability, enabling explicit CPU inference on GPU-equipped hosts.
12. Added boundary-aware count evaluation, confidence sweeps, one-to-one point
    localization metrics, ranked AP, tests, and fold aggregation.
13. Added a strictly validated inference-only SafeTensors export, standalone
    runtime, Hugging Face model-card package, and Gradio image demo.
14. Added CI, machine-readable citation metadata, a root academic-only license,
    and a conceptual social-preview banner clearly labeled as non-evidence.

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

The training evaluator counts predictions produced in the padded tensor. The
deployment wrapper removes points outside the source image. At threshold 0.50,
the boundary-aware deployment protocol gives MAE 10.0235 and RMSE 13.0795,
while the original padded protocol reproduces MAE 10.0255 and RMSE 13.0832.

The validation sweep selected threshold 0.60 for count MAE 9.2902 and RMSE
12.7684. This is a validation-calibrated result, not an independent-test score.

## 8. Point-localization results

Predicted and annotated points are matched one-to-one with Hungarian assignment.
A match is a true positive when its Euclidean distance does not exceed the
specified pixel radius. At the fixed threshold 0.50:

| Radius | Precision | Recall | F1 | Ranked AP |
|---:|---:|---:|---:|---:|
| 8 px | 0.7468 | 0.7773 | 0.7617 | 0.7729 |
| 16 px | **0.7988** | **0.8315** | **0.8148** | **0.8558** |
| 32 px | 0.8356 | 0.8698 | 0.8523 | 0.8943 |

At 16 pixels, 60,010 predictions match annotated heads, with 15,115 false
positives and 12,161 false negatives. The mean matched distance is 3.70 pixels
and the median is 3.03 pixels. Confidence-ranked AP evaluates in-bounds
proposals scoring above 0.001.

The validation sweep selected threshold 0.55 for F1@16px 0.8160 (precision
0.8095, recall 0.8227). Because the same fold selected both the checkpoint and
threshold, fixed-threshold results are the more defensible portfolio baseline.

VisDrone2020-CC supplies points but no head-width annotations. The reported AP
therefore uses a documented fixed pixel radius and must not be called the
head-size-normalized nAP used by benchmarks with head boxes.

## 9. Qualitative inference checks

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

## 10. Runtime observations

After warm-up, complete-resolution GPU forward passes were approximately
236–239 ms on selected 1,920×1,080 frames. The complete 510-frame localization
evaluation averaged 243.14 ms per forward pass, or about 4.11 forward passes
per second. Timing excludes model loading and most file I/O. A later process
encountered a CUDA initialization error and fell back to CPU, requiring about
3.92 seconds for the forward pass. CUDA availability must therefore be checked
before using runtime results.

This unoptimized VGG16-BN model is suitable as an accuracy baseline, not yet as
a real-time deployment. Candidate optimizations include fixed lower resolution,
FP16, compilation/TensorRT, a lighter backbone, scheduled inference, and
tracking or temporal smoothing between detector frames. Each optimization must
be evaluated for tiny-head recall.

## 11. Reproducibility boundaries

This report describes one deterministic sequence-level fold, not a full
five-fold experiment. Threshold calibration reuses the checkpoint-selection
fold rather than an independent test set. Hardware, CUDA, camera resolution,
preprocessing, and domain shift can all change both accuracy and latency.

The repository includes fixed-protocol fold aggregation, but four additional
models must be trained before a five-fold mean and standard deviation can be
reported. No missing-fold values are inferred or synthesized.

The SafeTensors deployment runtime matches the vendored model exactly on a
synthetic tensor: the maximum absolute difference is 0.0 for both logits and
predicted points. This verifies packaging equivalence, not model accuracy on a
new dataset.

## 12. Vehicle-detection extension

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

## 13. Conclusion

The retained system successfully converts the original density-oriented work
into genuine individual head-point prediction. The best fold-0 checkpoint
achieves a fixed-threshold validation MAE of 10.0255, F1@16px of 0.8148, and
ranked AP@16px of 0.8558. The next technical priorities are four remaining
cross-validation folds, independent-domain testing, CUDA/runtime optimization,
and a separately validated aerial vehicle detector before attempting a shared
multi-task model.
