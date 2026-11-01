# Technical report: P2PNet, UAV-Dot and tracking on VisDrone2020-CC

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

## 13. Test-set evaluation and inference speed

The VisDrone2020-CC test split holds 30 held-out sequences (900 frames, 30 per
sequence, 137.1 people per frame). They are DroneCrowd test videos sampled
every tenth frame, so our frame `j` is DroneCrowd frame `10(j-1)`. The ground
truth is the DroneCrowd toolkit's head annotation, which on the fold-0
validation frames matches the VisDrone point annotations exactly.

`scripts/evaluate_dronecrowd.py` scores every model the same way:

- **Counts:** MAE, RMSE and bias of the per-frame count.
- **Point matching:** precision, recall and F1 with one-to-one Hungarian
  matching within 16 px. Pairs farther apart are never matched.
- **L-mAP:** the toolkit's `map.py`, averaged over match distances of 1–25 px,
  with each detection as a 20×20 box centred on its point, as UAV-Dot's
  `pt_pred.py` does.

Each model uses its deployment threshold: 0.2 for UAV-Dot (upstream) and 0.5
for P2PNet. The released UAV-Dot weights reproduce their published L-mAP of
51.00 exactly.

| Model | MAE | RMSE | Bias | Precision | Recall | F1@16 px | L-mAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| P2PNet, fold-0 model from this report | **21.86** | **29.58** | −7.18 | 0.745 | **0.706** | 0.725 | **52.49** |
| UAV-Dot DOT+PD, released DroneCrowd weights | 25.04 | 33.66 | −10.24 | **0.755** | 0.699 | **0.726** | 51.00 |

The models were trained on different data. P2PNet saw 1,950 annotated frames
from 65 sequences; the released UAV-Dot weights saw the full DroneCrowd training
videos.

`scripts/benchmark_latency.py` times both models as deployed:

- **Setup:** fp16 weights, channels-last memory, resizing and normalization on
  the GPU, PyTorch 2.1.0 (CUDA 12.1), RTX 4060 Laptop GPU.
- **Frames:** 200 test frames after 20 warm-up frames.
- **Measured:** model latency covers preprocessing, the forward pass and point
  extraction for a 1920×1080 frame already in GPU memory. End-to-end latency
  adds JPEG decoding and upload.

| Model | Model latency (median) | 90th percentile | End-to-end | Model throughput |
|---|---:|---:|---:|---:|
| UAV-Dot DOT+PD | **29.3 ms** | 30.2 ms | **42.0 ms** | **34 FPS** |
| P2PNet | 114.7 ms | 115.7 ms | 127.6 ms | 8.7 FPS |

Half precision halves P2PNet's latency (236 ms in fp32, §10) at unchanged
accuracy, but UAV-Dot remains 3.9 times faster at the same F1.

## 14. UAV-Dot and the UP-COUNT domain experiment

### 14.1 Motivation

On the test set, UAV-Dot finds only about 40% of the people in the two
low-altitude sequences 00017 and 00018, where heads appear larger than in its
training data. Shrinking those frames to 0.6x cut its count error on 00017 from
97 to 7.5, while shrinking a typical sequence raised it from 13 to 118. The
model therefore works in a narrow head-size band. Its training zoom is only
`RandomResizedCrop(scale=(0.5, 1))` with probability 0.4, at most 1.4x.

The released UP-COUNT model handles the low-altitude sequences (00017 count
error 11.5) but fails on typical ones (00022: 222), so the limit is coverage of
the training data, not the architecture. UP-COUNT (CC BY-NC 4.0) provides
10,000 4K frames filmed at 26–101 m.

### 14.2 Training pipeline

`scripts/train_uavdot.py` and `scripts/uavdot_data.py` reuse UAV-Dot's
`DotRegressor`: a MiT-B2 encoder, a U-Net decoder, Pixel Distill and the DOT
loss. The setup:

- **Split:** the same fold-0 sequence split as P2PNet.
- **Optimization:** batch 2 with gradient accumulation 4 (upstream: batch 8),
  16-bit mixed precision, AdamW at 3e-4, the upstream cosine schedule, and
  early stopping on validation F1 at 5 px.
- **Mixing:** a weighted sampler draws a fixed share of each epoch from
  UP-COUNT. The epoch length stays one VisDrone fold (1,950 samples), so runs
  with and without UP-COUNT share steps, schedule and GPU time.
- **Resolution:** UP-COUNT frames are resized from 4K to the 1920×1088 input.
- **Zoom:** the combined runs widen the zoom to `scale=(0.16, 1)`, up to 2.5x.

Before training, the loader was checked against the P2PNet split:
- **Split:** the same 1,950/510 frames, with no test-sequence overlap.
- **Labels:** every label survives resizing, within 1 px.
- **Mask:** every label produces a heatmap peak.
- **Frame alignment:** the released weights score F1@5 px 0.68 on correctly
  aligned frames, against 0.37 when shifted by one frame.

### 14.3 Results on validation

| Run (fold 0) | VisDrone val MAE | VisDrone val F1@16 | VisDrone val L-mAP | UP-COUNT val L-mAP | below 40 m | 40–80 m | 80 m and above |
|---|---:|---:|---:|---:|---:|---:|---:|
| VisDrone only, 100 epochs (best epoch 81) | **11.57** | **0.825** | **65.39** | 20.2 | 5.2 | 25.4 | 23.5 |
| + UP-COUNT from scratch | stalled, see below | | | | | | |
| + UP-COUNT, fine-tuned 30 epochs from the run above | 12.82 | 0.816 | 64.23 | **66.1** | **58.6** | **67.6** | **67.8** |

UP-COUNT validation (1,391 frames, 24.8 people per frame) is scored like the
test set, with points scaled to 1920×1080 so that pixel radii mean the same.

- **From scratch, training stalled.** Validation F1 stayed near 0 for 19
  epochs, while the VisDrone-only run was detecting people by epoch 6. Its
  heatmap was flat on both datasets, and neither BatchNorm statistics nor empty
  frames explained it. The likely cause is sparse UP-COUNT frames (median 18
  people against about 143) at batch 2, which push the focal loss towards
  predicting nothing. Starting from a trained model avoided the problem.
- **Choose the last epoch.** The strict 5 px validation F1 picked epoch 6 of the
  fine-tune, which was worse on both domains (VisDrone L-mAP 62.5, UP-COUNT
  55.7) than the last epoch, so the recipe keeps the last epoch.

### 14.4 Final model and the test set

The final model applied the same recipe to the released DroneCrowd weights
(trained on DroneCrowd's training split, which shares no sequences with the
test set). It fine-tuned on all 82 annotated VisDrone sequences plus UP-COUNT
for 30 epochs and kept the last epoch. Before testing, the rule was fixed: it
replaces the released weights only if test L-mAP exceeds 51.00.

| Test set | MAE | Bias | Precision | Recall | F1@16 px | L-mAP | F1, low altitude | F1, night | F1, other 25 sequences |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Released weights | **25.04** | −10.2 | 0.755 | **0.699** | **0.726** | **51.00** | 0.37 | **0.41** | **0.76** |
| Final, 50% UP-COUNT, 30 epochs | 36.52 | −31.9 | **0.822** | 0.630 | 0.713 | 48.42 | **0.65** | 0.06 | 0.74 |
| Retry, 20% UP-COUNT, 15 epochs (second look) | 34.47 | −31.1 | 0.819 | 0.633 | 0.714 | 48.24 | 0.47 | 0.17 | 0.75 |

Low altitude is sequences 00017 and 00018; night is 00069, 00070 and 00074.
On UP-COUNT validation the final model scores L-mAP 61.6 (59.5 below 40 m),
against 21.9 for the released weights and 77.6 for the released UP-COUNT
specialist, which runs at 4K input.

The fine-tune fixed the low-altitude failure it targeted, but it failed the
rule. The released weights remain the deployed detector.

### 14.5 Diagnosis

- **Night is almost absent from training.** Only three of the 82 training
  sequences are night (mean brightness 42/255), and the test night sequences
  are darker (25–38). UP-COUNT has none, so a 50% share leaves night at under
  2% of training samples. Neither validation set contains night footage.
- **The darkness proxy failed.** Fold-0 validation frames darkened to test-night
  brightness did not reproduce the collapse; every model kept F1 0.78–0.85.
  Real night footage differs in lighting, glare and blur, not just brightness.
- **This is not forgetting.** On the night sequences it trained on, the final
  model keeps F1 0.86, matching the released weights' 0.87. It generalizes
  worse to unseen hard frames: precision rose, recall fell.
- **The undercount is not caused by UP-COUNT.** It persisted at a 20% share,
  pointing to the fine-tuning setup itself. Candidate causes, untested:
  BatchNorm statistics in the decoder at batch 2, a learning-rate restart on
  converged weights, and fewer VisDrone frames than the full DroneCrowd videos.

Lessons for this kind of work:

- **Coverage:** validation data must cover the operating conditions (night,
  altitude, density), not just hold out sequences.
- **Pre-registration:** a decision rule fixed before testing turns a
  disappointing result into a clean decision.
- **Disclosure:** a second test look must be reported as one.

## 15. Multi-person tracking

Per-frame counts cannot say how many distinct people passed through a scene,
so `scripts/track_video.py` adds the uav-dot-track tracker (Ptak & Kraft, IEEE
Access 2026, Apache-2.0) on top of the detector. The tracker's code is used
unchanged; only the detection path is ours:

- **Detection:** fp16 UAV-Dot with GPU preprocessing, from the authors' fork,
  which also returns decoder features. The authors' path is fp32 with CPU
  preprocessing.
- **Feature transfer:** the 16×544×960 decoder feature map is copied to the CPU
  only when DDCF is enabled. Copying it in fp32 (6 ms) proved faster than fp16
  plus a CPU conversion (15 ms).
- **MXNet:** pyECO imports MXNet for a ResNet feature that DDCF never uses, so a
  stub module replaces it at run time instead of installing the abandoned
  framework.

The variants follow the paper's table, each adding one option to a Kalman-filter
and Hungarian baseline:

- **CMC:** camera-motion compensation by sparse optical flow.
- **ALT:** altitude-aware gating, from the per-frame GPS logs.
- **CLS:** a small head classifier that vets new tracks.
- **DDCF:** ECO correlation filters on decoder features, which recover confirmed
  tracks with no matching detection.

All variants keep the authors' defaults: a track is confirmed after 30 frames
and dropped after 60 missing frames.

**Data and protocol.** Six UP-COUNT-TRACK test videos (CC BY-NC 4.0) were
fetched from the 19.7 GB archive with HTTP range requests
(`scripts/download_upcount_track.py`). They are 4K at 30 FPS, 12–23 s long and
filmed at 30–76 m. Detection uses the released UP-COUNT checkpoint at its 4K
input with UP-COUNT normalization, as the authors do. `scripts/evaluate_tracking.py`
runs the authors' TrackEval configuration (20×20 boxes, no preprocessing,
similarity threshold 0) against the published MOT ground truth. That ground
truth is identical to the copy bundled in the tracker repository.

Unique-count error is |predicted − true identities| per video, averaged (the
paper's Tr-MAE); its normalized form divides by the true count (Tr-nMAE). Speed
is the median per frame of the 4K-input detector plus the tracker, on an RTX
4060 Laptop GPU and its CPU.

| Five typical videos (9–38 people per frame) | HOTA | MOTA | IDF1 | ID switches | Tr-MAE | Tr-nMAE | Detection | Tracking | Speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | **0.421** | 0.697 | 0.806 | 69 | 11.8 | 0.32 | 38.0 ms | **2.1 ms** | **24.9 FPS** |
| + CMC | 0.410 | 0.685 | 0.783 | 96 | 14.6 | 0.39 | 37.3 ms | 10.5 ms | 20.9 FPS |
| + CMC + ALT | **0.423** | **0.711** | 0.821 | 92 | 9.2 | 0.28 | 37.4 ms | 10.0 ms | 21.1 FPS |
| + CMC + ALT + CLS | 0.411 | 0.674 | 0.804 | 89 | 7.0 | 0.24 | 37.4 ms | 9.9 ms | 21.1 FPS |
| + CMC + ALT + CLS + DDCF | 0.401 | 0.662 | **0.894** | **27** | **3.0** | **0.10** | 43.0 ms | 52.2 ms | 10.5 FPS |

| Dense video 0016 (1,022 people per frame, 1,182 identities) | HOTA | MOTA | IDF1 | ID switches | Tr-MAE | Tr-nMAE | Tracking |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.482 | 0.815 | 0.814 | 2,028 | 850 | 0.72 | **135 ms** |
| + CMC + ALT + CLS | 0.493 | 0.805 | 0.851 | 1,665 | 535 | 0.45 | 146 ms |
| + CMC + ALT + CLS + DDCF | **0.501** | 0.775 | **0.963** | **77** | **240** | **0.20** | 3,287 ms |

What the results show:

- **DDCF is the step that matters.** It removes most identity switches and
  brings the number of tracked people within 10% of the truth on typical videos
  (20% on the dense one). HOTA and MOTA barely move or drop slightly, matching
  the paper: DDCF repairs identity continuity, not detection.
- **CMC, ALT and CLS alone are mixed on this subset.** They improve the unique
  count but not the ID switches.
- **Speed limits deployment.** DDCF runs one ECO filter on the CPU per lost
  confirmed track, so its cost grows with crowd size: 19–36 ms per frame on
  sparse videos, 149 ms at 38 people per frame and 3.3 s at 1,022. The full
  tracker suits offline counting of dense crowds; without DDCF the pipeline
  stays above 20 FPS on typical scenes.

Our absolute HOTA (0.40–0.50) is below the paper's full-test-set figure (0.63).
The subset is small and includes the hardest video.

**Fidelity to the authors' pipeline.** `scripts/run_upstream_tracker.py` runs
the authors' unmodified `main.py` on the same videos. That pipeline uses an fp32
detector, resizes and normalizes on the CPU, and copies decoder features every
frame.

| Run | Typical videos: HOTA | ID switches | Tr-MAE | Video 0016: HOTA | ID switches | Tr-MAE | Detection per 4K frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| Authors' pipeline, baseline | 0.423 | 64 | 11.0 | 0.503 | 1,756 | 665 | 221 ms* |
| This pipeline, fp16, baseline | 0.421 | 69 | 11.8 | 0.482 | 2,028 | 850 | **42 ms** |
| This pipeline, fp32, baseline | – | – | – | 0.501 | 1,768 | 691 | 79 ms |
| Authors' pipeline, full | 0.402 | 26 | 3.0 | 0.513 | 50 | 190 | |
| This pipeline, fp16, full | 0.401 | 27 | 3.0 | 0.501 | 77 | 240 | |
| This pipeline, fp32, full | – | – | – | **0.514** | 62 | **189** | |

\* The authors' detection path timed inside our loop on video 0016; that run
reproduces their tracks (HOTA 0.503, 1,754 ID switches).

- **The detections are the same.** On 30 frames of video 0016, 99.3% of fp16
  points lie within 2 px of the authors' points, and F1 at 8 px against the
  ground truth is 0.9086 against 0.9089. fp16 and fp32 give identical points
  when the frame is resized on the GPU.
- **The tracking loop is faithful and deterministic.** With the authors'
  detection numerics our loop reproduces their tracks, and repeated runs give
  identical files.
- **Dense crowds amplify tiny differences.** On about 1,000 people per frame,
  the tracker turns small detection differences into different identity
  assignments. fp32 detection restores the authors' result.
- **Recommended precision:** fp16 for live or typical scenes; fp32
  (`--precision fp32`) for offline analysis of dense crowds, where DDCF
  dominates the run time anyway.

## 16. Conclusion

The repository turns density-style crowd counting into explicit head-point
prediction with two detectors evaluated under one official protocol:

- **P2PNet:** trained here, it is the most accurate counter (test MAE 21.9,
  L-mAP 52.5).
- **UAV-Dot:** with fp16 inference it matches P2PNet's F1 at 3.9 times the
  speed (29 ms per 1080p frame), so it is the practical real-time detector.

Adding UP-COUNT showed that more varied data can remove a scale blind spot
(low-altitude F1 0.37 to 0.65), but also that a validation set without night
footage cannot protect the deployment against a night regression.

The next priorities are:

- night and low-altitude footage in both training and validation;
- the remaining cross-validation folds;
- a controlled study of the fine-tuning calibration shift;
- a tracking evaluation on the full DroneCrowd test videos.
