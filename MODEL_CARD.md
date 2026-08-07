# Model card: P2PNet VisDrone fold 0

## Model details

| Field | Value |
|---|---|
| Model name | P2PNet VisDrone fold 0 |
| Release | `v1.0.0` |
| Architecture | P2PNet with VGG16-BN backbone |
| Task | Head-point localization and crowd counting |
| Framework | PyTorch |
| Training domain | VisDrone2020-CC aerial RGB imagery |
| Checkpoint | `best_mae.pth` |
| Best epoch | 49 |
| Confidence threshold evaluated | 0.5 |
| Upstream source | Tencent P2PNet, ICCV 2021 |
| License | Academic research only; see upstream `LICENSE` |

## Checkpoint integrity

```text
File: best_mae.pth
Size: 230,238,136 bytes
SHA-256: 8add58d38e1e0bb2d0de96500073eda9171d1af796488b5faf6e7b218df53233
```

The checkpoint contains the model state plus optimizer, scheduler, and epoch
metadata. Inference only loads the `model` state dictionary.

## Inference-only SafeTensors export

The portfolio package includes `scripts/export_huggingface_model.py`, which
strictly validates the architecture and exports only the 123 model tensors:

```text
File: model.safetensors
Size: 86,362,656 bytes
SHA-256: acb2f7196538fff3187b2e02fd9629802a9f7d5faec3d8bdbe9c47cd6ed003c9
Optimizer state included: no
Source checkpoint epoch: 49 (zero-based value 48)
```

The standalone runtime was compared tensor-for-tensor with the vendored
runtime on the same input. Both `pred_logits` and `pred_points` had a maximum
absolute difference of 0.0. A held-out full-resolution inference produced the
same 150 retained points at threshold 0.50.

The model-repository package is documented in
[`huggingface/README.md`](huggingface/README.md). Its generic bounding-box
widget is disabled because this checkpoint returns points rather than boxes.

## Intended use

- Academic analysis of crowd counts in aerial RGB images.
- Research on explicit head-point localization.
- Reproducible evaluation on the documented VisDrone fold.
- Initialization for additional aerial-domain fine-tuning.

## Out-of-scope use

- Identity recognition, demographic inference, or individual profiling.
- Fully autonomous safety or law-enforcement decisions.
- Guaranteed real-time processing.
- Unvalidated deployment on different altitudes, cameras, environments, or
  thermal imagery.
- Vehicle bounding-box detection; this checkpoint predicts heads only.

## Input and output

Input is a three-channel RGB image normalized with ImageNet mean and standard
deviation. The inference wrapper pads dimensions to a multiple of 128.

The network returns point proposals and logits. After a 0.5 confidence filter,
the wrapper emits:

```json
{
  "predicted_head_count": 150,
  "points": [
    {"x": 1478.10, "y": 3.75, "confidence": 0.8249}
  ]
}
```

## Training data

The sequence-level fold used 1,950 training frames with 290,556 annotated head
points. Validation used 510 frames with 72,171 points. VisDrone data is not
included in the repository or model release.

## Metrics

| Metric | Value |
|---|---:|
| Validation MAE | **10.0255** |
| Validation RMSE | **13.0832** |
| Precision @ 16 px | **0.7988** |
| Recall @ 16 px | **0.8315** |
| F1 @ 16 px | **0.8148** |
| Ranked AP @ 16 px | **0.8558** |
| Validation images | 510 |

Count values use the fixed 0.50 training-log protocol. Boundary-aware deployment
counting gives MAE 10.0235 and RMSE 13.0795. Localization uses one-to-one
Hungarian matching with a 16-pixel maximum Euclidean distance. Ranked AP uses
in-bounds proposals scoring above 0.001. It is a fixed-radius AP, not
head-size-normalized nAP. “Accuracy” should not be reported as a percentage
from the count values.

Validation-only calibration selected threshold 0.60 for MAE 9.2902 and
threshold 0.55 for F1@16px 0.8160. Those tuned values are not independent-test
estimates.

## Limitations and risks

- Small, occluded, blurred, or low-contrast heads can be missed.
- Repetitive background structures can create false positives.
- The validation result is from one fold and may overstate generalization.
- Performance varies with altitude, ground sampling distance, camera angle,
  compression, weather, and geographic domain.
- Validation-tuned thresholds may not transfer to a new deployment.
- Training curves show overfitting after the best epoch.
- Full-resolution inference is computationally expensive.
- The custom runtime is required; this is not a native Transformers pipeline.
- Crowd monitoring creates privacy and governance risks even when the model
  does not perform identity recognition.

## Recommended evaluation before deployment

1. Create a held-out set from the actual drone, camera, altitude, and location.
2. Repeat count MAE/RMSE and point precision/recall/F1 under the documented
   matching radii.
3. Sweep the confidence threshold without using the final test set.
4. Report failure cases by density, altitude, view angle, illumination, and
   motion blur.
5. Measure end-to-end latency including decoding, preprocessing, drawing, and
   encoding.
6. Establish retention, access-control, and human-oversight policies.
