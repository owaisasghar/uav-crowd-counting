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
| Validation images | 510 |

These metrics describe count error. They do not establish point-localization
precision, recall, F1, or nAP. “Accuracy” should not be reported as a percentage
from these values.

## Limitations and risks

- Small, occluded, blurred, or low-contrast heads can be missed.
- Repetitive background structures can create false positives.
- The validation result is from one fold and may overstate generalization.
- Performance varies with altitude, ground sampling distance, camera angle,
  compression, weather, and geographic domain.
- Confidence threshold 0.5 may not be optimal for a new deployment.
- Training curves show overfitting after the best epoch.
- Full-resolution inference is computationally expensive.
- Crowd monitoring creates privacy and governance risks even when the model
  does not perform identity recognition.

## Recommended evaluation before deployment

1. Create a held-out set from the actual drone, camera, altitude, and location.
2. Report count MAE/RMSE and point precision/recall/F1 under a documented
   matching radius.
3. Sweep the confidence threshold without using the final test set.
4. Report failure cases by density, altitude, view angle, illumination, and
   motion blur.
5. Measure end-to-end latency including decoding, preprocessing, drawing, and
   encoding.
6. Establish retention, access-control, and human-oversight policies.
