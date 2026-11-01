---
license: other
license_name: p2pnet-academic-research-only
license_link: LICENSE
library_name: pytorch
pipeline_tag: object-detection
inference: false
tags:
  - aerial-imagery
  - computer-vision
  - crowd-counting
  - head-detection
  - p2pnet
  - point-localization
  - pytorch
  - uav
  - visdrone
---

# P2PNet VisDrone head-point localization

This is an inference-only P2PNet checkpoint fine-tuned on fold 0 of a
deterministic, sequence-level VisDrone2020-CC split. It returns one
`(x, y, confidence)` proposal per retained head and uses the number of retained
points as the crowd count.

Source and complete training documentation:
[`owaisasghar/uav-crowd-counting`](https://github.com/owaisasghar/uav-crowd-counting).

The generic Hugging Face object-detection widget is disabled because its
bounding-box output contract does not match point localization. Use the
included inference script or the
[Colab notebook](https://colab.research.google.com/github/owaisasghar/uav-crowd-counting/blob/main/notebooks/demo.ipynb),
which compares this model with UAV-Dot side by side.

## Results

### Independent test set

These results use the official VisDrone2020-CC test sequences (900 frames from 30
held-out sequences, 137 people per frame), threshold 0.50 and the DroneCrowd
localization protocol:

| Metric | P2PNet (this model) | UAV-Dot, released weights |
|---|---:|---:|
| Count MAE / RMSE | **21.86** / **29.58** | 25.04 / 33.66 |
| F1 at 16 px | 0.725 | 0.726 |
| L-mAP (DroneCrowd toolkit, 1–25 px) | **52.49** | 51.00 |
| fp16 latency, 1920×1080, RTX 4060 Laptop GPU | 114.7 ms | **29.3 ms** |

### Validation fold

The values below use the 510-frame validation fold that also selected the
checkpoint. They are not independent-test results.

| Metric | Value |
|---|---:|
| Count MAE at threshold 0.50 | **10.0255** |
| Count RMSE at threshold 0.50 | **13.0832** |
| Precision at 16 px | **0.7988** |
| Recall at 16 px | **0.8315** |
| F1 at 16 px | **0.8148** |
| Ranked AP at 16 px | **0.8558** |
| Validation images | 510 |
| Validation head points | 72,171 |

Point metrics use one-to-one Hungarian matching and a fixed 16-pixel radius.
Ranked AP is fixed-radius AP, not a head-size-normalized nAP metric. Lower MAE
and RMSE are better; the count metrics are not classification accuracy.

## Files

| File | Purpose |
|---|---|
| `model.safetensors` | Inference-only model state |
| `p2pnet_runtime.py` | Standalone architecture and preprocessing |
| `inference.py` | Command-line image inference |
| `config.json` | Architecture and preprocessing metadata |
| `export_manifest.json` | Export checksum and provenance |
| `LICENSE` | Academic-research-only terms |

The original `.pth` training checkpoint also contains optimizer and scheduler
state. This model repository intentionally distributes only model tensors in
SafeTensors format.

Verified inference export:

```text
File: model.safetensors
Size: 86,362,656 bytes
SHA-256: acb2f7196538fff3187b2e02fd9629802a9f7d5faec3d8bdbe9c47cd6ed003c9
Tensors: 123
Source checkpoint epoch: 49 (zero-based value 48)
```

## Usage

Install the dependencies and run the included script from the model repository
directory:

```bash
python -m pip install -r requirements.txt
python inference.py \
  --weights model.safetensors \
  --input aerial_image.jpg \
  --output predicted_points.png \
  --json-output predicted_points.json \
  --threshold 0.5 \
  --max-side 1920 \
  --device auto
```

The JSON output contains the predicted count, inference time, device, input
scale, and every retained point with its confidence score.

## Training data and split

- Dataset: VisDrone2020-CC, obtained separately from its authorized source.
- Training: 1,950 frames and 290,556 annotated head points.
- Validation: 510 frames and 72,171 annotated head points.
- Split: sequence-level fold 0 of 5, seed 42.
- Best epoch: 49; training stopped at epoch 79 by early stopping.

VisDrone images and annotations are not included in this model repository.

## Relationship to the referenced UAV paper

This model is an extension inspired by aerial crowd-counting research, not a
reproduction of the model in Ptak et al., Remote Sensing 2022. That work used
density regression, optical-flow channels, multiple architectures and input
resolutions, embedded-device measurements, and five-fold validation. This
checkpoint instead adapts the point-based P2PNet architecture to return
individual head locations.

## Limitations

- Only one of five planned sequence-level folds has been trained.
- Threshold selection and checkpoint selection used the same validation fold.
- Full-resolution forward inference averaged 243.14 ms on an RTX 4060 Laptop
  GPU, approximately 4.11 forward passes per second before most I/O.
- Small, occluded, blurred, or low-contrast heads may be missed.
- Background structures may create false positives.
- Performance is not established for new cameras, altitudes, locations,
  weather, thermal imagery, or the separate task of vehicle detection.

## Intended use

Academic research on aerial crowd counting and explicit point localization.
The model is not intended for identity recognition, demographic inference,
autonomous law-enforcement decisions, or unvalidated safety-critical use.

## License

P2PNet redistribution and use are restricted to academic research. Review
`LICENSE` before downloading, using, or redistributing the checkpoint. This is
not an MIT, Apache, or commercially permissive model.

## Citation

Please cite the upstream P2PNet work:

```bibtex
@inproceedings{song2021p2pnet,
  title={Rethinking Counting and Localization in Crowds: A Purely Point-Based Framework},
  author={Song, Qingyu and Wang, Changan and Jiang, Zhengkai and Wang, Yabiao and Tai, Ying and Wang, Chengjie and Li, Jilin and Huang, Feiyue and Wu, Yang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```
