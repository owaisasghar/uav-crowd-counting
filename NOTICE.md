# Source and license notice

This project contains a modified copy of **CrowdCounting-P2PNet** from:

```text
https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
upstream commit: 5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455
```

The upstream copyright and license are preserved at
`third_party/CrowdCounting-P2PNet/LICENSE`. That license restricts P2PNet use
and redistribution to academic research. It is not an Apache or permissive
commercial license. A concise project-root copy of the governing P2PNet terms
is provided in `LICENSE`; the vendored file remains the complete dependency
notice.

Local modifications add VisDrone2020-CC loading, training augmentation,
checkpoint resume state, metrics, early stopping, compatibility fixes, point
inference, and a correction ensuring the point-localization loss participates
in training. The repository also includes an inference-only SafeTensors export
path and Gradio demo scaffold. Modified source files retain their upstream
history through the commit identifier above and are described in `REPORT.md`.

VisDrone2020-CC images and annotations are not redistributed in this
repository. Users are responsible for obtaining the dataset through an
authorized source and following its terms.

## External code used without vendoring

The UAV-Dot scripts (`scripts/train_uavdot.py`, `scripts/uavdot_data.py`,
`scripts/evaluate_dronecrowd.py`, `scripts/benchmark_latency.py`,
`scripts/track_video.py`, `scripts/evaluate_tracking.py`) import code that
`scripts/setup_uavdot.sh` clones into the Git-ignored `external/` directory at
pinned commits. None of it is redistributed here.

| Project | Source | Commit | License |
|---|---|---|---|
| UAV-Dot (DOT + Pixel Distill, DroneCrowd evaluation toolkit) | https://github.com/up-count/uav-dot | `a6d3664b50cc0cc6f1a08848eb294b71753b6600` | Apache-2.0 |
| uav-dot-track (tracker, its UAV-Dot fork, pyECO, TrackEval) | https://github.com/up-count/uav-dot-track | `505a7289b9602524383ae4260912f8e783000ccf` | Apache-2.0 |

`scripts/track_video.py` replaces pyECO's MXNet import with an empty stub
module at run time, because the tracker's DDCF variant uses UAV-Dot decoder
features rather than pyECO's MXNet ResNet feature. No upstream file is modified.

The released UAV-Dot checkpoints are downloaded from the links in the UAV-Dot
README and are not redistributed.

## Datasets

No images, videos or annotations are redistributed. Users must obtain each
dataset from its authors and follow its terms:

- **VisDrone2020-CC / DroneCrowd:** terms set by the VisDrone team.
- **UP-COUNT and UP-COUNT-TRACK:** Creative Commons
  Attribution-NonCommercial 4.0 (https://up-count.github.io/).
