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
