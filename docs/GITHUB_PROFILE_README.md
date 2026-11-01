# Hi, I'm Awais Asghar

I build computer-vision and deep-learning systems for aerial imagery, object
localization, counting, and video analytics. My work focuses on reproducible
experiments, honest evaluation, and practical inference pipelines.

## Featured project

### UAV crowd counting: head points, speed and tracking

Counting people in drone footage by locating every head, evaluated under the
official DroneCrowd protocol on 900 held-out test frames.

- Trained P2PNet for drone footage: the most accurate counter tested (MAE 21.9)
- Optimized UAV-Dot inference to 29 ms per 1080p frame, 3.9x faster than P2PNet at the same F1
- Multi-person tracking with 5x faster detection than the reference pipeline;
  unique-person count error cut from 32% to 10%
- A documented negative result: extra training data fixed low-altitude scenes
  but hurt night scenes, so a decision rule set before testing kept the original model
- Side-by-side Gradio demo of both detectors

[Source repository](https://github.com/owaisasghar/uav-crowd-counting)

## Technical focus

`Python` · `PyTorch` · `Computer Vision` · `Object Detection` ·
`Point Localization` · `Crowd Counting` · `Multi-Object Tracking` · `UAV Analytics` · `OpenCV` · `CUDA`

## Current direction

I am extending the aerial analytics work with TensorRT deployment, night and
low-altitude data, multi-fold validation, and aerial vehicle detection.

<!-- Add verified contact, portfolio, LinkedIn, and Hugging Face links here. -->
