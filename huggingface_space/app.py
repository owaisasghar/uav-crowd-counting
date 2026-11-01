from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path
from types import ModuleType

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from PIL import Image

import uavdot_runtime


MODEL_ID = os.getenv(
    "P2PNET_MODEL_ID",
    "uwaisasghar/p2pnet-visdrone-head-counting",
)
LOCAL_MODEL_DIR = os.getenv("P2PNET_MODEL_DIR")
UAVDOT_CHECKPOINT_DIR = Path(os.getenv("UAVDOT_CHECKPOINT_DIR", Path.home() / ".cache" / "uavdot"))
REPOSITORY_URL = "https://github.com/owaisasghar/uav-p2pnet-head-counting"
_MODEL_LOCK = threading.Lock()
_RUNTIME: ModuleType | None = None
_MODEL: torch.nn.Module | None = None
_DEVICE: torch.device | None = None
_UAVDOT: torch.nn.Module | None = None
_UAVDOT_DEVICE: torch.device | None = None


def _model_directory() -> Path:
    if LOCAL_MODEL_DIR:
        path = Path(LOCAL_MODEL_DIR).expanduser().resolve()
        if not path.is_dir():
            raise RuntimeError(f"P2PNET_MODEL_DIR is not a directory: {path}")
        return path
    return Path(
        snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "p2pnet_runtime.py",
            ],
        )
    )


def _load_runtime(path: Path) -> ModuleType:
    module_path = path / "p2pnet_runtime.py"
    spec = importlib.util.spec_from_file_location("p2pnet_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runtime module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_model() -> tuple[ModuleType, torch.nn.Module, torch.device]:
    global _RUNTIME, _MODEL, _DEVICE
    with _MODEL_LOCK:
        if _MODEL is None:
            model_dir = _model_directory()
            _RUNTIME = _load_runtime(model_dir)
            _MODEL, _DEVICE = _RUNTIME.load_model(
                model_dir / "model.safetensors",
                device="auto",
            )
    assert _RUNTIME is not None and _MODEL is not None and _DEVICE is not None
    return _RUNTIME, _MODEL, _DEVICE


def _get_uavdot() -> tuple[torch.nn.Module, torch.device]:
    global _UAVDOT, _UAVDOT_DEVICE
    with _MODEL_LOCK:
        if _UAVDOT is None:
            # The released weights are fetched from the UAV-Dot authors' mirror and checked by SHA-256.
            checkpoint = uavdot_runtime.download_checkpoint(UAVDOT_CHECKPOINT_DIR)
            _UAVDOT, _UAVDOT_DEVICE = uavdot_runtime.load_model(checkpoint, device="auto")
    assert _UAVDOT is not None and _UAVDOT_DEVICE is not None
    return _UAVDOT, _UAVDOT_DEVICE


def predict(
    image: Image.Image | None,
    models: str,
    uavdot_threshold: float,
    p2pnet_threshold: float,
    max_side: int,
    use_half: bool,
) -> tuple[Image.Image | None, Image.Image | None, dict[str, object]]:
    if image is None:
        raise gr.Error("Upload an aerial RGB image first.")
    summary: dict[str, object] = {}
    uavdot_image = p2pnet_image = None
    if models in ("Both", "UAV-Dot"):
        model, device = _get_uavdot()
        result = uavdot_runtime.predict_points(
            model, image, device, threshold=float(uavdot_threshold), use_half=bool(use_half)
        )
        uavdot_image = uavdot_runtime.annotate_image(image, result["points"], float(uavdot_threshold))
        summary["uav_dot"] = uavdot_runtime.serializable_result(result)
    if models in ("Both", "P2PNet"):
        runtime, model, device = _get_model()
        result = runtime.predict_points(
            model,
            image,
            device,
            threshold=float(p2pnet_threshold),
            max_side=int(max_side),
            use_half=bool(use_half),
        )
        p2pnet_image = runtime.annotate_image(image, result["points"], float(p2pnet_threshold))
        summary["p2pnet"] = runtime.serializable_result(result) | {"model": MODEL_ID}
    summary["notice"] = "Research output. Validate on the target drone/camera before use."
    return uavdot_image, p2pnet_image, summary


DESCRIPTION = f"""
# 🚁 UAV crowd counting: UAV-Dot vs P2PNet

Upload an aerial image. Both models mark every detected head as a point, and
the count is the number of points. Neither model identifies people.

| VisDrone2020-CC test set (900 frames) | Count MAE | F1 @ 16 px | L-mAP | fp16 latency, RTX 4060 Laptop |
|---|---:|---:|---:|---:|
| **UAV-Dot** (released weights) | 25.0 | 0.726 | 51.0 | **29 ms** |
| **P2PNet** (trained in this project) | **21.9** | 0.725 | **52.5** | 115 ms |

This Space may run on a CPU, so its timings are slower than the GPU figures
above. The test metrics use complete 1920×1080 frames; P2PNet's demo default
of a 1280-pixel maximum side trades some tiny-head recall for speed.
Code, evaluation protocol and limitations: [{REPOSITORY_URL.split('/')[-1]}]({REPOSITORY_URL}).
"""


with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                image_mode="RGB",
                sources=["upload", "webcam", "clipboard"],
                label="Aerial RGB image",
            )
            models = gr.Radio(
                choices=["Both", "UAV-Dot", "P2PNet"],
                value="Both",
                label="Models",
            )
            uavdot_threshold = gr.Slider(
                minimum=0.05,
                maximum=0.9,
                value=0.2,
                step=0.05,
                label="UAV-Dot confidence threshold",
                info="0.20 is the evaluated protocol.",
            )
            p2pnet_threshold = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.05,
                label="P2PNet confidence threshold",
                info="0.50 is the evaluated protocol.",
            )
            max_side = gr.Dropdown(
                choices=[960, 1280, 1920],
                value=1280,
                label="P2PNet maximum inference side",
                info="UAV-Dot always runs at 1920×1088.",
            )
            use_half = gr.Checkbox(
                value=True,
                label="Use FP16 when CUDA is available",
            )
            run_button = gr.Button("Count heads", variant="primary")
        with gr.Column(scale=2):
            with gr.Row():
                uavdot_output = gr.Image(label="UAV-Dot (blue points)")
                p2pnet_output = gr.Image(label="P2PNet (red points)")
            output_json = gr.JSON(label="Counts, timing and point coordinates")

    run_button.click(
        fn=predict,
        inputs=[input_image, models, uavdot_threshold, p2pnet_threshold, max_side, use_half],
        outputs=[uavdot_output, p2pnet_output, output_json],
        api_name="predict",
        concurrency_limit=1,
    )
    gr.Markdown(
        "Academic research only. Performance may change with altitude, camera, "
        "view angle, blur, weather, night-time lighting, crowd density, and "
        "geographic domain. UAV-Dot code and weights: "
        "[up-count/uav-dot](https://github.com/up-count/uav-dot) (Apache-2.0)."
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        theme=gr.themes.Soft(primary_hue="red", secondary_hue="blue")
    )
