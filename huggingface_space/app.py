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


MODEL_ID = os.getenv(
    "P2PNET_MODEL_ID",
    "owaisasghar/p2pnet-visdrone-head-counting",
)
LOCAL_MODEL_DIR = os.getenv("P2PNET_MODEL_DIR")
_MODEL_LOCK = threading.Lock()
_RUNTIME: ModuleType | None = None
_MODEL: torch.nn.Module | None = None
_DEVICE: torch.device | None = None


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


def predict(
    image: Image.Image | None,
    threshold: float,
    max_side: int,
    use_half: bool,
) -> tuple[Image.Image, dict[str, object]]:
    if image is None:
        raise gr.Error("Upload an aerial RGB image first.")
    runtime, model, device = _get_model()
    result = runtime.predict_points(
        model,
        image,
        device,
        threshold=float(threshold),
        max_side=int(max_side),
        use_half=bool(use_half),
    )
    annotated = runtime.annotate_image(image, result["points"], float(threshold))
    summary = runtime.serializable_result(result)
    summary["model"] = MODEL_ID
    summary["notice"] = (
        "Research output. Validate on the target drone/camera before use."
    )
    return annotated, summary


DESCRIPTION = """
# 🚁 UAV P2PNet head-point localization

Upload an aerial image and run direct point-proposal inference. Each red marker
is one retained head proposal; the count is the number of markers. This is not
a density-map visualization and it does not identify people.

The checkpoint was evaluated on one sequence-level VisDrone2020-CC validation
fold: **MAE 10.0255**, **F1@16 px 0.8148** at threshold 0.50. These are not
independent-test or cross-domain results. The reported metrics use complete
1,920×1,080 frames; the demo defaults to a 1,280-pixel maximum side to reduce
interactive latency and memory use.
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
            threshold = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.05,
                label="Confidence threshold",
                info="0.50 is the fixed reported protocol; 0.60 was validation-tuned for count MAE.",
            )
            max_side = gr.Dropdown(
                choices=[960, 1280, 1920],
                value=1280,
                label="Maximum inference side",
                info="Higher resolution may improve tiny-head recall but costs latency and memory.",
            )
            use_half = gr.Checkbox(
                value=True,
                label="Use FP16 when CUDA is available",
            )
            run_button = gr.Button("Detect individual heads", variant="primary")
        with gr.Column(scale=2):
            output_image = gr.Image(label="Detected head points")
            output_json = gr.JSON(label="Count, timing, and point coordinates")

    run_button.click(
        fn=predict,
        inputs=[input_image, threshold, max_side, use_half],
        outputs=[output_image, output_json],
        api_name="predict",
        concurrency_limit=1,
    )
    gr.Markdown(
        "Academic research only. Performance may change with altitude, camera, "
        "view angle, blur, weather, crowd density, and geographic domain."
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        theme=gr.themes.Soft(primary_hue="red", secondary_hue="blue")
    )
