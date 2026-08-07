from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from p2pnet_runtime import (
    annotate_image,
    load_model,
    predict_points,
    serializable_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="P2PNet head-point inference")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("model.safetensors"))
    parser.add_argument("--output", type=Path, default=Path("prediction.png"))
    parser.add_argument("--json-output", type=Path, default=Path("prediction.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-side", type=int, default=1920)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()

    image = Image.open(args.input).convert("RGB")
    model, device = load_model(args.weights, device=args.device)
    result = predict_points(
        model,
        image,
        device,
        threshold=args.threshold,
        max_side=args.max_side,
        use_half=args.half,
    )
    annotated = annotate_image(image, result["points"], args.threshold)
    summary = serializable_result(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(args.output)
    args.json_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "points"}, indent=2))


if __name__ == "__main__":
    main()
