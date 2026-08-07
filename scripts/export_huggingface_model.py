from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_ROOT = PROJECT_ROOT / "huggingface"
sys.path.insert(0, str(HF_ROOT))

from p2pnet_runtime import P2PNet  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export inference-only P2PNet weights as SafeTensors."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=HF_ROOT / "model.safetensors",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --force")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain a 'model' state dictionary")

    state_dict = {
        key: value.detach().cpu().contiguous()
        for key, value in checkpoint["model"].items()
    }
    model = P2PNet()
    model.load_state_dict(state_dict, strict=True)
    state_dict = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "architecture": "P2PNet VGG16-BN",
        "format": "pt",
        "source_checkpoint_epoch_zero_based": str(checkpoint.get("epoch", "unknown")),
        "training_dataset": "VisDrone2020-CC",
    }
    save_file(state_dict, str(args.output), metadata=metadata)
    manifest = {
        "file": args.output.name,
        "bytes": args.output.stat().st_size,
        "sha256": _sha256(args.output),
        "tensors": len(state_dict),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_epoch_zero_based": checkpoint.get("epoch"),
        "optimizer_state_included": False,
    }
    manifest_path = args.output.with_name("export_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
