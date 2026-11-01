"""Train UAV-Dot (DOT + Pixel Distill) on a VisDrone2020-CC sequence fold, optionally adding UP-COUNT.

Runs in the UAV-Dot environment, which provides pytorch-lightning and segmentation-models-pytorch:

    external/uav-dot/.venv/bin/python scripts/train_uavdot.py --name baseline_fold0
    external/uav-dot/.venv/bin/python scripts/train_uavdot.py --time-steps 30   # measure speed only
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from uavdot_data import IMAGE_SIZE, MASK_SIZE, PROJECT_ROOT, UavDotDataModule

from src.model.dot_regressor import DotRegressor  # noqa: E402  (path set up by uavdot_data)


class StepTimer(Callback):
    """Records GPU compute time per training batch and the wall time between batch starts."""

    def __init__(self) -> None:
        self.compute: list[float] = []
        self.wall: list[float] = []
        self._start = self._previous_start = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self._previous_start is not None:
            self.wall.append(now - self._previous_start)
        self._start = self._previous_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        torch.cuda.synchronize()
        self.compute.append(time.perf_counter() - self._start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="uavdot_fold0")
    parser.add_argument("--visdrone-root", type=Path, default=PROJECT_ROOT / "data/raw/VisDrone2020-CC")
    parser.add_argument("--upcount-root", type=Path, default=None)
    parser.add_argument("--upcount-share", type=float, default=0.5, help="fraction of each epoch drawn from UP-COUNT")
    parser.add_argument("--all-sequences", action="store_true",
                        help="also train on the fold's val sequences (final model; val is then seen, use last.ckpt)")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=4, help="upstream uses batch 8 without accumulation")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=30,
                        help="no early stop before this; our epochs are ~3x shorter than upstream's")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="start from these model weights, with a fresh optimizer and schedule")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mosaic", type=float, default=0.8)
    parser.add_argument("--zoom-min", type=float, default=0.5, help="RandomResizedCrop area; upstream 0.5")
    parser.add_argument("--zoom-p", type=float, default=0.4, help="RandomResizedCrop probability; upstream 0.4")
    parser.add_argument("--time-steps", type=int, default=0, help="time this many training steps and exit")
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")
    data = UavDotDataModule(
        args.visdrone_root, args.fold, args.folds, args.seed, args.upcount_root,
        args.batch_size, args.workers, args.mosaic, args.zoom_min, args.zoom_p, args.upcount_share, args.all_sequences,
    )
    model = DotRegressor(
        encoder_name="mit_b2",
        input_channels=3,
        output_channels=1,
        spatial_mode="pixel",
        loss_function="dot",
        lr=args.lr,
        train_steps=data.optimizer_steps_per_epoch(args.accumulate),
        visualize_test_images=False,
        obj_threshold=0.2,
        image_size=IMAGE_SIZE,
        mask_size=MASK_SIZE,
    )
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location="cpu")["state_dict"])
    common = dict(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=args.accumulate,
        benchmark=True,
    )

    if args.time_steps:
        timer = StepTimer()
        trainer = pl.Trainer(
            **common, max_steps=args.time_steps, limit_val_batches=0, num_sanity_val_steps=0,
            logger=False, enable_checkpointing=False, callbacks=[timer],
        )
        trainer.fit(model, data)
        compute = statistics.median(timer.compute[3:])
        wall = statistics.median(timer.wall[3:])
        batches = data.epoch_samples // args.batch_size
        print(json.dumps({**data.summary, "visdrone_val_sequences": len(data.summary["visdrone_val_sequences"])}))
        print(f"batch {args.batch_size} x accumulate {args.accumulate}: compute {compute:.3f} s/batch, "
              f"wall {wall:.3f} s/batch, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        print(f"{batches} batches per epoch -> about {batches * wall / 60:.1f} min of training per epoch "
              f"(plus validation)")
        return

    output = PROJECT_ROOT / "runs" / args.name
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({**vars(args), **data.summary}, indent=2, default=str))
    trainer = pl.Trainer(
        **common,
        max_epochs=args.epochs,
        min_epochs=args.min_epochs,
        logger=TensorBoardLogger(save_dir=str(output), name="tensorboard", version=""),
        callbacks=[
            ModelCheckpoint(
                dirpath=output / "checkpoints",
                filename="epoch{epoch}-f1{val_f1:.4f}",
                monitor="val_f1",
                mode="max",
                auto_insert_metric_name=False,
            ),
            # last.ckpt from its own callback: in Lightning 2.1, save_last on the best-model callback just symlinks
            # to the best checkpoint, so a resume would roll back to the best epoch instead of the latest one.
            ModelCheckpoint(dirpath=output / "checkpoints", save_top_k=0, save_last=True),
            EarlyStopping(monitor="val_f1", mode="max", patience=args.patience),
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(model, data, ckpt_path="last" if (output / "checkpoints" / "last.ckpt").exists() else None)


if __name__ == "__main__":
    main()
