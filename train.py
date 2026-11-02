from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from vit_yolo.dataset import VisDroneDetection, detection_collate
from vit_yolo.loss import YOLOLoss
from vit_yolo.model import ViTYOLO


@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    train_split: str
    val_split: str
    output: Path
    resume: Optional[Path]
    epochs: int
    image_size: int
    batch_size: int
    accumulate: int
    workers: int
    device: str
    amp: bool
    channels_last: bool
    gradient_checkpointing: bool
    seed: int
    augment: bool
    lr: float
    min_lr: float
    momentum: float
    warmup_momentum: float
    weight_decay: float
    warmup_epochs: float
    grad_clip: float
    num_classes: int
    neck_channels: int
    bifpn_depth: int
    num_heads: int
    width_mult: float
    depth_mult: float
    max_tokens: int
    dropout: float
    box_gain: float
    objectness_gain: float
    class_gain: float
    label_smoothing: float
    limit_train: int
    limit_val: int


# Edit this configuration directly. No command-line arguments are required.
CONFIG = TrainingConfig(
    data_root=Path(r"D:\Pycharm\Code\PycharmProjects\ViT_YOLO\VisDrone2019-DET"),
    train_split="train",
    val_split="val",
    output=Path(r"D:\Pycharm\Code\PycharmProjects\ViT_YOLO\runs\visdrone_mhsa_bifpn"),
    resume=None,
    epochs=3,
    # Use 640 on a 12 GB GPU. Set 1024 when enough memory is available.
    image_size=640,
    batch_size=1,
    accumulate=8,
    workers=8,
    device="cuda" if torch.cuda.is_available() else "cpu",
    amp=True,
    channels_last=True,
    gradient_checkpointing=True,
    seed=0,
    augment=True,
    lr=0.02,
    min_lr=0.002,
    momentum=0.937,
    warmup_momentum=0.8,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    grad_clip=10.0,
    num_classes=10,
    neck_channels=512,
    bifpn_depth=3,
    num_heads=4,
    width_mult=1.0,
    depth_mult=1.0,
    max_tokens=4096,
    dropout=0.0,
    box_gain=5.0,
    objectness_gain=1.0,
    class_gain=0.5,
    label_smoothing=0.0,
    limit_train=0,
    limit_val=0,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_model(args: TrainingConfig) -> ViTYOLO:
    return ViTYOLO(
        num_classes=args.num_classes,
        neck_channels=args.neck_channels,
        bifpn_depth=args.bifpn_depth,
        num_heads=args.num_heads,
        width_mult=args.width_mult,
        depth_mult=args.depth_mult,
        max_tokens=args.max_tokens,
        dropout=args.dropout,
    )


def build_optimizer(
    model: nn.Module,
    args: TrainingConfig,
) -> SGD:
    return SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )


def learning_rate_at_step(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    minimum_lr: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    cosine_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / cosine_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_lr + (base_lr - minimum_lr) * cosine


def set_optimizer_state(
    optimizer: SGD,
    learning_rate: float,
    momentum: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        group["momentum"] = momentum


def train_one_epoch(
    model: ViTYOLO,
    loader: DataLoader,
    loss_function: YOLOLoss,
    optimizer: SGD,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    args: TrainingConfig,
    epoch: int,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
) -> Tuple[Dict[str, float], int]:
    model.train()
    totals = {"loss": 0.0, "box": 0.0, "objectness": 0.0, "class": 0.0}
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    optimizer_steps_per_epoch = math.ceil(len(loader) / args.accumulate)
    first_step = (epoch - 1) * optimizer_steps_per_epoch

    for batch_index, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = [target.to(device, non_blocking=True) for target in targets]
        if args.channels_last:
            images = images.to(memory_format=torch.channels_last)

        step_index = first_step + batch_index // args.accumulate
        learning_rate = learning_rate_at_step(
            step_index,
            total_steps,
            warmup_steps,
            args.lr,
            args.min_lr,
        )
        momentum = (
            args.warmup_momentum
            if step_index < warmup_steps
            else args.momentum
        )
        set_optimizer_state(optimizer, learning_rate, momentum)

        with torch.autocast(
            device_type=device.type,
            enabled=args.amp and device.type == "cuda",
        ):
            predictions = model(images)
            loss, parts = loss_function(
                predictions,
                targets,
                model.anchors,
            )
            scaled_loss = loss / args.accumulate

        scaler.scale(scaled_loss).backward()
        should_step = (
            (batch_index + 1) % args.accumulate == 0
            or batch_index + 1 == len(loader)
        )
        if should_step:
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        totals["loss"] += float(loss.detach().item())
        totals["box"] += parts["box"]
        totals["objectness"] += parts["objectness"]
        totals["class"] += parts["class"]

        if (batch_index + 1) % 50 == 0 or batch_index + 1 == len(loader):
            average_loss = totals["loss"] / float(batch_index + 1)
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"epoch {epoch:03d} "
                f"batch {batch_index + 1:04d}/{len(loader):04d} "
                f"loss {average_loss:.4f} "
                f"lr {learning_rate:.5f} "
                f"{(batch_index + 1) / elapsed:.2f} batch/s",
                flush=True,
            )

    count = max(len(loader), 1)
    averaged = {key: value / count for key, value in totals.items()}
    return averaged, global_step


@torch.no_grad()
def validate(
    model: ViTYOLO,
    loader: DataLoader,
    loss_function: YOLOLoss,
    device: torch.device,
    amp: bool,
) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "box": 0.0, "objectness": 0.0, "class": 0.0}
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = [target.to(device, non_blocking=True) for target in targets]
        with torch.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
        ):
            predictions = model(images)
            loss, parts = loss_function(
                predictions,
                targets,
                model.anchors,
            )
        totals["loss"] += float(loss.item())
        totals["box"] += parts["box"]
        totals["objectness"] += parts["objectness"]
        totals["class"] += parts["class"]
    count = max(len(loader), 1)
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: ViTYOLO,
    optimizer: SGD,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    validation_loss: float,
    args: TrainingConfig,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "validation_loss": validation_loss,
        "anchors": model.anchors,
        "strides": model.strides,
        "model_kwargs": {
            "num_classes": args.num_classes,
            "neck_channels": args.neck_channels,
            "bifpn_depth": args.bifpn_depth,
            "num_heads": args.num_heads,
            "width_mult": args.width_mult,
            "depth_mult": args.depth_mult,
            "max_tokens": args.max_tokens,
            "dropout": args.dropout,
        },
        "training_args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main() -> None:
    args = CONFIG
    set_seed(args.seed)
    device = select_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    train_dataset = VisDroneDetection(
        root=args.data_root,
        split=args.train_split,
        image_size=args.image_size,
        augment=args.augment,
        limit=args.limit_train,
    )
    try:
        val_dataset = VisDroneDetection(
            root=args.data_root,
            split=args.val_split,
            image_size=args.image_size,
            augment=False,
            limit=args.limit_val,
        )
    except FileNotFoundError:
        val_dataset = None
        print("Validation split not found; validation loss is disabled.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=detection_collate,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            collate_fn=detection_collate,
            persistent_workers=args.workers > 0,
        )

    model = build_model(args).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.gradient_checkpointing:
        model.configure_gradient_checkpointing(True)
    optimizer = build_optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    loss_function = YOLOLoss(
        num_classes=args.num_classes,
        anchors=model.anchors,
        strides=model.strides,
        box_gain=args.box_gain,
        objectness_gain=args.objectness_gain,
        class_gain=args.class_gain,
        label_smoothing=args.label_smoothing,
    )

    start_epoch = 1
    global_step = 0
    best_validation_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_validation_loss = float(
            checkpoint.get("validation_loss", float("inf"))
        )
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")

    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / args.accumulate
    )
    total_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = int(optimizer_steps_per_epoch * args.warmup_epochs)

    print(
        f"device={device} train_images={len(train_dataset)} "
        f"val_images={0 if val_dataset is None else len(val_dataset)} "
        f"steps_per_epoch={optimizer_steps_per_epoch}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics, global_step = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            scaler,
            device,
            args,
            epoch,
            global_step,
            total_steps,
            warmup_steps,
        )

        validation_metrics = None
        if val_loader is not None:
            validation_metrics = validate(
                model,
                val_loader,
                loss_function,
                device,
                args.amp,
            )

        validation_loss = (
            validation_metrics["loss"]
            if validation_metrics is not None
            else train_metrics["loss"]
        )
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_box={train_metrics['box']:.4f} "
            f"train_obj={train_metrics['objectness']:.4f} "
            f"train_cls={train_metrics['class']:.4f} "
            f"val_loss={validation_loss:.4f}",
            flush=True,
        )

        save_checkpoint(
            args.output / "last.pt",
            model,
            optimizer,
            scaler,
            epoch,
            global_step,
            validation_loss,
            args,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                args.output / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                global_step,
                validation_loss,
                args,
            )


if __name__ == "__main__":
    main()
