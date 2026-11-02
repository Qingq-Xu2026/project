from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from vit_yolo.checkpoint import load_model_from_checkpoint
from vit_yolo.dataset import CLASS_NAMES, letterbox
from vit_yolo.postprocess import (
    postprocess_model_output,
    unletterbox_boxes,
)


@dataclass(frozen=True)
class PredictionConfig:
    weights: Path
    source: Path
    output: Path
    image_size: int
    confidence: float
    iou: float
    max_detections: int
    device: str


# Edit this configuration directly. No command-line arguments are required.
CONFIG = PredictionConfig(
    weights=Path(r"D:\Pycharm\Code\PycharmProjects\ViT_YOLO\runs\visdrone_mhsa_bifpn\best.pt"),
    source=Path(r"D:\Pycharm\Code\PycharmProjects\ViT_YOLO\VisDrone2019-DET"
                r"VisDrone2019-DET-val\images\0000001_08414_d_0000013.jpg"
                ),
    output=Path(r"D:\Pycharm\Code\PycharmProjects\ViT_YOLO\runs\predict"),
    image_size=1024,
    confidence=0.25,
    iou=0.45,
    max_detections=300,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@torch.no_grad()
def main() -> None:
    args = CONFIG
    device = select_device(args.device)
    model, _ = load_model_from_checkpoint(args.weights, device)
    args.output.mkdir(parents=True, exist_ok=True)

    with Image.open(args.source) as source:
        image = source.convert("RGB")
        original_size = image.size
        letterboxed, ratio, padding = letterbox(
            image,
            (args.image_size, args.image_size),
        )
        tensor = torch.from_numpy(np.array(letterboxed, copy=True))
        tensor = tensor.permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(device)

        predictions = model(tensor)
        detections = postprocess_model_output(
            predictions,
            model.anchors,
            model.strides,
            (args.image_size, args.image_size),
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            max_detections=args.max_detections,
        )[0]

        boxes, scores, labels = detections
        boxes = unletterbox_boxes(
            boxes,
            image_size=(args.image_size, args.image_size),
            original_size=original_size,
            ratio=ratio,
            padding=padding,
        )

        draw = ImageDraw.Draw(image)
        for box, score, label in zip(boxes, scores, labels):
            coordinates = [float(value) for value in box.tolist()]
            draw.rectangle(coordinates, outline="red", width=2)
            caption = f"{CLASS_NAMES[int(label)]} {float(score):.2f}"
            draw.text(
                (coordinates[0], max(coordinates[1] - 12, 0)),
                caption,
                fill="red",
            )

        output_path = args.output / args.source.name
        image.save(output_path)
        print(f"saved {output_path} with {boxes.shape[0]} detections")


if __name__ == "__main__":
    main()
