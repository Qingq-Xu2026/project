from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from vit_yolo.checkpoint import load_model_from_checkpoint
from vit_yolo.dataset import (
    CLASS_NAMES,
    VisDroneDetection,
    read_visdrone_annotations,
)
from vit_yolo.postprocess import (
    postprocess_model_output,
    unletterbox_boxes,
)


@dataclass(frozen=True)
class EvaluationConfig:
    weights: Path
    data_root: Path
    split: str
    image_size: int
    batch_size: int
    confidence: float
    iou: float
    max_detections: int
    limit: int
    device: str


# Edit this configuration directly. No command-line arguments are required.
CONFIG = EvaluationConfig(
    weights=Path("/root/ViT_YOLO/runs/visdrone_mhsa_bifpn/best.pt"),
    data_root=Path("/root/ViT_YOLO/VisDrone2019-DET"),
    split="val",
    image_size=640,
    batch_size=1,
    confidence=0.001,
    iou=0.65,
    max_detections=300,
    limit=0,
    device="cuda",
)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_ground_truth(
    dataset: VisDroneDetection,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    images: List[Dict[str, object]] = []
    annotations: List[Dict[str, object]] = []
    annotation_id = 1

    for image_id, image_path in enumerate(dataset.samples, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        label_path = dataset.annotations_dir / f"{image_path.stem}.txt"
        for category, x1, y1, x2, y2 in read_visdrone_annotations(label_path):
            box_width = max(x2 - x1, 0.0)
            box_height = max(y2 - y1, 0.0)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category + 1,
                    "bbox": [x1, y1, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return images, annotations


@torch.no_grad()
def main() -> None:
    args = CONFIG
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise SystemExit(
            "pycocotools is required for evaluate.py. "
            "Install it with: python -m pip install pycocotools"
        ) from error

    device = select_device(args.device)
    dataset = VisDroneDetection(
        root=args.data_root,
        split=args.split,
        image_size=args.image_size,
        augment=False,
        limit=args.limit,
    )
    if not dataset.annotations_dir.is_dir():
        raise FileNotFoundError(
            f"Ground-truth annotations not found: {dataset.annotations_dir}"
        )

    model, _ = load_model_from_checkpoint(args.weights, device)
    coco_images, coco_annotations = build_ground_truth(dataset)
    coco_ground_truth = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": category, "name": name}
            for category, name in enumerate(CLASS_NAMES, start=1)
        ],
    }
    coco_predictions: List[Dict[str, object]] = []

    for start in range(0, len(dataset), args.batch_size):
        indices = range(start, min(start + args.batch_size, len(dataset)))
        images = []
        metadata = []
        for index in indices:
            image, _, meta = dataset.load(index, augment=False)
            images.append(image)
            metadata.append(meta)

        tensor = torch.stack(images, dim=0).to(device)
        predictions = model(tensor)
        detections = postprocess_model_output(
            predictions,
            model.anchors,
            model.strides,
            (args.image_size, args.image_size),
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            max_detections=args.max_detections,
        )

        for batch_index, (boxes, scores, labels) in enumerate(detections):
            meta = metadata[batch_index]
            boxes = unletterbox_boxes(
                boxes,
                image_size=(args.image_size, args.image_size),
                original_size=meta["original_size"],
                ratio=float(meta["ratio"]),
                padding=meta["padding"],
            )
            image_id = start + batch_index + 1
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = (float(value) for value in box.tolist())
                coco_predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label) + 1,
                        "bbox": [x1, y1, max(x2 - x1, 0), max(y2 - y1, 0)],
                        "score": float(score),
                    }
                )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(coco_ground_truth, handle)
        ground_truth_path = handle.name
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(coco_predictions, handle)
        prediction_path = handle.name

    try:
        coco = COCO(ground_truth_path)
        if not coco_predictions:
            raise SystemExit("No predictions were produced.")
        results = coco.loadRes(prediction_path)
        evaluator = COCOeval(coco, results, iouType="bbox")
        evaluator.params.imgIds = sorted(
            image["id"] for image in coco_images
        )
        evaluator.params.catIds = list(range(1, 11))
        evaluator.params.maxDets = [1, 10, 100, 500]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        metrics = evaluator.stats
        names = (
            "AP",
            "AP50",
            "AP75",
            "AP-small",
            "AP-medium",
            "AP-large",
            "AR1",
            "AR10",
            "AR100",
            "AR-small",
            "AR-medium",
            "AR-large",
        )
        print("Metrics:")
        for name, value in zip(names, metrics):
            print(f"  {name}: {value:.4f}")
        recall = evaluator.eval["recall"][:, :, 0, -1]
        valid_recall = recall[recall > -1]
        ar500 = float(valid_recall.mean()) if valid_recall.size else -1.0
        print(f"  AR500: {ar500:.4f}")
    finally:
        Path(ground_truth_path).unlink(missing_ok=True)
        Path(prediction_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
