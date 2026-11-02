from __future__ import annotations

import csv
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

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

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as error:
    raise SystemExit(
        "matplotlib is required. Install it with: "
        "python -m pip install matplotlib"
    ) from error


# ============================================================================
# Configuration
# Edit these values directly. No command-line arguments are required.
# ============================================================================
WEIGHTS = Path(r"D:\Pycharm\Code\PycharmProjects\ViT_YOLO\runs\visdrone_mhsa_bifpn\best.pt")
DATA_ROOT = Path(r"D:\Pycharm\Code\PycharmProjects\ViT_YOLO\VisDrone2019-DET")
OUTPUT_DIR = Path(r"D:\Pycharm\Code\PycharmProjects\ViT_YOLO\runs\evaluation_report")
TRAIN_LOG: Path | None = None

SPLIT = "val"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 1024
BATCH_SIZE = 1
LIMIT = 0

CONFIDENCE = 0.001
DISPLAY_CONFIDENCE = 0.25
IOU = 0.65
MATCH_IOU = 0.5
MAX_DETECTIONS = 300
VISUAL_SAMPLES = 12
# ============================================================================


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU matrix for xyxy boxes."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    width_height = np.clip(bottom_right - top_left, a_min=0.0, a_max=None)
    intersection = width_height[..., 0] * width_height[..., 1]
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None) * np.clip(
        boxes_a[:, 3] - boxes_a[:, 1], 0.0, None
    )
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None) * np.clip(
        boxes_b[:, 3] - boxes_b[:, 1], 0.0, None
    )
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.clip(union, 1e-9, None)


def _greedy_class_match(
    ground_truth_boxes: np.ndarray,
    prediction_boxes: np.ndarray,
    prediction_scores: np.ndarray,
    iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match one class greedily by score and return score order and TP flags."""
    if prediction_boxes.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=bool)
    order = np.argsort(-prediction_scores)
    matched_ground_truth = np.zeros(len(ground_truth_boxes), dtype=bool)
    true_positive = np.zeros(len(prediction_boxes), dtype=bool)

    if ground_truth_boxes.size == 0:
        return order, true_positive[order]

    for prediction_index in order:
        overlaps = box_iou_matrix(
            prediction_boxes[prediction_index : prediction_index + 1],
            ground_truth_boxes,
        )[0]
        best_ground_truth = int(np.argmax(overlaps))
        if (
            overlaps[best_ground_truth] >= iou_threshold
            and not matched_ground_truth[best_ground_truth]
        ):
            matched_ground_truth[best_ground_truth] = True
            true_positive[prediction_index] = True
    return order, true_positive[order]


def per_class_pr_curve(
    records: Sequence[Dict[str, object]],
    class_id: int,
    iou_threshold: float,
) -> Dict[str, object]:
    scores: List[np.ndarray] = []
    true_positives: List[np.ndarray] = []
    total_ground_truth = 0

    for record in records:
        ground_truth_boxes = record["gt_boxes"][
            record["gt_labels"] == class_id
        ]
        prediction_mask = record["pred_labels"] == class_id
        prediction_boxes = record["pred_boxes"][prediction_mask]
        prediction_scores = record["pred_scores"][prediction_mask]
        total_ground_truth += len(ground_truth_boxes)
        order, matched = _greedy_class_match(
            ground_truth_boxes,
            prediction_boxes,
            prediction_scores,
            iou_threshold,
        )
        scores.append(prediction_scores[order])
        true_positives.append(matched)

    if not scores or sum(len(item) for item in scores) == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return {
            "ap": float("nan"),
            "precision": empty,
            "recall": empty,
            "scores": empty,
            "total_ground_truth": total_ground_truth,
        }

    all_scores = np.concatenate(scores)
    all_true_positives = np.concatenate(true_positives)
    order = np.argsort(-all_scores)
    all_true_positives = all_true_positives[order]
    all_scores = all_scores[order]

    cumulative_tp = np.cumsum(all_true_positives)
    cumulative_fp = np.cumsum(~all_true_positives)
    precision = cumulative_tp / np.clip(
        cumulative_tp + cumulative_fp, 1e-9, None
    )
    recall = cumulative_tp / max(total_ground_truth, 1)

    if total_ground_truth == 0:
        average_precision = float("nan")
    else:
        recall_levels = np.linspace(0.0, 1.0, 101)
        interpolated = np.zeros_like(recall_levels)
        for index, level in enumerate(recall_levels):
            valid = recall >= level
            interpolated[index] = precision[valid].max() if valid.any() else 0.0
        average_precision = float(interpolated.mean())

    return {
        "ap": average_precision,
        "precision": precision,
        "recall": recall,
        "scores": all_scores,
        "total_ground_truth": total_ground_truth,
    }


def per_class_counts(
    records: Sequence[Dict[str, object]],
    class_id: int,
    score_threshold: float,
    iou_threshold: float,
) -> Dict[str, float]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    prediction_count = 0

    for record in records:
        ground_truth_boxes = record["gt_boxes"][
            record["gt_labels"] == class_id
        ]
        prediction_mask = (record["pred_labels"] == class_id) & (
            record["pred_scores"] >= score_threshold
        )
        prediction_boxes = record["pred_boxes"][prediction_mask]
        prediction_scores = record["pred_scores"][prediction_mask]
        prediction_count += len(prediction_boxes)

        _, matched = _greedy_class_match(
            ground_truth_boxes,
            prediction_boxes,
            prediction_scores,
            iou_threshold,
        )
        class_true_positive = int(matched.sum())
        true_positive += class_true_positive
        false_positive += len(prediction_boxes) - class_true_positive
        false_negative += len(ground_truth_boxes) - class_true_positive

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(true_positive),
        "fp": float(false_positive),
        "fn": float(false_negative),
        "predictions": float(prediction_count),
        "support": float(true_positive + false_negative),
    }


def build_confusion_matrix(
    records: Sequence[Dict[str, object]],
    score_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    size = len(CLASS_NAMES) + 1
    matrix = np.zeros((size, size), dtype=np.float64)

    for record in records:
        ground_truth_boxes = np.asarray(record["gt_boxes"], dtype=np.float32)
        ground_truth_labels = np.asarray(record["gt_labels"], dtype=np.int64)
        prediction_mask = (
            np.asarray(record["pred_scores"], dtype=np.float32)
            >= score_threshold
        )
        prediction_boxes = np.asarray(
            record["pred_boxes"], dtype=np.float32
        )[prediction_mask]
        prediction_scores = np.asarray(
            record["pred_scores"], dtype=np.float32
        )[prediction_mask]
        prediction_labels = np.asarray(
            record["pred_labels"], dtype=np.int64
        )[prediction_mask]

        matched_ground_truth = np.zeros(len(ground_truth_boxes), dtype=bool)
        order = np.argsort(-prediction_scores)
        for prediction_index in order:
            if ground_truth_boxes.size == 0:
                matrix[-1, prediction_labels[prediction_index]] += 1
                continue
            overlaps = box_iou_matrix(
                prediction_boxes[prediction_index : prediction_index + 1],
                ground_truth_boxes,
            )[0]
            best_ground_truth = int(np.argmax(overlaps))
            if (
                overlaps[best_ground_truth] >= iou_threshold
                and not matched_ground_truth[best_ground_truth]
            ):
                matrix[
                    ground_truth_labels[best_ground_truth],
                    prediction_labels[prediction_index],
                ] += 1
                matched_ground_truth[best_ground_truth] = True
            else:
                matrix[-1, prediction_labels[prediction_index]] += 1

        for ground_truth_index, matched in enumerate(matched_ground_truth):
            if not matched:
                matrix[ground_truth_labels[ground_truth_index], -1] += 1
    return matrix


def compute_coco_metrics(
    records: Sequence[Dict[str, object]],
) -> Dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("pycocotools is not installed; COCO metrics are skipped.")
        return {}

    if not records:
        return {}

    images: List[Dict[str, object]] = []
    annotations: List[Dict[str, object]] = []
    predictions: List[Dict[str, object]] = []
    annotation_id = 1

    for image_id, record in enumerate(records, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": Path(str(record["path"])).name,
                "width": int(record["width"]),
                "height": int(record["height"]),
            }
        )
        for box, label in zip(record["gt_boxes"], record["gt_labels"]):
            x1, y1, x2, y2 = (float(value) for value in box)
            box_width = max(x2 - x1, 0.0)
            box_height = max(y2 - y1, 0.0)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "bbox": [x1, y1, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        for box, score, label in zip(
            record["pred_boxes"],
            record["pred_scores"],
            record["pred_labels"],
        ):
            x1, y1, x2, y2 = (float(value) for value in box)
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "bbox": [x1, y1, max(x2 - x1, 0), max(y2 - y1, 0)],
                    "score": float(score),
                }
            )

    if not predictions or not annotations:
        print("No predictions or annotations were found; COCO metrics are skipped.")
        return {}

    ground_truth_data = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name}
            for index, name in enumerate(CLASS_NAMES, start=1)
        ],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as ground_truth_file:
        json.dump(ground_truth_data, ground_truth_file)
        ground_truth_path = ground_truth_file.name
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as prediction_file:
        json.dump(predictions, prediction_file)
        prediction_path = prediction_file.name

    try:
        coco = COCO(ground_truth_path)
        result = coco.loadRes(prediction_path)
        evaluator = COCOeval(coco, result, iouType="bbox")
        evaluator.params.imgIds = sorted(image["id"] for image in images)
        evaluator.params.catIds = list(range(1, len(CLASS_NAMES) + 1))
        evaluator.params.maxDets = [1, 10, 100, 500]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        stats = evaluator.stats
        recall = evaluator.eval["recall"][:, :, 0, -1]
        valid_recall = recall[recall > -1]
        return {
            "AP": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "AP-small": float(stats[3]),
            "AP-medium": float(stats[4]),
            "AP-large": float(stats[5]),
            "AR1": float(stats[6]),
            "AR10": float(stats[7]),
            "AR100": float(stats[8]),
            "AR500": (
                float(valid_recall.mean()) if valid_recall.size else -1.0
            ),
            "AR-small": float(stats[9]),
            "AR-medium": float(stats[10]),
            "AR-large": float(stats[11]),
        }
    finally:
        Path(ground_truth_path).unlink(missing_ok=True)
        Path(prediction_path).unlink(missing_ok=True)


@torch.no_grad()
def collect_records(
    device: torch.device,
) -> Tuple[List[Dict[str, object]], VisDroneDetection]:
    dataset = VisDroneDetection(
        root=DATA_ROOT,
        split=SPLIT,
        image_size=IMAGE_SIZE,
        augment=False,
        limit=LIMIT,
    )
    model, _ = load_model_from_checkpoint(WEIGHTS, device)
    records: List[Dict[str, object]] = []

    for start in range(0, len(dataset), BATCH_SIZE):
        indices = range(start, min(start + BATCH_SIZE, len(dataset)))
        images: List[torch.Tensor] = []
        metadata: List[Dict[str, object]] = []
        for index in indices:
            image, _, meta = dataset.load(index, augment=False)
            images.append(image)
            metadata.append(meta)

        tensor = torch.stack(images, dim=0).to(device)
        detections = postprocess_model_output(
            model(tensor),
            model.anchors,
            model.strides,
            (IMAGE_SIZE, IMAGE_SIZE),
            confidence_threshold=CONFIDENCE,
            iou_threshold=IOU,
            max_detections=MAX_DETECTIONS,
        )

        for batch_index, (boxes, scores, labels) in enumerate(detections):
            meta = metadata[batch_index]
            boxes = unletterbox_boxes(
                boxes,
                image_size=(IMAGE_SIZE, IMAGE_SIZE),
                original_size=meta["original_size"],
                ratio=float(meta["ratio"]),
                padding=meta["padding"],
            )
            image_path = Path(str(meta["image_path"]))
            annotations = read_visdrone_annotations(
                dataset.annotations_dir / f"{image_path.stem}.txt"
            )
            ground_truth_boxes = np.asarray(
                [annotation[1:] for annotation in annotations],
                dtype=np.float32,
            ).reshape(-1, 4)
            ground_truth_labels = np.asarray(
                [annotation[0] for annotation in annotations],
                dtype=np.int64,
            )
            records.append(
                {
                    "path": str(image_path),
                    "width": int(meta["original_size"][0]),
                    "height": int(meta["original_size"][1]),
                    "gt_boxes": ground_truth_boxes,
                    "gt_labels": ground_truth_labels,
                    "pred_boxes": boxes.detach().cpu().numpy(),
                    "pred_scores": scores.detach().cpu().numpy(),
                    "pred_labels": labels.detach().cpu().numpy(),
                }
            )

        completed = min(start + BATCH_SIZE, len(dataset))
        print(f"processed {completed}/{len(dataset)} images", flush=True)
    return records, dataset


def build_per_class_metrics(
    records: Sequence[Dict[str, object]],
    display_confidence: float,
    match_iou: float,
) -> Tuple[List[Dict[str, object]], Dict[int, Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    curves: Dict[int, Dict[str, object]] = {}

    for class_id, class_name in enumerate(CLASS_NAMES):
        curve = per_class_pr_curve(records, class_id, match_iou)
        counts = per_class_counts(
            records,
            class_id,
            display_confidence,
            match_iou,
        )
        curves[class_id] = curve
        rows.append(
            {
                "class_id": class_id,
                "class": class_name,
                "support": int(counts["support"]),
                "predictions": int(counts["predictions"]),
                "precision": counts["precision"],
                "recall": counts["recall"],
                "f1": counts["f1"],
                "ap50": curve["ap"],
                "tp": int(counts["tp"]),
                "fp": int(counts["fp"]),
                "fn": int(counts["fn"]),
            }
        )
    return rows, curves


def plot_per_class_metrics(
    rows: Sequence[Dict[str, object]],
    output_path: Path,
) -> None:
    class_names = [str(row["class"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.25

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.bar(x - width, [row["precision"] for row in rows], width, label="Precision")
    axis.bar(x, [row["recall"] for row in rows], width, label="Recall")
    axis.bar(x + width, [row["f1"] for row in rows], width, label="F1")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Per-class Precision, Recall, and F1")
    axis.set_xticks(x)
    axis.set_xticklabels(class_names, rotation=35, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_pr_curves(
    curves: Dict[int, Dict[str, object]],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    for class_id, class_name in enumerate(CLASS_NAMES):
        curve = curves[class_id]
        recall = np.asarray(curve["recall"])
        precision = np.asarray(curve["precision"])
        if recall.size == 0:
            continue
        axis.plot(
            recall,
            precision,
            linewidth=1.5,
            label=f"{class_name} AP50={curve['ap']:.3f}",
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-Recall Curves at IoU 0.50")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_confusion_matrix(
    confusion: np.ndarray,
    output_path: Path,
) -> None:
    normalized = confusion / np.clip(
        confusion.sum(axis=1, keepdims=True), 1e-9, None
    )
    labels = list(CLASS_NAMES) + ["background"]
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    axis.set_title("Row-normalized Detection Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_class_counts(
    gt_counts: np.ndarray,
    prediction_counts: np.ndarray,
    output_path: Path,
) -> None:
    x = np.arange(len(CLASS_NAMES))
    width = 0.4
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.bar(x - width / 2, gt_counts, width, label="Ground truth")
    axis.bar(x + width / 2, prediction_counts, width, label="Predictions")
    axis.set_ylabel("Count")
    axis.set_title("Ground-truth and Predicted Object Counts")
    axis.set_xticks(x)
    axis.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_box_distribution(
    ground_truth_boxes: np.ndarray,
    prediction_boxes: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, name, ground_truth, predictions in (
        (
            axes[0],
            "Width",
            ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0],
            prediction_boxes[:, 2] - prediction_boxes[:, 0],
        ),
        (
            axes[1],
            "Height",
            ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1],
            prediction_boxes[:, 3] - prediction_boxes[:, 1],
        ),
    ):
        positive = np.concatenate(
            [
                ground_truth[ground_truth > 0],
                predictions[predictions > 0],
            ]
        )
        if positive.size == 0:
            continue
        bins = np.logspace(
            np.log10(max(positive.min(), 1.0)),
            np.log10(max(positive.max(), 10.0)),
            30,
        )
        axis.hist(
            ground_truth[ground_truth > 0],
            bins=bins,
            alpha=0.55,
            label="Ground truth",
            density=True,
        )
        axis.hist(
            predictions[predictions > 0],
            bins=bins,
            alpha=0.55,
            label="Predictions",
            density=True,
        )
        axis.set_xscale("log")
        axis.set_xlabel(f"Box {name.lower()} in pixels")
        axis.set_ylabel("Density")
        axis.set_title(f"{name} Distribution")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_confidence_distribution(
    scores: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    if scores.size:
        axes[0].hist(scores, bins=50, range=(0.0, 1.0), color="tab:blue")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Prediction count")
    axes[0].set_title("Prediction Confidence Distribution")
    axes[0].grid(alpha=0.2)

    thresholds = np.linspace(0.01, 0.95, 50)
    counts = [(scores >= threshold).sum() for threshold in thresholds]
    axes[1].plot(thresholds, counts, linewidth=2)
    axes[1].set_xlabel("Confidence threshold")
    axes[1].set_ylabel("Retained predictions")
    axes[1].set_title("Predictions vs Confidence Threshold")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_dashboard(
    rows: Sequence[Dict[str, object]],
    gt_counts: np.ndarray,
    prediction_counts: np.ndarray,
    confusion: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(17, 12))

    x = np.arange(len(rows))
    axes[0, 0].bar(x, [row["ap50"] for row in rows], color="tab:blue")
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_title("Per-class AP50")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(
        CLASS_NAMES,
        rotation=35,
        ha="right",
    )
    axes[0, 0].grid(axis="y", alpha=0.2)

    width = 0.4
    axes[0, 1].bar(x - width / 2, gt_counts, width, label="GT")
    axes[0, 1].bar(x + width / 2, prediction_counts, width, label="Predictions")
    axes[0, 1].set_title("Object Counts")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(
        CLASS_NAMES,
        rotation=35,
        ha="right",
    )
    axes[0, 1].grid(axis="y", alpha=0.2)
    axes[0, 1].legend()

    normalized = confusion / np.clip(
        confusion.sum(axis=1, keepdims=True), 1e-9, None
    )
    image = axes[1, 0].imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
    labels = list(CLASS_NAMES) + ["background"]
    axes[1, 0].set_title("Confusion Matrix")
    axes[1, 0].set_xticks(np.arange(len(labels)))
    axes[1, 0].set_yticks(np.arange(len(labels)))
    axes[1, 0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[1, 0].set_yticklabels(labels, fontsize=7)

    axes[1, 1].bar(
        x - width / 2,
        [row["precision"] for row in rows],
        width,
        label="Precision",
    )
    axes[1, 1].bar(
        x + width / 2,
        [row["f1"] for row in rows],
        width,
        label="F1",
    )
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_title("Precision and F1")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(
        CLASS_NAMES,
        rotation=35,
        ha="right",
    )
    axes[1, 1].grid(axis="y", alpha=0.2)
    axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def draw_detection_overlay(
    record: Dict[str, object],
    output_path: Path,
    score_threshold: float,
) -> None:
    colors = plt.get_cmap("tab10")
    with Image.open(str(record["path"])) as source:
        image = source.convert("RGB")
        draw = ImageDraw.Draw(image)

        for box, label in zip(record["gt_boxes"], record["gt_labels"]):
            x1, y1, x2, y2 = (float(value) for value in box)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=3)
            draw.text(
                (x1, max(y1 - 14, 0)),
                f"GT {CLASS_NAMES[int(label)]}",
                fill=(0, 220, 0),
            )

        for box, score, label in zip(
            record["pred_boxes"],
            record["pred_scores"],
            record["pred_labels"],
        ):
            if float(score) < score_threshold:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            color = tuple(
                int(channel * 255)
                for channel in colors(int(label))[:3]
            )
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text(
                (x1, max(y1 - 14, 0)),
                f"{CLASS_NAMES[int(label)]} {float(score):.2f}",
                fill=color,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)


def create_detection_grid(
    overlay_paths: Sequence[Path],
    output_path: Path,
    columns: int = 3,
) -> None:
    if not overlay_paths:
        return
    rows = math.ceil(len(overlay_paths) / columns)
    tile_size = (480, 320)
    canvas = Image.new(
        "RGB",
        (tile_size[0] * columns, tile_size[1] * rows),
        color=(245, 245, 245),
    )

    for index, overlay_path in enumerate(overlay_paths):
        with Image.open(overlay_path) as image:
            thumbnail = ImageOps.fit(image.convert("RGB"), tile_size)
        row = index // columns
        column = index % columns
        x = column * tile_size[0]
        y = row * tile_size[1]
        canvas.paste(thumbnail, (x, y))

    canvas.save(output_path)


def write_metrics(
    output_dir: Path,
    coco_metrics: Dict[str, float],
    rows: Sequence[Dict[str, object]],
    confusion: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "coco": coco_metrics,
        "macro": {
            "precision": float(
                np.nanmean([row["precision"] for row in rows])
            ),
            "recall": float(np.nanmean([row["recall"] for row in rows])),
            "f1": float(np.nanmean([row["f1"] for row in rows])),
            "ap50": float(np.nanmean([row["ap50"] for row in rows])),
        },
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "per_class_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    np.savetxt(
        output_dir / "confusion_matrix.csv",
        confusion,
        delimiter=",",
        fmt="%.0f",
    )


def parse_training_log(
    log_path: Path,
) -> Tuple[List[int], List[float], List[float]]:
    pattern = re.compile(
        r"epoch\s+(\d+).*?train_loss=([0-9.eE+-]+).*?"
        r"val_loss=([0-9.eE+-]+)"
    )
    epochs: List[int] = []
    train_losses: List[float] = []
    validation_losses: List[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            validation_losses.append(float(match.group(3)))
    return epochs, train_losses, validation_losses


def plot_training_log(log_path: Path, output_path: Path) -> None:
    epochs, train_losses, validation_losses = parse_training_log(log_path)
    if not epochs:
        print(f"No epoch metrics were found in {log_path}.")
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epochs, train_losses, label="Train loss")
    axis.plot(epochs, validation_losses, label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training and Validation Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def print_summary(
    coco_metrics: Dict[str, float],
    rows: Sequence[Dict[str, object]],
) -> None:
    print("\nCOCO metrics")
    if coco_metrics:
        for name, value in coco_metrics.items():
            print(f"  {name:>10}: {value:.4f}")
    else:
        print("  unavailable")

    print("\nPer-class metrics")
    print(
        f"{'class':<18} {'precision':>9} {'recall':>9} "
        f"{'f1':>9} {'ap50':>9} {'gt':>7} {'pred':>7}"
    )
    for row in rows:
        print(
            f"{row['class']:<18} "
            f"{row['precision']:>9.4f} "
            f"{row['recall']:>9.4f} "
            f"{row['f1']:>9.4f} "
            f"{row['ap50']:>9.4f} "
            f"{row['support']:>7d} "
            f"{row['predictions']:>7d}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = select_device(DEVICE)
    records, _ = collect_records(device)

    coco_metrics = compute_coco_metrics(records)
    rows, curves = build_per_class_metrics(
        records,
        display_confidence=DISPLAY_CONFIDENCE,
        match_iou=MATCH_IOU,
    )
    confusion = build_confusion_matrix(
        records,
        score_threshold=DISPLAY_CONFIDENCE,
        iou_threshold=MATCH_IOU,
    )

    gt_counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    prediction_counts = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    ground_truth_sizes: List[np.ndarray] = []
    prediction_sizes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []

    for record in records:
        gt_labels = np.asarray(record["gt_labels"], dtype=np.int64)
        gt_boxes = np.asarray(record["gt_boxes"], dtype=np.float32)
        pred_labels = np.asarray(record["pred_labels"], dtype=np.int64)
        pred_boxes = np.asarray(record["pred_boxes"], dtype=np.float32)
        pred_scores = np.asarray(record["pred_scores"], dtype=np.float32)
        display_mask = pred_scores >= DISPLAY_CONFIDENCE

        for class_id in range(len(CLASS_NAMES)):
            gt_counts[class_id] += int((gt_labels == class_id).sum())
            prediction_counts[class_id] += int(
                ((pred_labels == class_id) & display_mask).sum()
            )
        ground_truth_sizes.append(gt_boxes)
        prediction_sizes.append(pred_boxes[display_mask])
        all_scores.append(pred_scores)

    concatenated_ground_truth = (
        np.concatenate(ground_truth_sizes, axis=0)
        if ground_truth_sizes
        else np.zeros((0, 4), dtype=np.float32)
    )
    concatenated_predictions = (
        np.concatenate(prediction_sizes, axis=0)
        if prediction_sizes
        else np.zeros((0, 4), dtype=np.float32)
    )
    concatenated_scores = (
        np.concatenate(all_scores, axis=0)
        if all_scores
        else np.zeros((0,), dtype=np.float32)
    )

    visualization_dir = OUTPUT_DIR / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(OUTPUT_DIR, coco_metrics, rows, confusion)
    plot_per_class_metrics(
        rows,
        visualization_dir / "per_class_metrics.png",
    )
    plot_pr_curves(
        curves,
        visualization_dir / "precision_recall_curves.png",
    )
    plot_confusion_matrix(
        confusion,
        visualization_dir / "confusion_matrix.png",
    )
    plot_class_counts(
        gt_counts,
        prediction_counts,
        visualization_dir / "class_counts.png",
    )
    plot_box_distribution(
        concatenated_ground_truth,
        concatenated_predictions,
        visualization_dir / "box_size_distribution.png",
    )
    plot_confidence_distribution(
        concatenated_scores,
        visualization_dir / "confidence_distribution.png",
    )
    plot_dashboard(
        rows,
        gt_counts,
        prediction_counts,
        confusion,
        visualization_dir / "dashboard.png",
    )

    selected_records = sorted(
        records,
        key=lambda record: len(record["gt_boxes"]),
        reverse=True,
    )[:VISUAL_SAMPLES]
    overlay_paths: List[Path] = []
    for index, record in enumerate(selected_records):
        overlay_path = (
            visualization_dir
            / "detections"
            / f"{index:03d}_{Path(str(record['path'])).name}"
        )
        draw_detection_overlay(
            record,
            overlay_path,
            score_threshold=DISPLAY_CONFIDENCE,
        )
        overlay_paths.append(overlay_path)
    create_detection_grid(
        overlay_paths,
        visualization_dir / "detection_grid.png",
    )

    if TRAIN_LOG is not None:
        plot_training_log(
            TRAIN_LOG,
            visualization_dir / "training_curves.png",
        )

    print_summary(coco_metrics, rows)
    print(f"\nReport saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
