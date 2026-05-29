from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


BBox = list[float]


def box_area(box: BBox) -> float:
    """Return area for a bbox in [xmin, ymin, xmax, ymax] format."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def compute_iou(box_a: BBox, box_b: BBox) -> float:
    """Compute IoU between two boxes in [xmin, ymin, xmax, ymax] format."""
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter = box_area([inter_x1, inter_y1, inter_x2, inter_y2])
    union = box_area([ax1, ay1, ax2, ay2]) + box_area([bx1, by1, bx2, by2]) - inter
    return inter / union if union > 0 else 0.0


def compute_precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    """Compute scalar precision and recall from detection counts."""
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return precision, recall


def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    """Compute interpolated average precision from precision-recall points."""
    if not recalls:
        return 0.0

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def annotation_to_ground_truth(annotation: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Convert public annotation JSON to image_id -> boxes format."""
    gt: dict[str, list[dict[str, Any]]] = {image["id"]: [] for image in annotation["images"]}
    for ann in annotation["annotations"]:
        gt.setdefault(ann["image_id"], []).append(
            {"class": ann["class"], "bbox": [float(v) for v in ann["bbox"]]}
        )
    return gt


def prediction_list_to_dict(predictions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Convert prediction.json list format to image_id -> boxes format."""
    return {
        item["image_id"]: [
            {
                "class": box["class"],
                "confidence": float(box.get("confidence", 1.0)),
                "bbox": [float(v) for v in box["bbox"]],
            }
            for box in item.get("boxes", [])
        ]
        for item in predictions
    }


def _group_gt_by_class(
    ground_truth: dict[str, list[dict[str, Any]]], classes: list[str]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        class_name: defaultdict(list) for class_name in classes
    }
    for image_id, boxes in ground_truth.items():
        for box in boxes:
            grouped[box["class"]][image_id].append(
                {"bbox": [float(v) for v in box["bbox"]], "matched": False}
            )
    return grouped


def _group_predictions_by_class(
    predictions: dict[str, list[dict[str, Any]]], classes: list[str]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {class_name: [] for class_name in classes}
    for image_id, boxes in predictions.items():
        for box in boxes:
            class_name = box["class"]
            if class_name not in grouped:
                continue
            grouped[class_name].append(
                {
                    "image_id": image_id,
                    "class": class_name,
                    "confidence": float(box.get("confidence", 1.0)),
                    "bbox": [float(v) for v in box["bbox"]],
                }
            )
    return grouped


def evaluate_map(
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    classes: list[str],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute per-class AP, precision, recall, and mAP at IoU threshold."""
    gt_by_class = _group_gt_by_class(copy.deepcopy(ground_truth), classes)
    pred_by_class = _group_predictions_by_class(predictions, classes)

    aps: list[float] = []
    per_class: dict[str, dict[str, Any]] = {}
    total_tp = 0
    total_fp = 0
    total_gt = 0

    for class_name in classes:
        class_gt = gt_by_class[class_name]
        class_preds = sorted(
            pred_by_class[class_name], key=lambda item: item["confidence"], reverse=True
        )
        num_gt = sum(len(items) for items in class_gt.values())

        tp_flags: list[int] = []
        fp_flags: list[int] = []

        for pred in class_preds:
            candidates = class_gt.get(pred["image_id"], [])
            best_iou = 0.0
            best_index = -1

            for index, gt in enumerate(candidates):
                if gt["matched"]:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_index >= 0 and best_iou >= iou_threshold:
                candidates[best_index]["matched"] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        cumulative_tp: list[int] = []
        cumulative_fp: list[int] = []
        tp_sum = 0
        fp_sum = 0
        for tp, fp in zip(tp_flags, fp_flags):
            tp_sum += tp
            fp_sum += fp
            cumulative_tp.append(tp_sum)
            cumulative_fp.append(fp_sum)

        recalls = [tp / num_gt if num_gt else 0.0 for tp in cumulative_tp]
        precisions = [
            tp / max(tp + fp, 1) for tp, fp in zip(cumulative_tp, cumulative_fp)
        ]
        ap = compute_ap(recalls, precisions) if num_gt else 0.0
        if num_gt:
            aps.append(ap)

        precision, recall = compute_precision_recall(tp_sum, fp_sum, num_gt - tp_sum)
        total_tp += tp_sum
        total_fp += fp_sum
        total_gt += num_gt

        per_class[class_name] = {
            "ap": ap,
            "precision": precision,
            "recall": recall,
            "true_positives": tp_sum,
            "false_positives": fp_sum,
            "num_ground_truth": num_gt,
            "num_predictions": len(class_preds),
        }

    micro_precision, micro_recall = compute_precision_recall(
        total_tp, total_fp, total_gt - total_tp
    )
    return {
        "mAP@0.5": sum(aps) / len(aps) if aps else 0.0,
        "iou_threshold": iou_threshold,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "num_ground_truth_boxes": total_gt,
        "num_predictions": sum(len(v) for v in predictions.values()),
        "per_class": per_class,
    }


def evaluate_files(
    ground_truth_path: str | Path,
    predictions_path: str | Path,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    annotation = load_json(ground_truth_path)
    predictions_json = load_json(predictions_path)
    ground_truth = annotation_to_ground_truth(annotation)
    predictions = prediction_list_to_dict(predictions_json)
    return evaluate_map(ground_truth, predictions, annotation["classes"], iou_threshold)


def _self_test() -> None:
    classes = ["person"]
    gt = {"image_1.jpg": [{"class": "person", "bbox": [0, 0, 100, 100]}]}
    preds = {
        "image_1.jpg": [
            {"class": "person", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
            {"class": "person", "confidence": 0.2, "bbox": [200, 200, 300, 300]},
        ]
    }
    result = evaluate_map(gt, preds, classes)
    assert compute_iou([0, 0, 10, 10], [5, 5, 15, 15]) == 25 / 175
    assert result["per_class"]["person"]["true_positives"] == 1
    assert result["per_class"]["person"]["false_positives"] == 1
    assert result["mAP@0.5"] == 1.0
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute object detection metrics.")
    parser.add_argument("--ground_truth", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ground_truth and args.predictions:
        result = evaluate_files(args.ground_truth, args.predictions, args.iou_threshold)
        print(json.dumps(result, indent=2))
    else:
        _self_test()


if __name__ == "__main__":
    main()
