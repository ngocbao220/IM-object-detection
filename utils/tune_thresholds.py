from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils.helper import save_json
from utils.metric import (
    annotation_to_ground_truth,
    apply_confidence_and_nms,
    evaluate_map,
    load_json,
    prediction_list_to_dict,
)


def parse_float_list(value: str) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(item < 0 or item > 1 for item in values):
        raise ValueError("Threshold values must be comma-separated numbers between 0 and 1.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune confidence and NMS thresholds offline.")
    parser.add_argument("--ground_truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", default="saved_results/threshold_tuning.json", type=Path)
    parser.add_argument("--confidence_thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--nms_thresholds", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    return parser.parse_args()


def tune_thresholds(
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    classes: list[str],
    confidence_thresholds: list[float],
    nms_thresholds: list[float],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for confidence_threshold in confidence_thresholds:
        for nms_threshold in nms_thresholds:
            filtered = apply_confidence_and_nms(
                predictions, confidence_threshold, nms_threshold
            )
            metrics = evaluate_map(ground_truth, filtered, classes, iou_threshold)
            precision = metrics["micro_precision"]
            recall = metrics["micro_recall"]
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append(
                {
                    "confidence_threshold": confidence_threshold,
                    "nms_threshold": nms_threshold,
                    "mAP": metrics["mAP"],
                    "micro_precision": precision,
                    "micro_recall": recall,
                    "f1": f1,
                    "num_predictions": metrics["num_predictions"],
                }
            )
    return sorted(rows, key=lambda row: (row["mAP"], row["f1"]), reverse=True)


def main() -> None:
    args = parse_args()
    annotation = load_json(args.ground_truth)
    predictions = prediction_list_to_dict(load_json(args.predictions))
    results = tune_thresholds(
        annotation_to_ground_truth(annotation),
        predictions,
        annotation["classes"],
        parse_float_list(args.confidence_thresholds),
        parse_float_list(args.nms_thresholds),
        args.iou_threshold,
    )
    save_json({"best": results[0], "results": results}, args.output)
    print(f"conf | nms  | mAP@{args.iou_threshold:g} | precision | recall | F1")
    for row in results[:10]:
        print(
            f"{row['confidence_threshold']:.2f} | {row['nms_threshold']:.2f} | "
            f"{row['mAP']:.4f}  | {row['micro_precision']:.4f}    | "
            f"{row['micro_recall']:.4f} | {row['f1']:.4f}"
        )
    print(f"Saved threshold sweep to {args.output}")


if __name__ == "__main__":
    main()
