from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from notebooks.notebook_helpers import draw_boxes_on_axis
from utils.helper import print_run_configuration, save_json
from utils.metric import (
    analyze_detection_errors,
    annotation_to_ground_truth,
    compute_confusion_matrix,
    evaluate_extended_metrics,
    load_json,
    prediction_list_to_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze detection metrics and visualize errors.")
    parser.add_argument("--ground_truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--output_dir", default="saved_results/baseline/analysis", type=Path)
    parser.add_argument("--max_visualizations", type=int, default=50)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    return parser.parse_args()


def select_gallery_images(
    per_image: dict[str, dict[str, Any]], max_visualizations: int
) -> list[tuple[str, dict[str, Any]]]:
    grouped: defaultdict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for image_id, details in per_image.items():
        grouped[details["category"]].append((image_id, details))

    categories = ["good", "incorrect", "missed", "mixed"]
    selected: list[tuple[str, dict[str, Any]]] = []
    quota = max(1, max_visualizations // len(categories))
    for category in categories:
        selected.extend(grouped[category][:quota])

    if len(selected) < max_visualizations:
        already_selected = {image_id for image_id, _ in selected}
        remaining = [
            item
            for category in categories
            for item in grouped[category]
            if item[0] not in already_selected
        ]
        selected.extend(remaining[: max_visualizations - len(selected)])
    return selected[:max_visualizations]


def save_gallery(
    image_dir: Path,
    output_dir: Path,
    ground_truth: dict[str, list[dict[str, Any]]],
    per_image: dict[str, dict[str, Any]],
    classes: list[str],
    max_visualizations: int,
) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    gallery_dir = output_dir / "gallery"
    selected = select_gallery_images(per_image, max_visualizations)
    index: list[dict[str, Any]] = []

    for image_id, details in selected:
        category = details["category"]
        category_dir = gallery_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / image_id
        output_path = category_dir / f"{Path(image_id).stem}.png"

        fig, ax = plt.subplots(figsize=(12, 8))
        draw_boxes_on_axis(ax, image_path, [], classes=classes)
        for prediction in details["predictions"]:
            error_type = prediction["error_type"]
            is_true_positive = error_type == "true_positive"
            draw_boxes_on_axis(
                ax,
                image_path,
                [prediction],
                classes=classes,
                label_prefix="Pred TP: " if is_true_positive else f"Pred {error_type}: ",
                edge_color="#2ca02c" if is_true_positive else "#d62728",
                label_position="top_left",
            )
        draw_boxes_on_axis(
            ax,
            image_path,
            ground_truth.get(image_id, []),
            classes=classes,
            label_prefix="GT: ",
            edge_color="black",
            line_style="--",
            label_position="bottom_right",
        )
        ax.set_title(
            f"{category.upper()} | TP={details['true_positives']} "
            f"FP={details['false_positives']} FN={details['false_negatives']}"
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        index.append(
            {
                "image_id": image_id,
                "category": category,
                "output": str(output_path),
                "true_positives": details["true_positives"],
                "false_positives": details["false_positives"],
                "false_negatives": details["false_negatives"],
            }
        )
    return index


def save_confusion_outputs(confusion: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    matrix_df = pd.DataFrame.from_dict(confusion["matrix"], orient="index").fillna(0).astype(int)
    pairs_df = pd.DataFrame(confusion["pairs"])
    matrix_df.to_csv(output_dir / "confusion_matrix.csv", index_label="gt_class")
    pairs_df.to_csv(output_dir / "confusion_pairs.csv", index=False)

    plot_rows = [class_name for class_name in confusion["classes"] if class_name in matrix_df.index]
    plot_cols = list(confusion["classes"]) + ["missed"]
    plot_df = matrix_df.loc[plot_rows, plot_cols]
    fig_width = max(8, 1.2 * len(plot_cols))
    fig_height = max(5, 0.8 * len(plot_rows))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(plot_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title(f"Confusion matrix @ IoU {confusion['iou_threshold']}")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    print_run_configuration(
        "Prediction Analysis",
        {
            "ground_truth": args.ground_truth,
            "predictions": args.predictions,
            "image_dir": args.image_dir,
            "output_dir": args.output_dir,
            "max_visualizations": args.max_visualizations,
            "iou_threshold": args.iou_threshold,
        },
    )
    annotation = load_json(args.ground_truth)
    ground_truth = annotation_to_ground_truth(annotation)
    predictions = prediction_list_to_dict(load_json(args.predictions))
    metrics = evaluate_extended_metrics(ground_truth, predictions, annotation["classes"])
    errors = analyze_detection_errors(ground_truth, predictions, args.iou_threshold)
    confusion = compute_confusion_matrix(
        ground_truth,
        predictions,
        annotation["classes"],
        args.iou_threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, args.output_dir / "metrics.json")
    save_json(errors, args.output_dir / "errors.json")
    save_json(confusion, args.output_dir / "confusion_matrix.json")
    save_confusion_outputs(confusion, args.output_dir)
    gallery = save_gallery(
        args.image_dir,
        args.output_dir,
        ground_truth,
        errors["per_image"],
        annotation["classes"],
        args.max_visualizations,
    )
    save_json(gallery, args.output_dir / "gallery" / "index.json")

    print(f"mAP@0.5: {metrics['mAP@0.5']:.4f}")
    print(f"mAP@0.75: {metrics['mAP@0.75']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    print(f"Error counts: {errors['error_counts']}")
    print(f"Saved {len(gallery)} visualizations to {args.output_dir / 'gallery'}")


if __name__ == "__main__":
    main()
