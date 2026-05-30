from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_best_epoch(experiment_dir: Path) -> dict[str, Any] | None:
    metrics_files = sorted((experiment_dir / "metrics").glob("epoch_*.json"))
    if not metrics_files:
        return None
    metrics = [load_json(path) for path in metrics_files]
    return max(metrics, key=lambda item: item["mAP@0.5"])


def build_summary(results_dir: Path) -> list[dict[str, Any]]:
    summary = []
    for experiment_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        best = find_best_epoch(experiment_dir)
        if best is None:
            continue
        summary.append(
            {
                "experiment": experiment_dir.name,
                "best_epoch": best["epoch"],
                "mAP@0.5": best["mAP@0.5"],
                "precision": best["micro_precision"],
                "recall": best["micro_recall"],
                "val_loss": best["val"].get("loss", 0.0),
                "num_predictions": best["num_predictions"],
            }
        )
    return summary


def format_markdown(summary: list[dict[str, Any]]) -> str:
    lines = [
        "| Experiment | Best epoch | mAP@0.5 | Precision | Recall | Val loss | Predictions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['experiment']} | {item['best_epoch']} | {item['mAP@0.5']:.4f} | "
            f"{item['precision']:.4f} | {item['recall']:.4f} | {item['val_loss']:.4f} | "
            f"{item['num_predictions']} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize augmentation ablation results.")
    parser.add_argument("--results_dir", type=Path, default=Path("saved_results/augmentation_ablation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Ablation results directory does not exist: {args.results_dir}")

    summary = build_summary(args.results_dir)
    if not summary:
        raise FileNotFoundError(f"No epoch metrics found under {args.results_dir}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_dir / "summary.json"
    markdown_path = args.results_dir / "summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = format_markdown(summary)
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"\nSaved summary to {markdown_path}")


if __name__ == "__main__":
    main()
