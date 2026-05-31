from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import functional as F
from tqdm.auto import tqdm

from models.faster_rcnn import create_faster_rcnn_resnet50
from models.modules import get_device, load_checkpoint
from utils.helper import load_classes


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Faster R-CNN inference.")
    parser.add_argument("--image_dir", required=True, help="Image file or directory.")
    parser.add_argument("--output", required=True, help="Output predictions.json path.")
    parser.add_argument("--checkpoint", default="saved_results/checkpoints/best_model.pth")
    parser.add_argument("--classes", default="public/classes.json")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def list_images(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def clamp_box(box: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


@torch.no_grad()
def predict_images(
    model: torch.nn.Module,
    image_paths: list[Path],
    idx_to_class: dict[int, str],
    device: torch.device,
    score_threshold: float,
) -> list[dict[str, Any]]:
    model.eval()
    results: list[dict[str, Any]] = []

    for path in tqdm(image_paths, desc="predict"):
        image = Image.open(path).convert("RGB")
        width, height = image.size
        tensor = F.to_tensor(image).to(device)
        output = model([tensor])[0]

        boxes = []
        for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
            confidence = float(score.detach().cpu())
            if confidence < score_threshold:
                continue
            label_id = int(label.detach().cpu())
            boxes.append(
                {
                    "class": idx_to_class.get(label_id, str(label_id)),
                    "confidence": round(confidence, 6),
                    "bbox": clamp_box([float(v) for v in box.detach().cpu().tolist()], width, height),
                }
            )

        results.append({"image_id": path.name, "boxes": boxes})
    return results


def main() -> None:
    args = parse_args()
    image_paths = list_images(args.image_dir)
    classes = load_classes(args.classes)
    idx_to_class = {idx + 1: name for idx, name in enumerate(classes)}
    device = get_device(args.device)

    model = create_faster_rcnn_resnet50(
        num_classes=len(classes) + 1,
        box_score_thresh=args.score_threshold,
        box_nms_thresh=args.nms_threshold,
    ).to(device)
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, model, device)
        if checkpoint.get("classes"):
            idx_to_class = {idx + 1: name for idx, name in enumerate(checkpoint["classes"])}
    else:
        print(f"Warning: checkpoint not found at {checkpoint_path}; using an untrained model.")

    predictions = predict_images(model, image_paths, idx_to_class, device, args.score_threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Saved {len(predictions)} image predictions to {output}")


if __name__ == "__main__":
    main()
