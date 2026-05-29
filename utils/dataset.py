from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from utils.helper import build_class_maps, resolve_image_path


class OdDataset(Dataset):
    """Torch Dataset for public object-detection annotations."""

    def __init__(
        self,
        annotation_path: str | Path,
        image_dir: str | Path,
        transforms: Callable[[Any, dict[str, torch.Tensor]], tuple[Any, dict[str, torch.Tensor]]]
        | None = None,
        classes: list[str] | None = None,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.image_dir = Path(image_dir)
        self.transforms = transforms

        with self.annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.classes = classes or list(data["classes"])
        self.class_to_idx, self.idx_to_class = build_class_maps(self.classes)
        self.images = list(data["images"])

        self.annotations_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ann in data.get("annotations", []):
            self.annotations_by_image[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_info = self.images[index]
        image_id = image_info["id"]
        image_path = resolve_image_path(self.image_dir, image_id, image_info.get("file_name"))
        image = Image.open(image_path).convert("RGB")

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        for ann in self.annotations_by_image.get(image_id, []):
            x1, y1, x2, y2 = [float(v) for v in ann["bbox"]]
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_idx[ann["class"]])
            areas.append((x2 - x1) * (y2 - y1))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        image_tensor = F.to_tensor(image)
        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)
        return image_tensor, target


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list, list]:
    return tuple(zip(*batch))  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test OdDataset.")
    parser.add_argument("--annotation", default="public/annotations/train.json")
    parser.add_argument("--image_dir", default="public/train/images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = OdDataset(args.annotation, args.image_dir)
    image, target = dataset[0]
    print(f"Dataset size: {len(dataset)}")
    print(f"Image tensor: {tuple(image.shape)} {image.dtype}")
    print(f"Target boxes: {target['boxes'].shape}, labels: {target['labels'].tolist()}")


if __name__ == "__main__":
    main()
