from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as torch_F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F

from utils.helper import build_class_maps, resolve_image_path


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class DetectionCompose:
    def __init__(self, transforms: list[Callable]) -> None:
        self.transforms = transforms

    def __call__(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if random.random() >= self.probability:
            return image, target

        image = F.hflip(image)
        width = image.shape[-1]
        boxes = target["boxes"].clone()
        if boxes.numel():
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        target["boxes"] = boxes
        return image, target


class RandomColorJitter:
    def __init__(
        self,
        probability: float = 0.3,
        brightness: float = 0.15,
        contrast: float = 0.15,
        saturation: float = 0.15,
        hue: float = 0.03,
    ) -> None:
        self.probability = probability
        self.transform = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if random.random() < self.probability:
            image = self.transform(image)
        return image, target


class RandomGrayscale:
    def __init__(self, probability: float = 0.05) -> None:
        self.probability = probability

    def __call__(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if random.random() < self.probability:
            image = F.rgb_to_grayscale(image, num_output_channels=3)
        return image, target


def build_train_transforms(
    horizontal_flip_probability: float = 0.5,
    color_jitter_probability: float = 0.3,
    grayscale_probability: float = 0.05,
) -> DetectionCompose:
    """Build conservative augmentations that preserve detection boxes."""
    return DetectionCompose(
        [
            RandomHorizontalFlip(horizontal_flip_probability),
            RandomColorJitter(color_jitter_probability),
            RandomGrayscale(grayscale_probability),
        ]
    )


def resize_image_and_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor | None,
    min_size: int,
    max_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    _, height, width = image.shape
    short_side = min(height, width)
    long_side = max(height, width)
    scale = min_size / short_side
    if long_side * scale > max_size:
        scale = max_size / long_side
    new_height = int(round(height * scale))
    new_width = int(round(width * scale))
    image = torch_F.interpolate(
        image.unsqueeze(0),
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    if boxes is not None:
        boxes = boxes * scale
    return image, boxes, scale


def pad_images(
    images: list[torch.Tensor],
    size_divisible: int = 16,
    fixed_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    image_sizes = [(image.shape[-2], image.shape[-1]) for image in images]
    if fixed_size is None:
        max_height = max(size[0] for size in image_sizes)
        max_width = max(size[1] for size in image_sizes)
        max_height = int(math.ceil(max_height / size_divisible) * size_divisible)
        max_width = int(math.ceil(max_width / size_divisible) * size_divisible)
    else:
        max_height, max_width = fixed_size
        too_tall = any(height > max_height for height, _width in image_sizes)
        too_wide = any(width > max_width for _height, width in image_sizes)
        if too_tall or too_wide:
            raise ValueError(f"fixed_size={fixed_size} is smaller than at least one resized image: {image_sizes}")
    batch = images[0].new_zeros((len(images), 3, max_height, max_width))
    for index, image in enumerate(images):
        _, height, width = image.shape
        batch[index, :, :height, :width] = image
    return batch, image_sizes


class DetectionModelTransform:
    def __init__(
        self,
        min_size: int,
        max_size: int,
        size_divisible: int = 16,
        fixed_batch_shape: bool = False,
    ) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self.size_divisible = size_divisible
        self.fixed_batch_shape = fixed_batch_shape

    def __call__(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[
        torch.Tensor,
        list[tuple[int, int]],
        list[tuple[int, int]],
        list[float],
        list[dict[str, torch.Tensor]] | None,
    ]:
        normalized_images = []
        original_sizes = []
        resized_sizes = []
        scales = []
        new_targets = [] if targets is not None else None
        mean = IMAGENET_MEAN.to(images[0].device)
        std = IMAGENET_STD.to(images[0].device)
        for index, image in enumerate(images):
            original_sizes.append((image.shape[-2], image.shape[-1]))
            boxes = targets[index]["boxes"] if targets is not None else None
            resized, resized_boxes, scale = resize_image_and_boxes(
                image,
                boxes,
                self.min_size,
                self.max_size,
            )
            normalized_images.append((resized - mean) / std)
            resized_sizes.append((resized.shape[-2], resized.shape[-1]))
            scales.append(scale)
            if targets is not None and new_targets is not None:
                target = {key: value for key, value in targets[index].items()}
                target["boxes"] = resized_boxes
                new_targets.append(target)
        fixed_size = (self.max_size, self.max_size) if self.fixed_batch_shape else None
        batch, _ = pad_images(
            normalized_images,
            size_divisible=self.size_divisible,
            fixed_size=fixed_size,
        )
        return batch, original_sizes, resized_sizes, scales, new_targets


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
    dataset = OdDataset(args.annotation, args.image_dir, transforms=build_train_transforms())
    image, target = dataset[0]
    print(f"Dataset size: {len(dataset)}")
    print(f"Image tensor: {tuple(image.shape)} {image.dtype}")
    print(f"Target boxes: {target['boxes'].shape}, labels: {target['labels'].tolist()}")


if __name__ == "__main__":
    main()
