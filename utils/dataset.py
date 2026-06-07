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


def clip_boxes_to_image(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)
    return boxes


def filter_boxes_by_size(boxes: torch.Tensor, min_size: float = 2.0) -> torch.Tensor:
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    return (widths >= min_size) & (heights >= min_size)


class RandomScaleJitter:
    def __init__(
        self,
        probability: float = 0.5,
        min_scale: float = 0.85,
        max_scale: float = 1.15,
    ) -> None:
        self.probability = probability
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if random.random() >= self.probability:
            return image, target
        scale = random.uniform(self.min_scale, self.max_scale)
        _, height, width = image.shape
        new_height = max(8, int(round(height * scale)))
        new_width = max(8, int(round(width * scale)))
        image = torch_F.interpolate(
            image.unsqueeze(0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        boxes = target["boxes"].clone()
        if boxes.numel():
            boxes *= scale
            target["boxes"] = boxes
            target["area"] = target["area"] * (scale * scale)
        return image, target


class RandomSafeCrop:
    def __init__(
        self,
        probability: float = 0.2,
        min_crop_scale: float = 0.7,
        min_box_visibility: float = 0.5,
        attempts: int = 10,
    ) -> None:
        self.probability = probability
        self.min_crop_scale = min_crop_scale
        self.min_box_visibility = min_box_visibility
        self.attempts = attempts

    def __call__(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        boxes = target["boxes"]
        if random.random() >= self.probability or boxes.numel() == 0:
            return image, target

        _, height, width = image.shape
        original_boxes = boxes.clone()
        original_areas = ((original_boxes[:, 2] - original_boxes[:, 0]) * (original_boxes[:, 3] - original_boxes[:, 1])).clamp(min=1e-6)

        for _ in range(self.attempts):
            crop_h = random.randint(max(8, int(height * self.min_crop_scale)), height)
            crop_w = random.randint(max(8, int(width * self.min_crop_scale)), width)
            max_top = height - crop_h
            max_left = width - crop_w
            top = 0 if max_top <= 0 else random.randint(0, max_top)
            left = 0 if max_left <= 0 else random.randint(0, max_left)

            cropped_boxes = original_boxes.clone()
            cropped_boxes[:, 0::2] -= left
            cropped_boxes[:, 1::2] -= top
            cropped_boxes = clip_boxes_to_image(cropped_boxes, crop_h, crop_w)
            keep = filter_boxes_by_size(cropped_boxes, min_size=2.0)
            if not keep.any():
                continue

            cropped_boxes = cropped_boxes[keep]
            kept_labels = target["labels"][keep]
            kept_iscrowd = target["iscrowd"][keep]
            visible_areas = ((cropped_boxes[:, 2] - cropped_boxes[:, 0]) * (cropped_boxes[:, 3] - cropped_boxes[:, 1])).clamp(min=0.0)
            visibility = visible_areas / original_areas[keep]
            visible_keep = visibility >= self.min_box_visibility
            if not visible_keep.any():
                continue

            cropped_boxes = cropped_boxes[visible_keep]
            kept_labels = kept_labels[visible_keep]
            kept_iscrowd = kept_iscrowd[visible_keep]
            visible_areas = visible_areas[visible_keep]

            image = F.crop(image, top=top, left=left, height=crop_h, width=crop_w)
            target["boxes"] = cropped_boxes
            target["labels"] = kept_labels
            target["iscrowd"] = kept_iscrowd
            target["area"] = visible_areas
            return image, target

        return image, target


class RandomGaussianBlur:
    def __init__(
        self,
        probability: float = 0.1,
        kernel_size: int = 5,
        sigma: tuple[float, float] = (0.1, 1.5),
    ) -> None:
        self.probability = probability
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma

    def __call__(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if random.random() < self.probability:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            image = F.gaussian_blur(image, kernel_size=[self.kernel_size, self.kernel_size], sigma=[sigma, sigma])
        return image, target


class RandomGaussianNoise:
    def __init__(self, probability: float = 0.1, std: float = 0.02) -> None:
        self.probability = probability
        self.std = std

    def __call__(
        self, image: torch.Tensor, target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if random.random() < self.probability:
            image = (image + torch.randn_like(image) * self.std).clamp(0.0, 1.0)
        return image, target


def build_train_transforms(
    horizontal_flip_probability: float = 0.5,
    color_jitter_probability: float = 0.3,
    grayscale_probability: float = 0.05,
    scale_jitter_probability: float = 0.4,
    scale_jitter_min: float = 0.85,
    scale_jitter_max: float = 1.15,
    safe_crop_probability: float = 0.2,
    safe_crop_min_scale: float = 0.7,
    safe_crop_min_visibility: float = 0.5,
    blur_probability: float = 0.1,
    blur_kernel_size: int = 5,
    noise_probability: float = 0.1,
    noise_std: float = 0.02,
) -> DetectionCompose:
    """Build conservative augmentations that preserve detection boxes."""
    return DetectionCompose(
        [
            RandomHorizontalFlip(horizontal_flip_probability),
            RandomScaleJitter(scale_jitter_probability, scale_jitter_min, scale_jitter_max),
            RandomSafeCrop(safe_crop_probability, safe_crop_min_scale, safe_crop_min_visibility),
            RandomColorJitter(color_jitter_probability),
            RandomGrayscale(grayscale_probability),
            RandomGaussianBlur(blur_probability, blur_kernel_size),
            RandomGaussianNoise(noise_probability, noise_std),
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
