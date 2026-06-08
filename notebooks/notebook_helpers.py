from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from utils.helper import load_json, load_classes, resolve_image_path


def summarize_effective_class_distribution(
    annotation_path: str | Path,
    image_dir: str | Path,
    sampler_strategy: str = "none",
    oversample_class: str | None = None,
    oversample_factor: float = 1.0,
    small_object_boost: float = 1.5,
    small_object_threshold: float = 0.01,
    empty_image_weight: float = 0.5,
) -> Any:
    """Estimate per-class object counts after one epoch of weighted sampling.

    Augmentation does not change class labels, so the effective class distribution
    changes only through the image sampler. This function computes:
    - original object count per class
    - expected sampled object count per class in one epoch
    - ratio between sampled/original
    """
    import pandas as pd

    from train import build_training_sampler
    from utils.dataset import OdDataset

    dataset = OdDataset(annotation_path, image_dir)
    sampler, sampler_info = build_training_sampler(
        dataset=dataset,
        strategy=sampler_strategy,
        class_name=oversample_class,
        factor=oversample_factor,
        small_object_boost=small_object_boost,
        small_object_threshold=small_object_threshold,
        empty_image_weight=empty_image_weight,
    )

    original_counts = Counter()
    expected_counts = Counter()
    expected_image_hits = Counter()

    if sampler is None:
        for image in dataset.images:
            image_id = image["id"]
            annotations = dataset.annotations_by_image.get(image_id, [])
            classes_in_image = set()
            for ann in annotations:
                class_name = ann["class"]
                original_counts[class_name] += 1
                expected_counts[class_name] += 1
                classes_in_image.add(class_name)
            for class_name in classes_in_image:
                expected_image_hits[class_name] += 1
    else:
        weights = sampler.weights.detach().cpu().tolist()
        total_weight = max(sum(weights), 1e-12)
        num_draws = sampler.num_samples
        for index, image in enumerate(dataset.images):
            image_id = image["id"]
            annotations = dataset.annotations_by_image.get(image_id, [])
            draw_expectation = num_draws * (weights[index] / total_weight)
            classes_in_image = set()
            for ann in annotations:
                class_name = ann["class"]
                original_counts[class_name] += 1
                expected_counts[class_name] += draw_expectation
                classes_in_image.add(class_name)
            for class_name in classes_in_image:
                expected_image_hits[class_name] += draw_expectation

    rows = []
    for class_name in dataset.classes:
        original = float(original_counts.get(class_name, 0))
        expected = float(expected_counts.get(class_name, 0.0))
        rows.append(
            {
                "class": class_name,
                "original_objects": int(original),
                "expected_sampled_objects": expected,
                "sampling_ratio": expected / max(original, 1.0),
                "expected_sampled_images": float(expected_image_hits.get(class_name, 0.0)),
            }
        )

    table = pd.DataFrame(rows).sort_values("expected_sampled_objects", ascending=False).reset_index(drop=True)
    return table, sampler_info

def index_annotations(annotation: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotation.get("annotations", []):
        indexed[ann["image_id"]].append(ann)
    return dict(indexed)


def dataset_summary(annotation_path: str | Path) -> dict[str, Any]:
    data = load_json(annotation_path)
    class_counts = Counter(ann["class"] for ann in data.get("annotations", []))
    boxes_per_image = Counter(ann["image_id"] for ann in data.get("annotations", []))
    return {
        "num_images": len(data.get("images", [])),
        "num_annotations": len(data.get("annotations", [])),
        "classes": data.get("classes", []),
        "class_counts": dict(class_counts),
        "images_without_boxes": len(data.get("images", [])) - len(boxes_per_image),
        "max_boxes_per_image": max(boxes_per_image.values(), default=0),
    }


def _read_image_size(
    image_dir: str | Path,
    image_id: str,
    image_info: dict[str, Any],
) -> tuple[int, int]:
    width = int(image_info.get("width", 0) or 0)
    height = int(image_info.get("height", 0) or 0)
    if width > 0 and height > 0:
        return width, height
    image_path = resolve_image_path(image_dir, image_id, image_info.get("file_name"))
    with Image.open(image_path) as image:
        return image.size


def object_size_group(
    relative_area: float,
    small_threshold: float = 0.01,
    medium_threshold: float = 0.10,
) -> str:
    """Group object size by bbox/image area ratio."""
    if relative_area < small_threshold:
        return "small"
    if relative_area < medium_threshold:
        return "medium"
    return "large"


def build_split_box_table(
    annotation_path: str | Path,
    image_dir: str | Path,
    split: str,
    small_threshold: float = 0.01,
    medium_threshold: float = 0.10,
) -> Any:
    """Return one row per object with absolute and relative bbox size fields."""
    import pandas as pd

    annotation = load_json(annotation_path)
    images = {image["id"]: image for image in annotation.get("images", [])}
    rows = []
    for ann in annotation.get("annotations", []):
        image_id = ann["image_id"]
        image_info = images.get(image_id, {})
        image_width, image_height = _read_image_size(image_dir, image_id, image_info)
        box_width, box_height = bbox_wh(ann["bbox"])
        area = box_width * box_height
        image_area = max(float(image_width * image_height), 1.0)
        relative_area = area / image_area
        rows.append(
            {
                "split": split,
                "image_id": image_id,
                "class": ann.get("class", "unknown"),
                "bbox": ann["bbox"],
                "image_width": image_width,
                "image_height": image_height,
                "image_area": image_area,
                "width": box_width,
                "height": box_height,
                "area": area,
                "relative_area": relative_area,
                "aspect_ratio": box_width / box_height if box_height > 0 else float("nan"),
                "size_group": object_size_group(relative_area, small_threshold, medium_threshold),
            }
        )
    return pd.DataFrame(rows)


def analyze_dataset_splits(
    data_root: str | Path = "public",
    splits: tuple[str, ...] = ("train", "val", "test"),
    small_threshold: float = 0.01,
    medium_threshold: float = 0.10,
) -> dict[str, Any]:
    """Analyze dataset-level split, class, box-size, and imbalance statistics."""
    import pandas as pd

    root = resolve_public_data_root(data_root)
    split_rows = []
    box_tables = []
    classes: set[str] = set()
    loaded_splits = []

    for split in splits:
        annotation_path = root / "annotations" / f"{split}.json"
        image_dir = root / split / "images"
        if not annotation_path.exists():
            continue
        annotation = load_json(annotation_path)
        loaded_splits.append(split)
        split_classes = list(annotation.get("classes", []))
        classes.update(split_classes)
        num_images = len(annotation.get("images", []))
        num_objects = len(annotation.get("annotations", []))
        split_rows.append(
            {
                "split": split,
                "annotation_path": str(annotation_path),
                "image_dir": str(image_dir),
                "num_images": num_images,
                "num_objects": num_objects,
                "num_classes": len(split_classes),
                "objects_per_image": num_objects / max(num_images, 1),
            }
        )
        if num_objects:
            box_tables.append(
                build_split_box_table(
                    annotation_path,
                    image_dir,
                    split,
                    small_threshold=small_threshold,
                    medium_threshold=medium_threshold,
                )
            )

    split_table = pd.DataFrame(split_rows)
    box_table = pd.concat(box_tables, ignore_index=True) if box_tables else pd.DataFrame()

    if split_table.empty:
        raise FileNotFoundError(f"No annotation files found under {root / 'annotations'}.")

    total_images = int(split_table["num_images"].sum())
    total_objects = int(split_table["num_objects"].sum())
    overall = pd.DataFrame(
        [
            {
                "splits": ", ".join(loaded_splits),
                "total_images": total_images,
                "total_objects": total_objects,
                "num_classes": len(classes),
                "objects_per_image": total_objects / max(total_images, 1),
            }
        ]
    )

    if box_table.empty:
        class_counts = pd.DataFrame(columns=["class", "count"])
        class_by_split = pd.DataFrame()
        size_groups = pd.DataFrame()
        box_summary = pd.DataFrame()
    else:
        class_counts = (
            box_table.groupby("class")
            .size()
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False)
        )
        class_counts["ratio"] = class_counts["count"] / max(class_counts["count"].sum(), 1)
        class_by_split = (
            box_table.pivot_table(
                index="class",
                columns="split",
                values="bbox",
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
        )
        size_groups = (
            box_table.pivot_table(
                index="size_group",
                columns="split",
                values="bbox",
                aggfunc="count",
                fill_value=0,
            )
            .reindex(["small", "medium", "large"])
            .fillna(0)
            .astype(int)
            .reset_index()
        )
        box_summary = box_table[["width", "height", "area", "relative_area", "aspect_ratio"]].describe(
            percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        )

    imbalance = {
        "max_count": int(class_counts["count"].max()) if not class_counts.empty else 0,
        "min_count": int(class_counts["count"].min()) if not class_counts.empty else 0,
        "max_min_ratio": (
            float(class_counts["count"].max() / max(class_counts["count"].min(), 1))
            if not class_counts.empty
            else 0.0
        ),
        "minority_classes": (
            class_counts[class_counts["count"] <= class_counts["count"].median()]["class"].tolist()
            if not class_counts.empty
            else []
        ),
    }

    return {
        "root": root,
        "overall": overall,
        "splits": split_table,
        "class_counts": class_counts,
        "class_by_split": class_by_split,
        "box_table": box_table,
        "box_summary": box_summary,
        "size_groups": size_groups,
        "imbalance": imbalance,
        "size_thresholds": {
            "small": f"relative_area < {small_threshold}",
            "medium": f"{small_threshold} <= relative_area < {medium_threshold}",
            "large": f"relative_area >= {medium_threshold}",
        },
    }


def plot_dataset_analysis(analysis: dict[str, Any]) -> Any:
    """Plot split counts, class imbalance, and object-size groups."""
    import matplotlib.pyplot as plt

    split_table = analysis["splits"]
    class_counts = analysis["class_counts"]
    size_groups = analysis["size_groups"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    split_table.set_index("split")[["num_images", "num_objects"]].plot.bar(
        ax=axes[0],
        rot=0,
        color=["#4C78A8", "#F58518"],
    )
    axes[0].set_title("Images and objects by split")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.3)

    class_counts.plot.bar(x="class", y="count", ax=axes[1], legend=False, color="#54A24B")
    axes[1].set_title("Objects by class")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Object count")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.3)

    if not size_groups.empty:
        size_groups.set_index("size_group").plot.bar(
            ax=axes[2],
            rot=0,
            color=["#4C78A8", "#F58518", "#E45756"][: max(len(size_groups.columns) - 1, 1)],
        )
    axes[2].set_title("Object size groups")
    axes[2].set_xlabel("Size group")
    axes[2].set_ylabel("Object count")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return axes


def resolve_image_path(image_dir: str | Path, image_id: str, file_name: str | None = None) -> Path:
    image_dir = Path(image_dir)
    candidates = [image_dir / image_id]
    if file_name:
        candidates.extend([image_dir / file_name, image_dir.parent / file_name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_public_data_root(data_root: str | Path | None = None) -> Path:
    """Find the public dataset directory from common notebook working directories."""
    candidates = []
    if data_root is not None:
        candidates.append(Path(data_root))
    candidates.extend(
        [
            Path("public"),
            Path("../public"),
            Path.cwd() / "public",
            Path.cwd().parent / "public",
        ]
    )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "annotations").exists() and (candidate / "train").exists():
            return candidate
    raise FileNotFoundError("Could not find public dataset directory. Pass data_root explicitly.")


def normalize_split(folder: str | Path) -> str:
    split = Path(str(folder).strip().rstrip("/")).name.lower()
    if split not in {"train", "val"}:
        raise ValueError("folder must be 'train', 'val', or a path ending with train/val.")
    return split


def load_split_annotations(
    folder: str | Path = "train",
    data_root: str | Path | None = None,
) -> tuple[Path, str, dict[str, Any], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Load one public split in a notebook-friendly indexed format."""
    root = resolve_public_data_root(data_root)
    split = normalize_split(folder)
    data = load_json(root / "annotations" / f"{split}.json")
    images = {image["id"]: image for image in data["images"]}
    return root, split, data, images, index_annotations(data)


def bbox_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in bbox)  # type: ignore[return-value]


def bbox_wh(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy(bbox)
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def bbox_area(bbox: list[float]) -> float:
    width, height = bbox_wh(bbox)
    return width * height


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = bbox_xyxy(box_a)
    bx1, by1, bx2, by2 = bbox_xyxy(box_b)
    intersection = bbox_area(
        [max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)]
    )
    union = bbox_area(box_a) + bbox_area(box_b) - intersection
    return intersection / union if union > 0 else 0.0


def bbox_center_x(annotation: dict[str, Any]) -> float:
    x1, _, x2, _ = bbox_xyxy(annotation["bbox"])
    return (x1 + x2) / 2


def bbox_center_y(annotation: dict[str, Any]) -> float:
    _, y1, _, y2 = bbox_xyxy(annotation["bbox"])
    return (y1 + y2) / 2


def draw_boxes_on_axis(
    ax: Any,
    image_path: str | Path,
    boxes: list[dict[str, Any]],
    classes: list[str] | None = None,
    title: str | None = None,
    label_prefix: str = "",
    edge_color: str | None = None,
    line_style: str = "-",
    label_position: str = "top_left",
) -> Any:
    """Draw ground-truth or prediction boxes on an existing matplotlib axis."""
    import matplotlib.pyplot as plt

    image = Image.open(image_path).convert("RGB")
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title)

    cmap = plt.get_cmap("tab10")
    class_to_idx = {name: index for index, name in enumerate(classes or [])}
    for box in boxes:
        x1, y1, x2, y2 = bbox_xyxy(box["bbox"])
        class_name = box.get("class", "object")
        confidence = box.get("confidence", box.get("score"))
        label = f"{class_name} {float(confidence):.2f}" if confidence is not None else class_name
        label = f"{label_prefix}{label}"
        color = edge_color or cmap(class_to_idx.get(class_name, 0) % 10)
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
            edgecolor=color,
            linestyle=line_style,
        )
        ax.add_patch(rect)
        if label_position == "top_left":
            label_x, label_y = x1, max(0, y1 - 4)
            horizontal_alignment = "left"
            vertical_alignment = "bottom"
        elif label_position == "bottom_right":
            label_x, label_y = x2, y2
            horizontal_alignment = "right"
            vertical_alignment = "top"
        else:
            raise ValueError("label_position must be 'top_left' or 'bottom_right'.")
        ax.text(
            label_x,
            label_y,
            label,
            color="white",
            fontsize=9,
            horizontalalignment=horizontal_alignment,
            verticalalignment=vertical_alignment,
            bbox={"facecolor": color, "alpha": 0.85, "edgecolor": "none", "pad": 2},
        )
    return ax


def browse_boxes_with_slider(
    image_ids: list[str],
    boxes_by_image: dict[str, list[dict[str, Any]]],
    image_dir: str | Path,
    classes: list[str] | None = None,
    title_prefix: str = "",
    overlay_boxes_by_image: dict[str, list[dict[str, Any]]] | None = None,
    overlay_label_prefix: str = "GT: ",
) -> Any:
    """Browse detection boxes with the same slider UI for ground truth and predictions."""
    import matplotlib.pyplot as plt

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as error:
        raise RuntimeError("Install ipywidgets to use the notebook slider viewer.") from error

    if not image_ids:
        raise ValueError("No images available for slider viewer.")

    output = widgets.Output()
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(image_ids) - 1,
        step=1,
        description="Index",
        continuous_update=False,
    )

    def render(index: int) -> None:
        image_id = image_ids[index]
        boxes = boxes_by_image.get(image_id, [])
        overlay_boxes = (overlay_boxes_by_image or {}).get(image_id, [])
        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(9, 7))
            draw_boxes_on_axis(
                ax,
                resolve_image_path(image_dir, image_id),
                boxes,
                classes=classes,
                title=f"{title_prefix}{image_id} - {len(boxes)} boxes",
            )
            if overlay_boxes:
                draw_boxes_on_axis(
                    ax,
                    resolve_image_path(image_dir, image_id),
                    overlay_boxes,
                    classes=classes,
                    label_prefix=overlay_label_prefix,
                    edge_color="black",
                    line_style="--",
                    label_position="bottom_right",
                )
            plt.show()
            plt.close(fig)

    slider.observe(lambda change: render(change["new"]), names="value")
    render(0)
    display(widgets.VBox([slider, output]))
    return slider


def show_groundtruth_slider(
    folder: str | Path = "train",
    data_root: str | Path | None = None,
) -> Any:
    root, split, data, images, annotations_by_image = load_split_annotations(folder, data_root)
    return browse_boxes_with_slider(
        list(images),
        annotations_by_image,
        root / split / "images",
        classes=data["classes"],
        title_prefix=f"{split}/ground-truth/",
    )


def show_predictions_slider(
    predictions_path: str | Path,
    image_dir: str | Path = "public/val/images",
    classes_path: str | Path = "public/classes.json",
    show_ground_truth: bool = False,
    ground_truth_path: str | Path = "public/annotations/val.json",
) -> Any:
    predictions = load_json(predictions_path)
    boxes_by_image = {item["image_id"]: item.get("boxes", []) for item in predictions}
    ground_truth_by_image = None
    if show_ground_truth:
        ground_truth_by_image = index_annotations(load_json(ground_truth_path))
    return browse_boxes_with_slider(
        list(boxes_by_image),
        boxes_by_image,
        image_dir,
        classes=load_classes(classes_path),
        title_prefix="prediction/",
        overlay_boxes_by_image=ground_truth_by_image,
    )


def _dataset_target_to_boxes(target: dict[str, Any], idx_to_class: dict[int, str]) -> list[dict[str, Any]]:
    boxes = target["boxes"].detach().cpu().tolist()
    labels = target["labels"].detach().cpu().tolist()
    return [
        {"class": idx_to_class[int(label)], "bbox": [float(value) for value in box]}
        for box, label in zip(boxes, labels)
    ]


def _tensor_to_image_array(image_tensor: Any) -> Any:
    image = image_tensor.detach().cpu().clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def draw_tensor_boxes_on_axis(
    ax: Any,
    image_tensor: Any,
    boxes: list[dict[str, Any]],
    classes: list[str] | None = None,
    title: str | None = None,
    label_prefix: str = "",
    edge_color: str | None = None,
    line_style: str = "-",
    label_position: str = "top_left",
) -> Any:
    """Draw boxes on an image tensor exactly as produced by OdDataset."""
    import matplotlib.pyplot as plt

    ax.imshow(_tensor_to_image_array(image_tensor))
    ax.axis("off")
    if title:
        ax.set_title(title)

    cmap = plt.get_cmap("tab10")
    class_to_idx = {name: index for index, name in enumerate(classes or [])}
    for box in boxes:
        x1, y1, x2, y2 = bbox_xyxy(box["bbox"])
        class_name = box.get("class", "object")
        color = edge_color or cmap(class_to_idx.get(class_name, 0) % 10)
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
            edgecolor=color,
            linestyle=line_style,
        )
        ax.add_patch(rect)
        if label_position == "top_left":
            label_x, label_y = x1, max(0, y1 - 4)
            horizontal_alignment = "left"
            vertical_alignment = "bottom"
        elif label_position == "bottom_right":
            label_x, label_y = x2, y2
            horizontal_alignment = "right"
            vertical_alignment = "top"
        else:
            raise ValueError("label_position must be 'top_left' or 'bottom_right'.")
        ax.text(
            label_x,
            label_y,
            f"{label_prefix}{class_name}",
            color="white",
            fontsize=9,
            horizontalalignment=horizontal_alignment,
            verticalalignment=vertical_alignment,
            bbox={"facecolor": color, "alpha": 0.85, "edgecolor": "none", "pad": 2},
        )
    return ax


def _image_has_class(dataset: Any, image_index: int, class_name: str) -> bool:
    image_id = dataset.images[image_index]["id"]
    return any(
        ann["class"] == class_name
        for ann in dataset.annotations_by_image.get(image_id, [])
    )


def _oversampling_weights(dataset: Any, class_name: str | None, factor: float) -> Any:
    import torch

    if not class_name:
        return torch.ones(len(dataset), dtype=torch.double)
    if class_name not in dataset.classes:
        raise ValueError(f"class_name must be one of {dataset.classes}.")
    if factor < 1.0:
        raise ValueError("factor must be greater than or equal to 1.0.")

    weights = [
        factor if _image_has_class(dataset, index, class_name) else 1.0
        for index in range(len(dataset))
    ]
    return torch.as_tensor(weights, dtype=torch.double)


def sample_training_indices(
    dataset: Any,
    oversample_class: str | None = None,
    oversample_factor: float = 1.0,
    num_samples: int | None = None,
    seed: int = 42,
) -> list[int]:
    """Return dataset indices in the same style as image-level oversampling."""
    import torch

    total = len(dataset) if num_samples is None else min(num_samples, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    weights = _oversampling_weights(dataset, oversample_class, oversample_factor)
    if oversample_class:
        indices = torch.multinomial(weights, total, replacement=True, generator=generator)
        return [int(index) for index in indices.tolist()]
    return torch.randperm(len(dataset), generator=generator)[:total].tolist()


def summarize_training_input(
    annotation_path: str | Path = "public/annotations/train.json",
    image_dir: str | Path = "public/train/images",
    oversample_class: str | None = "chair",
    oversample_factor: float = 2.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize raw annotations and the sampled training stream before the model."""
    import pandas as pd

    from utils.dataset import OdDataset

    dataset = OdDataset(annotation_path, image_dir)
    sampled_indices = sample_training_indices(
        dataset,
        oversample_class=oversample_class,
        oversample_factor=oversample_factor,
        num_samples=len(dataset),
        seed=seed,
    )

    raw_box_counts = Counter()
    raw_image_counts = Counter()
    for index, image in enumerate(dataset.images):
        image_id = image["id"]
        image_classes = set()
        for ann in dataset.annotations_by_image.get(image_id, []):
            raw_box_counts[ann["class"]] += 1
            image_classes.add(ann["class"])
        raw_image_counts.update(image_classes)

    sampled_box_counts = Counter()
    sampled_image_counts = Counter()
    for index in sampled_indices:
        image_id = dataset.images[index]["id"]
        image_classes = set()
        for ann in dataset.annotations_by_image.get(image_id, []):
            sampled_box_counts[ann["class"]] += 1
            image_classes.add(ann["class"])
        sampled_image_counts.update(image_classes)

    rows = []
    for class_name in dataset.classes:
        rows.append(
            {
                "class": class_name,
                "raw_images": raw_image_counts[class_name],
                "sampled_images": sampled_image_counts[class_name],
                "raw_boxes": raw_box_counts[class_name],
                "sampled_boxes": sampled_box_counts[class_name],
                "image_ratio_before": raw_image_counts[class_name] / max(len(dataset), 1),
                "image_ratio_after": sampled_image_counts[class_name] / max(len(sampled_indices), 1),
                "box_ratio_before": raw_box_counts[class_name] / max(sum(raw_box_counts.values()), 1),
                "box_ratio_after": sampled_box_counts[class_name] / max(sum(sampled_box_counts.values()), 1),
            }
        )

    image_box_counts = [
        len(dataset.annotations_by_image.get(image["id"], []))
        for image in dataset.images
    ]
    sampled_box_counts_per_image = [
        len(dataset.annotations_by_image.get(dataset.images[index]["id"], []))
        for index in sampled_indices
    ]
    summary_table = pd.DataFrame(
        [
            {
                "stage": "raw_dataset",
                "images_per_epoch": len(dataset),
                "boxes_per_epoch": sum(image_box_counts),
                "mean_boxes_per_image": sum(image_box_counts) / max(len(image_box_counts), 1),
                "empty_images": sum(1 for value in image_box_counts if value == 0),
            },
            {
                "stage": "after_oversampler",
                "images_per_epoch": len(sampled_indices),
                "boxes_per_epoch": sum(sampled_box_counts_per_image),
                "mean_boxes_per_image": sum(sampled_box_counts_per_image)
                / max(len(sampled_box_counts_per_image), 1),
                "empty_images": sum(1 for value in sampled_box_counts_per_image if value == 0),
            },
        ]
    )

    return {
        "dataset": dataset,
        "sampled_indices": sampled_indices,
        "summary": summary_table,
        "per_class": pd.DataFrame(rows),
        "oversampling": {
            "class": oversample_class,
            "factor": oversample_factor,
            "seed": seed,
        },
    }


def plot_training_input_distribution(
    training_input: dict[str, Any],
    ratio_kind: str = "image",
) -> Any:
    """Plot per-class ratios before and after oversampling."""
    import matplotlib.pyplot as plt

    if ratio_kind not in {"image", "box"}:
        raise ValueError("ratio_kind must be 'image' or 'box'.")
    before_col = f"{ratio_kind}_ratio_before"
    after_col = f"{ratio_kind}_ratio_after"
    table = training_input["per_class"].set_index("class")[[before_col, after_col]]
    ax = table.plot.bar(figsize=(10, 4), rot=0, color=["#4C78A8", "#F58518"])
    ax.set_title(f"Training input {ratio_kind} ratio before/after oversampling")
    ax.set_xlabel("Class")
    ax.set_ylabel("Ratio")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(["before", "after"])
    plt.tight_layout()
    return ax


def faster_rcnn_resize_scale(
    width: int,
    height: int,
    min_size: int = 512,
    max_size: int = 768,
) -> float:
    """Return the scale factor used by Faster R-CNN's resize transform."""
    short_side = min(width, height)
    long_side = max(width, height)
    if short_side <= 0 or long_side <= 0:
        raise ValueError("width and height must be positive.")
    scale = min_size / short_side
    if long_side * scale > max_size:
        scale = max_size / long_side
    return scale


def _nearest_anchor_size(value: float) -> int:
    candidates = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]
    return min(candidates, key=lambda candidate: abs(candidate - value))


def _recommend_anchor_sizes(sqrt_areas: Any, num_sizes: int = 5) -> list[int]:
    import numpy as np

    values = np.asarray(sqrt_areas, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return [32, 64, 128, 256, 512]

    percentiles = np.linspace(10, 90, num_sizes)
    sizes = [_nearest_anchor_size(float(np.percentile(values, percentile))) for percentile in percentiles]
    unique_sizes = []
    for size in sizes:
        if size not in unique_sizes:
            unique_sizes.append(size)

    fallback = [16, 32, 64, 128, 256, 512, 768]
    for size in fallback:
        if len(unique_sizes) >= num_sizes:
            break
        if size not in unique_sizes:
            unique_sizes.append(size)
    return sorted(unique_sizes[:num_sizes])


def _recommend_anchor_ratios(aspect_ratios: Any) -> list[float]:
    import numpy as np

    values = np.asarray(aspect_ratios, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    ratios = [0.5, 1.0, 2.0]
    if values.size == 0:
        return ratios

    q10, q90 = np.percentile(values, [10, 90])
    if q10 < 0.4:
        ratios.insert(0, 0.33)
    if q90 > 2.5:
        ratios.append(3.0)
    return ratios


def analyze_box_size_distribution(
    annotation_path: str | Path = "public/annotations/train.json",
    image_dir: str | Path = "public/train/images",
    min_size: int = 512,
    max_size: int = 768,
    top_k: int = 10,
) -> dict[str, Any]:
    """Analyze smallest/largest boxes and recommend resize/anchor settings."""
    import numpy as np
    import pandas as pd

    annotation = load_json(annotation_path)
    images = {image["id"]: image for image in annotation["images"]}
    rows = []
    missing_images = []

    for ann in annotation.get("annotations", []):
        image_info = images.get(ann["image_id"], {})
        image_path = resolve_image_path(image_dir, ann["image_id"], image_info.get("file_name"))
        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        except FileNotFoundError:
            missing_images.append(ann["image_id"])
            image_width = int(image_info.get("width", 0) or 0)
            image_height = int(image_info.get("height", 0) or 0)
            if image_width <= 0 or image_height <= 0:
                continue

        box_width, box_height = bbox_wh(ann["bbox"])
        if box_width <= 0 or box_height <= 0:
            continue
        scale = faster_rcnn_resize_scale(image_width, image_height, min_size, max_size)
        resized_width = box_width * scale
        resized_height = box_height * scale
        resized_area = resized_width * resized_height
        rows.append(
            {
                "image_id": ann["image_id"],
                "class": ann.get("class", "unknown"),
                "image_width": image_width,
                "image_height": image_height,
                "image_short_side": min(image_width, image_height),
                "image_long_side": max(image_width, image_height),
                "resize_scale": scale,
                "bbox": ann["bbox"],
                "width": box_width,
                "height": box_height,
                "area": box_width * box_height,
                "sqrt_area": (box_width * box_height) ** 0.5,
                "aspect_ratio": box_width / box_height,
                "resized_width": resized_width,
                "resized_height": resized_height,
                "resized_area": resized_area,
                "resized_sqrt_area": resized_area ** 0.5,
                "resized_aspect_ratio": resized_width / resized_height,
            }
        )

    box_df = pd.DataFrame(rows)
    if box_df.empty:
        raise ValueError("No valid boxes found for box size analysis.")

    numeric_columns = [
        "width",
        "height",
        "area",
        "sqrt_area",
        "aspect_ratio",
        "resized_width",
        "resized_height",
        "resized_area",
        "resized_sqrt_area",
        "resize_scale",
    ]
    summary = box_df[numeric_columns].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    per_class = (
        box_df.groupby("class")
        .agg(
            boxes=("bbox", "count"),
            min_resized_sqrt_area=("resized_sqrt_area", "min"),
            p10_resized_sqrt_area=("resized_sqrt_area", lambda value: float(np.percentile(value, 10))),
            median_resized_sqrt_area=("resized_sqrt_area", "median"),
            p90_resized_sqrt_area=("resized_sqrt_area", lambda value: float(np.percentile(value, 90))),
            max_resized_sqrt_area=("resized_sqrt_area", "max"),
            median_aspect_ratio=("aspect_ratio", "median"),
            p10_aspect_ratio=("aspect_ratio", lambda value: float(np.percentile(value, 10))),
            p90_aspect_ratio=("aspect_ratio", lambda value: float(np.percentile(value, 90))),
        )
        .reset_index()
        .sort_values("median_resized_sqrt_area")
    )

    anchor_sizes = _recommend_anchor_sizes(box_df["resized_sqrt_area"])
    anchor_ratios = _recommend_anchor_ratios(box_df["aspect_ratio"])
    resized_sqrt = box_df["resized_sqrt_area"]
    image_short_sides = box_df.drop_duplicates("image_id")["image_short_side"]
    image_long_sides = box_df.drop_duplicates("image_id")["image_long_side"]
    recommendations = {
        "current_resize": {"min_size": min_size, "max_size": max_size},
        "box_size_after_resize": {
            "p05": float(np.percentile(resized_sqrt, 5)),
            "p10": float(np.percentile(resized_sqrt, 10)),
            "median": float(np.percentile(resized_sqrt, 50)),
            "p90": float(np.percentile(resized_sqrt, 90)),
            "p95": float(np.percentile(resized_sqrt, 95)),
        },
        "image_size": {
            "median_short_side": float(image_short_sides.median()),
            "median_long_side": float(image_long_sides.median()),
            "p90_long_side": float(np.percentile(image_long_sides, 90)),
        },
        "recommended_anchor_sizes": anchor_sizes,
        "recommended_anchor_ratios": anchor_ratios,
        "torchvision_default_anchor_sizes": [32, 64, 128, 256, 512],
        "torchvision_default_anchor_ratios": [0.5, 1.0, 2.0],
        "suggestion": (
            "Keep min_size/max_size if resized boxes mostly fall between 16 and 512 px. "
            "If many resized_sqrt_area values are below 16 px, raise min_size or add a 16 px anchor. "
            "If many values exceed 512 px, raise max_size or keep a 512/768 anchor."
        ),
    }

    display_columns = [
        "image_id",
        "class",
        "bbox",
        "width",
        "height",
        "area",
        "aspect_ratio",
        "resized_width",
        "resized_height",
        "resized_area",
        "resized_sqrt_area",
    ]
    return {
        "boxes": box_df,
        "summary": summary,
        "per_class": per_class,
        "smallest": box_df.nsmallest(top_k, "resized_area")[display_columns],
        "largest": box_df.nlargest(top_k, "resized_area")[display_columns],
        "recommendations": recommendations,
        "missing_images": sorted(set(missing_images)),
    }


def plot_box_size_distribution(box_analysis: dict[str, Any]) -> Any:
    """Plot resized box scale and aspect-ratio distributions."""
    import matplotlib.pyplot as plt

    box_df = box_analysis["boxes"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    box_df["resized_sqrt_area"].plot.hist(bins=40, ax=axes[0], color="#4C78A8")
    for size in box_analysis["recommendations"]["recommended_anchor_sizes"]:
        axes[0].axvline(size, color="#F58518", linestyle="--", linewidth=1)
    axes[0].set_title("Resized bbox sqrt(area)")
    axes[0].set_xlabel("sqrt(area) after resize")
    axes[0].set_ylabel("Box count")

    box_df["aspect_ratio"].clip(upper=5).plot.hist(bins=40, ax=axes[1], color="#54A24B")
    for ratio in box_analysis["recommendations"]["recommended_anchor_ratios"]:
        axes[1].axvline(ratio, color="#E45756", linestyle="--", linewidth=1)
    axes[1].set_title("BBox aspect ratio")
    axes[1].set_xlabel("width / height, clipped at 5")
    axes[1].set_ylabel("Box count")
    plt.tight_layout()
    return axes


def show_training_input_slider(
    annotation_path: str | Path = "public/annotations/train.json",
    image_dir: str | Path = "public/train/images",
    oversample_class: str | None = "chair",
    oversample_factor: float = 2.0,
    augment: bool = True,
    horizontal_flip_probability: float = 0.5,
    color_jitter_probability: float = 0.0,
    grayscale_probability: float = 0.0,
    max_samples: int = 50,
    seed: int = 42,
) -> Any:
    """Browse images after optional oversampling and augmentation before model input."""
    import matplotlib.pyplot as plt

    from utils.dataset import OdDataset, build_train_transforms

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as error:
        raise RuntimeError("Install ipywidgets to use the notebook slider viewer.") from error

    transforms = (
        build_train_transforms(
            horizontal_flip_probability=horizontal_flip_probability,
            color_jitter_probability=color_jitter_probability,
            grayscale_probability=grayscale_probability,
        )
        if augment
        else None
    )
    dataset = OdDataset(annotation_path, image_dir, transforms=transforms)
    sampled_indices = sample_training_indices(
        dataset,
        oversample_class=oversample_class,
        oversample_factor=oversample_factor,
        num_samples=max_samples,
        seed=seed,
    )
    if not sampled_indices:
        raise ValueError("No samples available.")

    output = widgets.Output()
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(sampled_indices) - 1,
        step=1,
        description="Index",
        continuous_update=False,
    )

    def render(position: int) -> None:
        dataset_index = sampled_indices[position]
        image_info = dataset.images[dataset_index]
        image_tensor, target = dataset[dataset_index]
        boxes = _dataset_target_to_boxes(target, dataset.idx_to_class)
        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(9, 7))
            draw_tensor_boxes_on_axis(
                ax,
                image_tensor,
                boxes,
                classes=dataset.classes,
                title=(
                    f"{position + 1}/{len(sampled_indices)} | dataset_index={dataset_index} | "
                    f"{image_info['id']} | augment={augment}"
                ),
            )
            plt.show()
            plt.close(fig)

    slider.observe(lambda change: render(change["new"]), names="value")
    render(0)
    display(widgets.VBox([slider, output]))
    return slider


def show_augmentation_comparison_slider(
    annotation_path: str | Path = "public/annotations/train.json",
    image_dir: str | Path = "public/train/images",
    oversample_class: str | None = "chair",
    oversample_factor: float = 2.0,
    horizontal_flip_probability: float = 0.5,
    color_jitter_probability: float = 0.0,
    grayscale_probability: float = 0.0,
    max_samples: int = 30,
    seed: int = 42,
) -> Any:
    """Compare raw OdDataset output with augmented OdDataset output on the same image."""
    import matplotlib.pyplot as plt

    from utils.dataset import OdDataset, build_train_transforms

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as error:
        raise RuntimeError("Install ipywidgets to use the notebook slider viewer.") from error

    raw_dataset = OdDataset(annotation_path, image_dir)
    augmented_dataset = OdDataset(
        annotation_path,
        image_dir,
        transforms=build_train_transforms(
            horizontal_flip_probability=horizontal_flip_probability,
            color_jitter_probability=color_jitter_probability,
            grayscale_probability=grayscale_probability,
        ),
    )
    sampled_indices = sample_training_indices(
        raw_dataset,
        oversample_class=oversample_class,
        oversample_factor=oversample_factor,
        num_samples=max_samples,
        seed=seed,
    )
    if not sampled_indices:
        raise ValueError("No samples available.")

    output = widgets.Output()
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(sampled_indices) - 1,
        step=1,
        description="Index",
        continuous_update=False,
    )

    def render(position: int) -> None:
        dataset_index = sampled_indices[position]
        image_info = raw_dataset.images[dataset_index]
        raw_image, raw_target = raw_dataset[dataset_index]
        aug_image, aug_target = augmented_dataset[dataset_index]
        raw_boxes = _dataset_target_to_boxes(raw_target, raw_dataset.idx_to_class)
        aug_boxes = _dataset_target_to_boxes(aug_target, augmented_dataset.idx_to_class)
        with output:
            output.clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            draw_tensor_boxes_on_axis(
                axes[0],
                raw_image,
                raw_boxes,
                classes=raw_dataset.classes,
                title=f"Raw | dataset_index={dataset_index}",
            )
            draw_tensor_boxes_on_axis(
                axes[1],
                aug_image,
                aug_boxes,
                classes=augmented_dataset.classes,
                title=f"Augmented | {image_info['id']}",
            )
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    slider.observe(lambda change: render(change["new"]), names="value")
    render(0)
    display(widgets.VBox([slider, output]))
    return slider


def load_prediction_analysis(
    predictions_path: str | Path = "saved_results/baseline/predictions.json",
    ground_truth_path: str | Path = "public/annotations/val.json",
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Load extended metrics and error details for notebook exploration."""
    from utils.metric import (
        analyze_detection_errors,
        annotation_to_ground_truth,
        compute_confusion_matrix,
        evaluate_extended_metrics,
        prediction_list_to_dict,
    )

    annotation = load_json(ground_truth_path)
    ground_truth = annotation_to_ground_truth(annotation)
    predictions = prediction_list_to_dict(load_json(predictions_path))
    return {
        "classes": annotation["classes"],
        "ground_truth": ground_truth,
        "predictions": predictions,
        "metrics": evaluate_extended_metrics(ground_truth, predictions, annotation["classes"]),
        "errors": analyze_detection_errors(ground_truth, predictions, iou_threshold),
        "confusion": compute_confusion_matrix(
            ground_truth,
            predictions,
            annotation["classes"],
            iou_threshold,
        ),
    }


def confusion_matrix_table(analysis: dict[str, Any]) -> Any:
    """Return GT-vs-predicted-class confusion matrix as a DataFrame."""
    import pandas as pd

    matrix = analysis["confusion"]["matrix"]
    return pd.DataFrame.from_dict(matrix, orient="index").fillna(0).astype(int)


def confusion_pairs_table(analysis: dict[str, Any], include_correct: bool = False) -> Any:
    """Return non-zero confusion pairs sorted by count."""
    import pandas as pd

    rows = analysis["confusion"]["pairs"]
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=["gt_class", "predicted_class", "count"])
    if not include_correct:
        table = table[table["gt_class"] != table["predicted_class"]]
    return table.sort_values("count", ascending=False).reset_index(drop=True)


def plot_confusion_matrix(analysis: dict[str, Any]) -> Any:
    """Plot confusion matrix excluding background false-positive row."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    table = confusion_matrix_table(analysis)
    classes = analysis["classes"]
    plot_table = table.loc[classes, [*classes, "missed"]]
    fig_width = max(8, 1.2 * len(plot_table.columns))
    fig_height = max(5, 0.8 * len(plot_table.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(plot_table, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title(f"Confusion matrix @ IoU {analysis['confusion']['iou_threshold']}")
    plt.tight_layout()
    return ax


def prediction_metrics_table(analysis: dict[str, Any]) -> Any:
    """Return one-row summary DataFrame for extended detection metrics."""
    import pandas as pd

    metrics = analysis["metrics"]
    return pd.DataFrame(
        [
            {
                "mAP@0.5": metrics["mAP@0.5"],
                "mAP@0.75": metrics["mAP@0.75"],
                "mAP@0.5:0.95": metrics["mAP@0.5:0.95"],
                "precision": metrics["micro_precision"],
                "recall": metrics["micro_recall"],
                "ground_truth_boxes": metrics["num_ground_truth_boxes"],
                "predictions": metrics["num_predictions"],
            }
        ]
    )


def per_class_ap_table(analysis: dict[str, Any]) -> Any:
    """Return per-class AP, precision, recall, and detection counts."""
    import pandas as pd

    rows = []
    for class_name, metrics in analysis["metrics"]["per_class"].items():
        rows.append(
            {
                "class": class_name,
                "AP@0.5": metrics["ap@0.5"],
                "AP@0.75": metrics["ap@0.75"],
                "AP@0.5:0.95": metrics["ap@0.5:0.95"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "ground_truth": metrics["num_ground_truth"],
                "predictions": metrics["num_predictions"],
            }
        )
    return pd.DataFrame(rows).sort_values("AP@0.5:0.95", ascending=False).reset_index(drop=True)


def detection_error_table(analysis: dict[str, Any]) -> Any:
    """Return the main error categories sorted by frequency."""
    import pandas as pd

    return (
        pd.DataFrame(
            [
                {"error_type": error_type, "count": count}
                for error_type, count in analysis["errors"]["error_counts"].items()
            ]
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


ERROR_TYPE_LABELS = {
    "missed_detection": "false_negative",
    "background_false_positive": "false_positive",
    "classification_error": "wrong_class",
    "localization_error": "poor_localization",
    "duplicate_detection": "duplicate_boxes",
}


ERROR_GUIDANCE_ROWS = [
    {
        "Loại lỗi": "False Negative",
        "Dấu hiệu": "Có object thật nhưng model không detect",
        "Hướng xử lý": "Giảm confidence threshold, tăng image size, thêm dữ liệu, train thêm",
    },
    {
        "Loại lỗi": "False Positive",
        "Dấu hiệu": "Model detect nhầm background",
        "Hướng xử lý": "Tăng confidence threshold, thêm ảnh negative, kiểm tra annotation thiếu",
    },
    {
        "Loại lỗi": "Wrong Class",
        "Dấu hiệu": "Box đúng nhưng class sai",
        "Hướng xử lý": "Kiểm tra label, xử lý class imbalance, xem class có giống nhau không",
    },
    {
        "Loại lỗi": "Poor Localization",
        "Dấu hiệu": "Đúng class nhưng box lệch, mAP@0.75 thấp",
        "Hướng xử lý": "Kiểm tra annotation, train lâu hơn với LR nhỏ, tăng image size",
    },
    {
        "Loại lỗi": "Duplicate Boxes",
        "Dấu hiệu": "Một object có nhiều predicted boxes",
        "Hướng xử lý": "Điều chỉnh NMS threshold",
    },
    {
        "Loại lỗi": "Small Object Miss",
        "Dấu hiệu": "Object nhỏ thường bị bỏ sót",
        "Hướng xử lý": "Tăng resolution, oversampling ảnh có object nhỏ",
    },
    {
        "Loại lỗi": "Occlusion / Hard Background",
        "Dấu hiệu": "Object bị che khuất hoặc background giống object",
        "Hướng xử lý": "Thêm dữ liệu tương tự, augmentation phù hợp",
    },
]


def error_guidance_table() -> Any:
    import pandas as pd

    return pd.DataFrame(ERROR_GUIDANCE_ROWS)


def detection_error_table_readable(analysis: dict[str, Any]) -> Any:
    """Return required error categories with readable names."""
    import pandas as pd

    raw_counts = analysis["errors"]["error_counts"]
    rows = []
    for raw_name, readable in ERROR_TYPE_LABELS.items():
        rows.append(
            {
                "error_type": readable,
                "raw_error_type": raw_name,
                "count": int(raw_counts.get(raw_name, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def _lookup_image_info(ground_truth_path: str | Path) -> dict[str, dict[str, Any]]:
    annotation = load_json(ground_truth_path)
    return {image["id"]: image for image in annotation.get("images", [])}


def _box_relative_area_for_image(
    bbox: list[float],
    image_id: str,
    image_info_by_id: dict[str, dict[str, Any]],
    image_dir: str | Path,
) -> float:
    image_info = image_info_by_id.get(image_id, {})
    width = int(image_info.get("width", 0) or 0)
    height = int(image_info.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        try:
            with Image.open(resolve_image_path(image_dir, image_id, image_info.get("file_name"))) as image:
                width, height = image.size
        except FileNotFoundError:
            return 0.0
    return bbox_area(bbox) / max(float(width * height), 1.0)


def error_dataset_context_tables(
    analysis: dict[str, Any],
    ground_truth_path: str | Path,
    image_dir: str | Path,
    small_threshold: float = 0.01,
    medium_threshold: float = 0.10,
) -> dict[str, Any]:
    """Connect detection errors with class frequency and object size."""
    import pandas as pd

    image_info_by_id = _lookup_image_info(ground_truth_path)
    class_counts = Counter(
        box["class"]
        for boxes in analysis["ground_truth"].values()
        for box in boxes
    )
    class_error_counts: defaultdict[str, Counter] = defaultdict(Counter)
    missed_size_counts: Counter = Counter()
    false_positive_score_rows = []

    for image_id, details in analysis["errors"]["per_image"].items():
        for missed in details.get("missed_ground_truth", []):
            class_name = missed.get("class", "unknown")
            class_error_counts[class_name]["false_negative"] += 1
            relative_area = _box_relative_area_for_image(
                missed["bbox"],
                image_id,
                image_info_by_id,
                image_dir,
            )
            size_group = object_size_group(relative_area, small_threshold, medium_threshold)
            missed_size_counts[size_group] += 1

        for prediction in details.get("predictions", []):
            readable = ERROR_TYPE_LABELS.get(prediction.get("error_type"))
            if readable is None:
                continue
            class_name = prediction.get("class", "unknown")
            class_error_counts[class_name][readable] += 1
            if readable == "false_positive":
                false_positive_score_rows.append(
                    {
                        "image_id": image_id,
                        "class": class_name,
                        "confidence": prediction.get("confidence", 1.0),
                        "bbox": prediction.get("bbox"),
                    }
                )

    class_rows = []
    all_classes = sorted(set(class_counts) | set(class_error_counts))
    for class_name in all_classes:
        errors = class_error_counts[class_name]
        class_rows.append(
            {
                "class": class_name,
                "gt_objects": class_counts[class_name],
                "false_negative": errors["false_negative"],
                "false_positive": errors["false_positive"],
                "wrong_class": errors["wrong_class"],
                "poor_localization": errors["poor_localization"],
                "duplicate_boxes": errors["duplicate_boxes"],
                "fn_rate": errors["false_negative"] / max(class_counts[class_name], 1),
            }
        )

    return {
        "per_class_errors": pd.DataFrame(class_rows).sort_values("fn_rate", ascending=False),
        "missed_size_groups": pd.DataFrame(
            [
                {"size_group": group, "false_negative": missed_size_counts[group]}
                for group in ["small", "medium", "large"]
            ]
        ),
        "false_positive_scores": pd.DataFrame(false_positive_score_rows).sort_values(
            "confidence",
            ascending=False,
        )
        if false_positive_score_rows
        else pd.DataFrame(columns=["image_id", "class", "confidence", "bbox"]),
    }


def recommended_experiments_from_errors(
    analysis: dict[str, Any],
    context: dict[str, Any],
) -> Any:
    """Suggest targeted experiments from observed error counts."""
    import pandas as pd

    counts = detection_error_table_readable(analysis).set_index("error_type")["count"].to_dict()
    missed_sizes = context["missed_size_groups"].set_index("size_group")["false_negative"].to_dict()
    per_class = context["per_class_errors"]
    weak_classes = per_class.sort_values("fn_rate", ascending=False).head(3)["class"].tolist()
    rows = []

    if counts.get("false_negative", 0) > 0:
        rows.append(
            {
                "priority": counts["false_negative"],
                "evidence": "False negatives exist; model misses real objects.",
                "experiment": "Sweep lower SCORE_THRESHOLD during predict/evaluate; if recall improves, use lower threshold for submission.",
            }
        )
    if missed_sizes.get("small", 0) >= max(missed_sizes.get("medium", 0), missed_sizes.get("large", 0)):
        rows.append(
            {
                "priority": missed_sizes.get("small", 0),
                "evidence": "Small objects dominate missed detections.",
                "experiment": "Increase MIN_SIZE/MAX_SIZE or oversample images containing small objects.",
            }
        )
    if counts.get("false_positive", 0) > 0:
        rows.append(
            {
                "priority": counts["false_positive"],
                "evidence": "False positives indicate background confusion.",
                "experiment": "Raise SCORE_THRESHOLD; inspect high-confidence FP and add similar negative/background-heavy images.",
            }
        )
    if counts.get("wrong_class", 0) > 0:
        rows.append(
            {
                "priority": counts["wrong_class"],
                "evidence": f"Wrong-class errors; weakest classes by FN rate: {weak_classes}.",
                "experiment": "Check labels and class similarity; try class/instance oversampling for weak classes.",
            }
        )
    if counts.get("poor_localization", 0) > 0:
        rows.append(
            {
                "priority": counts["poor_localization"],
                "evidence": "Poor localization hurts stricter IoU metrics.",
                "experiment": "Train longer with cosine/plateau LR, inspect annotation tightness, consider higher image size.",
            }
        )
    if counts.get("duplicate_boxes", 0) > 0:
        rows.append(
            {
                "priority": counts["duplicate_boxes"],
                "evidence": "Duplicate boxes remain after NMS.",
                "experiment": "Sweep lower NMS_THRESHOLD and compare precision/recall tradeoff.",
            }
        )

    if not rows:
        rows.append(
            {
                "priority": 0,
                "evidence": "No dominant error category found.",
                "experiment": "Keep current model and inspect qualitative examples before changing architecture.",
            }
        )
    return pd.DataFrame(rows).sort_values("priority", ascending=False).reset_index(drop=True)


def plot_per_class_ap(analysis: dict[str, Any]) -> Any:
    """Plot AP50, AP75, and AP50:95 for each class."""
    import matplotlib.pyplot as plt

    table = per_class_ap_table(analysis).set_index("class")
    ax = table[["AP@0.5", "AP@0.75", "AP@0.5:0.95"]].plot.bar(
        figsize=(10, 5),
        ylim=(0, 1),
        rot=0,
        color=["#2ca02c", "#ff7f0e", "#1f77b4"],
    )
    ax.set_title("Average Precision by class")
    ax.set_ylabel("AP")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return ax


def plot_detection_errors(analysis: dict[str, Any]) -> Any:
    """Plot the number of detections in each error category."""
    import matplotlib.pyplot as plt

    table = detection_error_table(analysis)
    ax = table.plot.barh(
        x="error_type",
        y="count",
        figsize=(9, 4),
        legend=False,
        color="#d62728",
    )
    ax.invert_yaxis()
    ax.set_title("Detection error categories")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return ax


def _select_analysis_images(
    per_image: dict[str, dict[str, Any]],
    categories: tuple[str, ...],
    max_images: int,
) -> list[str]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for image_id, details in per_image.items():
        grouped[details["category"]].append(image_id)

    quota = max(1, max_images // max(len(categories), 1))
    selected = [image_id for category in categories for image_id in grouped[category][:quota]]
    if len(selected) < max_images:
        selected_set = set(selected)
        remaining = [
            image_id
            for category in categories
            for image_id in grouped[category]
            if image_id not in selected_set
        ]
        selected.extend(remaining[: max_images - len(selected)])
    return selected[:max_images]


def show_prediction_analysis_slider(
    analysis: dict[str, Any],
    image_dir: str | Path = "public/val/images",
    categories: tuple[str, ...] = ("good", "incorrect", "missed", "mixed"),
    max_images: int = 50,
) -> Any:
    """Browse a balanced validation sample with TP, FP, FN, and error labels."""
    import matplotlib.pyplot as plt

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as error:
        raise RuntimeError("Install ipywidgets to use the notebook slider viewer.") from error

    per_image = analysis["errors"]["per_image"]
    image_ids = _select_analysis_images(per_image, categories, max_images)
    if not image_ids:
        raise ValueError("No images match the selected error categories.")

    output = widgets.Output()
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(image_ids) - 1,
        step=1,
        description="Index",
        continuous_update=False,
    )

    def render(index: int) -> None:
        image_id = image_ids[index]
        details = per_image[image_id]
        image_path = resolve_image_path(image_dir, image_id)
        with output:
            output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 7))
            draw_boxes_on_axis(ax, image_path, [], classes=analysis["classes"])
            for prediction in details["predictions"]:
                is_true_positive = prediction["error_type"] == "true_positive"
                readable_error = ERROR_TYPE_LABELS.get(prediction["error_type"], prediction["error_type"])
                draw_boxes_on_axis(
                    ax,
                    image_path,
                    [prediction],
                    classes=analysis["classes"],
                    label_prefix=(
                        "Pred TP: "
                        if is_true_positive
                        else f"Pred {readable_error}: "
                    ),
                    edge_color="#2ca02c" if is_true_positive else "#d62728",
                    label_position="top_left",
                )
            draw_boxes_on_axis(
                ax,
                image_path,
                analysis["ground_truth"].get(image_id, []),
                classes=analysis["classes"],
                label_prefix="GT: ",
                edge_color="black",
                line_style="--",
                label_position="bottom_right",
            )
            ax.set_title(
                f"{index + 1}/{len(image_ids)} | {details['category'].upper()} | {image_id} | "
                f"TP={details['true_positives']} FP={details['false_positives']} "
                f"FN={details['false_negatives']}"
            )
            plt.show()
            plt.close(fig)

    slider.observe(lambda change: render(change["new"]), names="value")
    render(0)
    display(widgets.VBox([slider, output]))
    return slider


def tune_prediction_thresholds(
    predictions_path: str | Path = "saved_results/baseline/predictions_raw.json",
    ground_truth_path: str | Path = "public/annotations/val.json",
    confidence_thresholds: list[float] | None = None,
    nms_thresholds: list[float] | None = None,
    iou_threshold: float = 0.5,
) -> Any:
    """Run an offline confidence/NMS sweep and return a sorted DataFrame."""
    import pandas as pd

    from utils.metric import annotation_to_ground_truth, prediction_list_to_dict
    from utils.tune_thresholds import tune_thresholds

    annotation = load_json(ground_truth_path)
    results = tune_thresholds(
        annotation_to_ground_truth(annotation),
        prediction_list_to_dict(load_json(predictions_path)),
        annotation["classes"],
        confidence_thresholds or [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        nms_thresholds or [0.3, 0.4, 0.5, 0.6, 0.7],
        iou_threshold,
    )
    return pd.DataFrame(results)


def plot_threshold_tuning_heatmap(results: Any, metric: str = "mAP") -> Any:
    """Plot an offline confidence/NMS sweep as a heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if metric not in results.columns:
        raise ValueError(f"Unknown metric {metric!r}. Choose one of: {list(results.columns)}")
    matrix = results.pivot(
        index="confidence_threshold",
        columns="nms_threshold",
        values=metric,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(matrix, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax)
    ax.set_title(f"Threshold tuning: {metric}")
    ax.set_xlabel("NMS threshold")
    ax.set_ylabel("Confidence threshold")
    fig.tight_layout()
    return ax


def draw_boxes(
    image_path: str | Path,
    boxes: list[dict[str, Any]],
    output_path: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
) -> None:
    """Draw ground-truth or prediction boxes for notebook/debug usage."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title)

    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box["bbox"]]
        label = box.get("class", "object")
        confidence = box.get("confidence")
        caption = f"{label} {confidence:.2f}" if confidence is not None else label
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(0, y1 - 4),
            caption,
            color="black",
            fontsize=9,
            bbox={"facecolor": "lime", "alpha": 0.8, "pad": 2},
        )

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def visualize_random_sample(
    annotation_path: str | Path,
    image_dir: str | Path,
    output_path: str | Path = "debug_sample.jpg",
    seed: int = 42,
) -> Path:
    data = load_json(annotation_path)
    annotations = index_annotations(data)
    rng = random.Random(seed)
    image = rng.choice(data["images"])
    boxes = annotations.get(image["id"], [])
    image_path = resolve_image_path(image_dir, image["id"], image.get("file_name"))
    draw_boxes(image_path, boxes, output_path, title=image["id"])
    return Path(output_path)
