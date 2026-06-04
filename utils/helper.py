from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def print_run_configuration(title: str, parameters: dict[str, Any]) -> None:
    """Print a readable configuration block before a CLI task starts."""
    border = "=" * 12
    print(f"{border} {title} {border}", flush=True)
    for key, value in parameters.items():
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False)
        print(f"{key}: {value}", flush=True)
    print("=" * (len(title) + len(border) * 2 + 2), flush=True)


def find_kaggle_dataset_slug(
    metadata_paths: list[str | Path] | None = None,
) -> str | None:
    """Find a Kaggle dataset slug from env or common metadata files.

    Expected slug format is "owner/dataset-name". Set KAGGLE_DATASET_SLUG on cloud
    if the project metadata does not contain the dataset source.
    """
    env_slug = os.getenv("KAGGLE_DATASET_SLUG")
    if env_slug:
        return env_slug

    paths = metadata_paths or ["kernel-metadata.json", "kaggle.yml"]
    for path in paths:
        candidate = Path(path)
        if not candidate.exists():
            continue
        try:
            if candidate.suffix == ".json":
                data = load_json(candidate)
                sources = data.get("dataset_sources") or data.get("datasets") or []
                if sources:
                    source = sources[0]
                    if isinstance(source, str):
                        return source.replace("kaggle/input/", "").strip("/")
                    if isinstance(source, dict):
                        return source.get("source") or source.get("slug") or source.get("dataset")
            else:
                text = candidate.read_text(encoding="utf-8")
                for line in text.splitlines():
                    stripped = line.strip().strip("'\"")
                    if "/" in stripped and not stripped.startswith("#"):
                        return stripped.lstrip("- ").strip("'\"")
        except (OSError, json.JSONDecodeError):
            continue
    return None


def split_kaggle_dataset_reference(reference: str) -> tuple[str, str | None]:
    """Split owner/dataset[/file.zip] into a Kaggle slug and optional inner file."""
    parts = reference.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError("Kaggle dataset reference must look like owner/dataset-name.")
    slug = "/".join(parts[:2])
    inner_file = "/".join(parts[2:]) if len(parts) > 2 else None
    return slug, inner_file


def kaggle_slug_candidates(kaggle_slug: str) -> list[str]:
    """Return likely Kaggle slug variants for hyphen/underscore naming."""
    owner, dataset_name = kaggle_slug.split("/", maxsplit=1)
    names = [
        dataset_name,
        dataset_name.replace("_", "-"),
        dataset_name.replace("-", "_"),
    ]
    candidates = []
    for name in names:
        candidate = f"{owner}/{name}"
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def maybe_extract_nested_zip(extract_dir: Path, inner_file: str | None = None) -> Path:
    """Return the directory containing dataset files, extracting nested zips if needed."""
    if inner_file:
        nested_zip = extract_dir / inner_file
        if not nested_zip.exists():
            matches = list(extract_dir.rglob(Path(inner_file).name))
            if not matches:
                print(
                    f"Could not find {inner_file} inside Kaggle dataset; "
                    "using extracted dataset contents instead."
                )
                return maybe_extract_nested_zip(extract_dir, inner_file=None)
            nested_zip = matches[0]
        if nested_zip.suffix.lower() != ".zip":
            return nested_zip.parent
        nested_extract_dir = extract_dir / "nested_extracted"
        extract_zip(nested_zip, nested_extract_dir)
        return nested_extract_dir

    top_level_zips = [path for path in extract_dir.iterdir() if path.suffix.lower() == ".zip"]
    top_level_dirs = [path for path in extract_dir.iterdir() if path.is_dir()]
    if len(top_level_zips) == 1 and not top_level_dirs:
        nested_extract_dir = extract_dir / "nested_extracted"
        extract_zip(top_level_zips[0], nested_extract_dir)
        return nested_extract_dir
    return extract_dir


def find_dataset_source_dir(extract_dir: Path, dataset_dir_name: str = "public") -> Path:
    """Find the directory that contains the public OD dataset layout."""
    direct_public = extract_dir / dataset_dir_name
    if direct_public.exists():
        return direct_public

    if (extract_dir / "annotations").exists() and (extract_dir / "classes.json").exists():
        return extract_dir

    for candidate in extract_dir.rglob(dataset_dir_name):
        if candidate.is_dir() and (candidate / "annotations").exists():
            return candidate

    for candidate in extract_dir.rglob("classes.json"):
        parent = candidate.parent
        if (parent / "annotations").exists():
            return parent

    return extract_dir


def move_dataset_source_to_target(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    for item in source_dir.iterdir():
        shutil.move(str(item), str(target_dir / item.name))


def find_mounted_kaggle_file(dataset_reference: str) -> Path | None:
    """Find an attached Kaggle dataset file under /kaggle/input when available."""
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None

    kaggle_slug, inner_file = split_kaggle_dataset_reference(dataset_reference)
    dataset_name = kaggle_slug.split("/", maxsplit=1)[1]
    file_name = Path(inner_file).name if inner_file else None

    candidates: list[Path] = []
    if file_name:
        candidates.extend(
            [
                kaggle_input / dataset_name / file_name,
                kaggle_input / dataset_name.replace("_", "-") / file_name,
                kaggle_input / dataset_name.replace("-", "_") / file_name,
            ]
        )
        candidates.extend(kaggle_input.rglob(file_name))
    else:
        candidates.extend(
            [
                kaggle_input / dataset_name,
                kaggle_input / dataset_name.replace("_", "-"),
                kaggle_input / dataset_name.replace("-", "_"),
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def install_dataset_from_source(
    source_path: Path,
    target_dir: Path,
    dataset_dir_name: str = "public",
) -> Path:
    """Install a dataset from a local zip or directory into target_dir."""
    tmp_dir = target_dir.parent / ".kaggle_local_extract"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        if source_path.is_dir():
            dataset_source_dir = source_path
        else:
            extract_dir = tmp_dir / "extracted"
            extract_zip(source_path, extract_dir)
            dataset_source_dir = maybe_extract_nested_zip(extract_dir)

        normalized_source_dir = find_dataset_source_dir(dataset_source_dir, dataset_dir_name)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(normalized_source_dir, target_dir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    print(f"Dataset ready at {target_dir}")
    return target_dir


def download_public_dataset_from_kaggle(
    dataset_slug: str | None = None,
    local_zip: str | Path | None = None,
    output_dir: str | Path = ".",
    dataset_dir_name: str = "public",
    force: bool = False,
) -> Path:
    """Download and extract the Kaggle public dataset into the current project.

    Requires Kaggle credentials to be available through kaggle.json or the
    KAGGLE_USERNAME/KAGGLE_KEY environment variables.
    """
    output_dir = Path(output_dir)
    target_dir = output_dir / dataset_dir_name
    if target_dir.exists() and not force:
        print(f"Dataset already exists at {target_dir}. Use force=True to re-download.")
        return target_dir

    if local_zip is not None:
        return install_dataset_from_source(Path(local_zip), target_dir, dataset_dir_name)

    dataset_reference = dataset_slug or find_kaggle_dataset_slug()
    if not dataset_reference:
        raise ValueError(
            "Missing Kaggle dataset slug. Pass dataset_slug='owner/dataset-name' "
            "or set KAGGLE_DATASET_SLUG."
        )
    kaggle_slug, inner_file = split_kaggle_dataset_reference(dataset_reference)

    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI is not installed. Install it with `pip install kaggle` "
            "and configure credentials before downloading."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / ".kaggle_download"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    command = []
    last_error: subprocess.CalledProcessError | None = None
    try:
        for candidate_slug in kaggle_slug_candidates(kaggle_slug):
            command = [
                "kaggle",
                "datasets",
                "download",
                "-d",
                candidate_slug,
                "-p",
                str(tmp_dir),
            ]
            print(f"Downloading Kaggle dataset {candidate_slug}...")
            try:
                subprocess.run(command, check=True)
                kaggle_slug = candidate_slug
                break
            except subprocess.CalledProcessError as error:
                last_error = error
                print(f"Download failed for {candidate_slug}; trying next candidate if any.")
        else:
            raise last_error or RuntimeError("Kaggle download failed.")
    except subprocess.CalledProcessError as error:
        mounted_file = find_mounted_kaggle_file(dataset_reference)
        if mounted_file is not None:
            print(f"Kaggle API failed, using mounted dataset file: {mounted_file}")
            shutil.rmtree(tmp_dir)
            return install_dataset_from_source(mounted_file, target_dir, dataset_dir_name)
        raise RuntimeError(
            "Kaggle refused the dataset download. Common fixes: make the dataset public, "
            "verify the exact slug from the Kaggle dataset URL, use a kaggle.json token "
            "from an account that can access it, or attach the dataset to a Kaggle "
            "notebook and pass `--local_zip /kaggle/input/<dataset>/final_public.zip`."
        ) from error

    zip_files = sorted(tmp_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"Kaggle download did not create a zip file in {tmp_dir}.")

    extract_dir = tmp_dir / "extracted"
    extract_dir.mkdir()
    extract_zip(zip_files[0], extract_dir)

    dataset_source_dir = maybe_extract_nested_zip(extract_dir, inner_file)
    normalized_source_dir = find_dataset_source_dir(dataset_source_dir, dataset_dir_name)
    move_dataset_source_to_target(normalized_source_dir, target_dir)

    shutil.rmtree(tmp_dir)
    print(f"Dataset ready at {target_dir}")
    return target_dir


def load_classes(path: str | Path = "public/classes.json") -> list[str]:
    data = load_json(path)
    if isinstance(data, dict) and "classes" in data:
        return list(data["classes"])
    return list(data)


def build_class_maps(classes: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Return Faster R-CNN label maps. Label 0 is reserved for background."""
    class_to_idx = {name: idx + 1 for idx, name in enumerate(classes)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


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
    }


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
                draw_boxes_on_axis(
                    ax,
                    image_path,
                    [prediction],
                    classes=analysis["classes"],
                    label_prefix=(
                        "Pred TP: "
                        if is_true_positive
                        else f"Pred {prediction['error_type']}: "
                    ),
                    edge_color="#2ca02c" if is_true_positive else "#d62728",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug helpers for the OD dataset.")
    parser.add_argument("--annotation", default="public/annotations/train.json")
    parser.add_argument("--image_dir", default="public/train/images")
    parser.add_argument("--output", default="debug_sample.jpg")
    parser.add_argument("--download_dataset", action="store_true")
    parser.add_argument("--dataset_slug", default=None, help="Kaggle dataset slug: owner/name.")
    parser.add_argument(
        "--local_zip",
        default=None,
        help="Local dataset zip, e.g. /kaggle/input/<dataset>/final_public.zip.",
    )
    parser.add_argument("--dataset_output_dir", default=".")
    parser.add_argument("--dataset_dir_name", default="public")
    parser.add_argument("--force_download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download_dataset:
        print_run_configuration(
            "Dataset Download",
            {
                "dataset_slug": args.dataset_slug or find_kaggle_dataset_slug(),
                "local_zip": args.local_zip or "not set",
                "dataset_output_dir": Path(args.dataset_output_dir),
                "dataset_dir_name": args.dataset_dir_name,
                "force_download": args.force_download,
            },
        )
        download_public_dataset_from_kaggle(
            dataset_slug=args.dataset_slug,
            local_zip=args.local_zip,
            output_dir=args.dataset_output_dir,
            dataset_dir_name=args.dataset_dir_name,
            force=args.force_download,
        )
        return

    print_run_configuration(
        "Dataset Debug Sample",
        {
            "annotation": Path(args.annotation),
            "image_dir": Path(args.image_dir),
            "output": Path(args.output),
        },
    )
    summary = dataset_summary(args.annotation)
    print(json.dumps(summary, indent=2))
    output = visualize_random_sample(args.annotation, args.image_dir, args.output)
    print(f"Saved debug visualization to {output}")


if __name__ == "__main__":
    main()
