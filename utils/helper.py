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


def load_prediction_analysis(
    predictions_path: str | Path = "saved_results/predictions.json",
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
    predictions_path: str | Path = "saved_results/predictions_raw.json",
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
        download_public_dataset_from_kaggle(
            dataset_slug=args.dataset_slug,
            local_zip=args.local_zip,
            output_dir=args.dataset_output_dir,
            dataset_dir_name=args.dataset_dir_name,
            force=args.force_download,
        )
        return

    summary = dataset_summary(args.annotation)
    print(json.dumps(summary, indent=2))
    output = visualize_random_sample(args.annotation, args.image_dir, args.output)
    print(f"Saved debug visualization to {output}")


if __name__ == "__main__":
    main()
