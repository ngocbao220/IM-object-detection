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
