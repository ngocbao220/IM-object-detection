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

import torch
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


def get_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_checkpoint_value_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _move_checkpoint_value_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_checkpoint_value_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_checkpoint_value_to_cpu(item) for item in value)
    return value


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    classes: list[str],
    metrics: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": _move_checkpoint_value_to_cpu(model.state_dict()),
        "optimizer_state_dict": (
            _move_checkpoint_value_to_cpu(optimizer.state_dict()) if optimizer is not None else None
        ),
        "classes": classes,
        "metrics": metrics or {},
        "model_config": model_config or {},
    }
    torch.save(checkpoint, output)
    del checkpoint


def save_checkpoint_with_alias(
    path: str | Path,
    alias_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    classes: list[str],
    metrics: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> None:
    save_checkpoint(path, model, optimizer, epoch, classes, metrics, model_config)
    alias = Path(alias_path)
    alias.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, alias)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state_to_device(optimizer, device)
    return checkpoint


def move_targets_to_device(
    targets: list[dict[str, torch.Tensor]], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = box_area(boxes1)[:, None] + box_area(boxes2) - inter
    return inter / union.clamp(min=1e-6)


def clip_boxes_to_image(boxes: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)
    return boxes


def remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    return torch.where((widths >= min_size) & (heights >= min_size))[0]


def encode_boxes(reference_boxes: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
    widths = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
    heights = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights

    gt_widths = (reference_boxes[:, 2] - reference_boxes[:, 0]).clamp(min=1e-6)
    gt_heights = (reference_boxes[:, 3] - reference_boxes[:, 1]).clamp(min=1e-6)
    gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

    return torch.stack(
        [
            (gt_ctr_x - ctr_x) / widths,
            (gt_ctr_y - ctr_y) / heights,
            torch.log(gt_widths / widths),
            torch.log(gt_heights / heights),
        ],
        dim=1,
    )


def decode_boxes(deltas: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0].clamp(min=-10, max=10)
    dy = deltas[:, 1].clamp(min=-10, max=10)
    dw = deltas[:, 2].clamp(min=-5, max=5)
    dh = deltas[:, 3].clamp(min=-5, max=5)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights
    return torch.stack(
        [
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ],
        dim=1,
    )


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



def resolve_image_path(image_dir: str | Path, image_id: str, file_name: str | None = None) -> Path:
    image_dir = Path(image_dir)
    candidates = [image_dir / image_id]
    if file_name:
        candidates.extend([image_dir / file_name, image_dir.parent / file_name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime helpers for the OD project.")
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
    if not args.download_dataset:
        raise SystemExit("Nothing to do. Use --download_dataset to fetch/install the public dataset.")

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


if __name__ == "__main__":
    main()
