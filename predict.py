from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import functional as F
from tqdm.auto import tqdm

from models.factory import MODEL_IMPL_CHOICES, create_detection_model, normalize_model_impl
from models.modules import BACKBONE_WEIGHTS
from utils.helper import get_device, load_checkpoint, load_classes, print_run_configuration


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Faster R-CNN inference.")
    parser.add_argument("--image_dir", required=True, help="Image file or directory.")
    parser.add_argument("--output", required=True, help="Output predictions.json path.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint path. If omitted with --model_name, predict.py uses "
            "saved_results/<model_name>/checkpoints/<version>_model.pth."
        ),
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Optional Hugging Face model/run name to download when the checkpoint is missing.",
    )
    parser.add_argument(
        "--model_version",
        default="latest",
        help="Hugging Face revision to download when --model_name is used. Default: latest.",
    )
    parser.add_argument(
        "--version",
        choices=("best", "last"),
        default="best",
        help="Checkpoint alias to use with --model_name. Default: best.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download from Hugging Face even if the selected checkpoint already exists locally.",
    )
    parser.add_argument("--classes", default="public/classes.json")
    parser.add_argument("--score_threshold", type=float, default=0.0001)
    parser.add_argument("--nms_threshold", type=float, default=0.4)
    parser.add_argument("--backbone", choices=sorted(BACKBONE_WEIGHTS), default="resnet101")
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Use the repository's Faster R-CNN implementation.",
    )
    parser.add_argument(
        "--model_impl",
        choices=MODEL_IMPL_CHOICES,
        default="faster_rcnn",
        help="Detection model implementation to load.",
    )
    parser.add_argument("--min_size", type=int, default=768)
    parser.add_argument("--max_size", type=int, default=1024)
    parser.add_argument("--anchor_sizes", default="", help="Optional comma-separated anchor sizes.")
    parser.add_argument("--anchor_ratios", default="", help="Optional comma-separated anchor aspect ratios.")
    parser.add_argument("--retina_topk_candidates", type=int, default=1000)
    parser.add_argument("--retina_max_detections", type=int, default=300)
    parser.add_argument("--yolo_topk_candidates", type=int, default=1000)
    parser.add_argument("--yolo_max_detections", type=int, default=300)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def default_checkpoint_for_model(model_name: str, version: str = "best") -> Path:
    return Path("saved_results") / model_name / "checkpoints" / f"{version}_model.pth"


def find_downloaded_checkpoint(model_name: str, preferred_path: Path, version: str = "best") -> Path:
    if preferred_path.exists():
        return preferred_path

    checkpoint_dir = Path("saved_results") / model_name / "checkpoints"
    ordered_filenames = [f"{version}_model.pth"]
    ordered_filenames.extend(
        filename
        for filename in ("best_model.pth", "last_model.pth")
        if filename not in ordered_filenames
    )
    for filename in ordered_filenames:
        candidate = checkpoint_dir / filename
        if candidate.exists():
            return candidate

    candidates = sorted(
        checkpoint_dir.glob("*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return preferred_path


def ensure_checkpoint_available(args: argparse.Namespace) -> Path:
    if args.force_download and not args.model_name:
        raise ValueError("--force-download requires --model_name so predict.py knows what to download.")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif args.model_name:
        checkpoint_path = default_checkpoint_for_model(args.model_name, args.version)
    else:
        checkpoint_path = Path("saved_results/baseline/checkpoints/best_model.pth")

    if checkpoint_path.exists() and not args.force_download:
        return checkpoint_path
    if not args.model_name:
        return checkpoint_path

    download_script = Path(__file__).resolve().with_name("download.sh")
    if not download_script.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}, and download.sh was not found."
        )

    if args.force_download:
        print("Force download enabled; downloading checkpoint from Hugging Face...")
    else:
        print("Checkpoint not found; downloading checkpoint from Hugging Face...")
    command = [
        "bash",
        str(download_script),
        "--model",
        f"MODEL_NAME={args.model_name}",
        f"MODEL_VERSION={args.model_version}",
    ]
    subprocess.run(command, check=True)
    checkpoint_path = find_downloaded_checkpoint(args.model_name, checkpoint_path, args.version)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Downloaded Hugging Face model '{args.model_name}', but no .pth checkpoint was found."
        )
    return checkpoint_path


def parse_optional_int_tuple(value: str) -> tuple[int, ...] | None:
    if not value.strip():
        return None
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if any(item <= 0 for item in parsed):
        raise ValueError("Anchor sizes must contain positive integers.")
    return parsed


def parse_optional_float_tuple(value: str) -> tuple[float, ...] | None:
    if not value.strip():
        return None
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if any(item <= 0 for item in parsed):
        raise ValueError("Anchor ratios must contain positive values.")
    return parsed


def list_images(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def clamp_box(box: list[float], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = box
    x1 = int(math.floor(max(0, min(width, x1))))
    x2 = int(math.ceil(max(0, min(width, x2))))
    y1 = int(math.floor(max(0, min(height, y1))))
    y2 = int(math.ceil(max(0, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


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

    for path in tqdm(image_paths, desc="predict", file=sys.stdout):
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
            bbox = clamp_box([float(v) for v in box.detach().cpu().tolist()], width, height)
            if bbox is None:
                continue
            boxes.append(
                {
                    "class": idx_to_class.get(label_id, str(label_id)),
                    "confidence": round(confidence, 6),
                    "bbox": bbox,
                }
            )

        results.append({"image_id": path.name, "boxes": boxes})
    return results


def main() -> None:
    args = parse_args()
    if args.custom:
        args.model_impl = "faster_rcnn"
    args.model_impl = normalize_model_impl(args.model_impl)
    if args.min_size <= 0 or args.max_size <= 0 or args.min_size > args.max_size:
        raise ValueError("--min_size and --max_size must be positive with min_size <= max_size.")
    image_paths = list_images(args.image_dir)
    classes = load_classes(args.classes)
    idx_to_class = {idx + 1: name for idx, name in enumerate(classes)}
    device = get_device(args.device)
    checkpoint_path = ensure_checkpoint_available(args)
    checkpoint_metadata = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint_path.exists()
        else {}
    )
    model_config = checkpoint_metadata.get("model_config", {})
    checkpoint_state = checkpoint_metadata.get("model_state_dict", {})
    class_loss_weights = checkpoint_state.get("class_loss_weights")
    model_impl = normalize_model_impl(model_config.get("model_impl", args.model_impl))
    backbone = model_config.get("backbone", args.backbone)
    min_size = int(model_config.get("min_size", args.min_size))
    max_size = int(model_config.get("max_size", args.max_size))
    anchor_sizes = model_config.get("anchor_sizes") or parse_optional_int_tuple(args.anchor_sizes)
    anchor_ratios = model_config.get("anchor_ratios") or parse_optional_float_tuple(args.anchor_ratios)
    anchor_sizes = tuple(anchor_sizes) if anchor_sizes else None
    anchor_ratios = tuple(anchor_ratios) if anchor_ratios else None

    print_run_configuration(
        "Prediction Session",
        {
            "image_source": Path(args.image_dir),
            "num_images": len(image_paths),
            "output": Path(args.output),
            "checkpoint": checkpoint_path,
            "checkpoint_exists": checkpoint_path.exists(),
            "model_name": args.model_name or "not_provided",
            "model_version": args.model_version if args.model_name else "not_used",
            "classes": classes,
            "device": device,
            "score_threshold": args.score_threshold,
            "nms_threshold": args.nms_threshold,
            "model": f"{model_impl} {backbone}",
            "custom_model": model_impl == "faster_rcnn",
            "model_impl": model_impl,
            "min_size": min_size,
            "max_size": max_size,
            "anchor_sizes": anchor_sizes or "model_default",
            "anchor_ratios": anchor_ratios or "model_default",
            "class_loss_weights": "checkpoint" if class_loss_weights is not None else "not_used",
            "model_config_source": "checkpoint" if model_config else "CLI",
        },
    )

    model = create_detection_model(
        model_impl=model_impl,
        num_classes=len(classes) + 1,
        backbone_name=backbone,
        pretrained_backbone=False,
        trainable_backbone_layers=int(model_config.get("trainable_backbone_layers", 2)),
        box_score_thresh=args.score_threshold,
        box_nms_thresh=args.nms_threshold,
        min_size=min_size,
        max_size=max_size,
        anchor_sizes=anchor_sizes,
        anchor_ratios=anchor_ratios,
        train_pre_nms_top_n=2000,
        train_post_nms_top_n=2000,
        test_pre_nms_top_n=1000,
        test_post_nms_top_n=1000,
        fixed_batch_shape=bool(model_config.get("fixed_batch_shape", False)),
        roi_dropout=float(model_config.get("roi_dropout", 0.0)),
        retina_topk_candidates=int(model_config.get("retina_topk_candidates", args.retina_topk_candidates)),
        retina_max_detections=int(model_config.get("retina_max_detections", args.retina_max_detections)),
        yolo_topk_candidates=int(model_config.get("yolo_topk_candidates", args.yolo_topk_candidates)),
        yolo_max_detections=int(model_config.get("yolo_max_detections", args.yolo_max_detections)),
        class_loss_weights=class_loss_weights,
    ).to(device)
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
    num_boxes = sum(len(item["boxes"]) for item in predictions)
    print(f"Saved {len(predictions)} image predictions with {num_boxes} boxes to {output}")


if __name__ == "__main__":
    main()
