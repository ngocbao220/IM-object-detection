from __future__ import annotations

import argparse
import ctypes
import gc
import math
import os
import platform
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from models.faster_rcnn import BACKBONE_WEIGHTS, CUSTOM_MODEL_VERSION, create_faster_rcnn
from utils.dataset import OdDataset, build_train_transforms, collate_fn
from utils.helper import (
    get_device,
    load_checkpoint,
    move_targets_to_device,
    save_checkpoint_with_alias,
)
from utils.metric import evaluate_extended_metrics
from utils.metric import evaluate_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN with a ResNet backbone.")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--val_image_dir", required=True)
    parser.add_argument("--saved_results_dir", default="./saved_results")
    parser.add_argument("--checkpoint_dir", default=None, help="Deprecated alias for --saved_results_dir.")
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Optional checkpoint path to continue training from, e.g. saved_results/run/checkpoints/last_model.pth.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--backbone", choices=sorted(BACKBONE_WEIGHTS), default="resnet101")
    parser.add_argument(
        "--trainable_backbone_layers",
        type=int,
        default=3,
        help="Number of trainable ResNet stages. Custom model supports 0-3; torchvision supports 0-5.",
    )
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Use the repository's custom Faster R-CNN implementation. Default uses torchvision detection.",
    )
    parser.add_argument("--min_size", type=int, default=768)
    parser.add_argument("--max_size", type=int, default=1024)
    parser.add_argument(
        "--anchor_sizes",
        default="",
        help="Optional comma-separated anchor sizes, e.g. 64,128,192,256,512.",
    )
    parser.add_argument(
        "--anchor_ratios",
        default="",
        help="Optional comma-separated anchor aspect ratios, e.g. 0.33,0.5,1.0,2.0.",
    )
    parser.add_argument("--train_pre_nms_top_n", type=int, default=1000)
    parser.add_argument("--train_post_nms_top_n", type=int, default=300)
    parser.add_argument("--test_pre_nms_top_n", type=int, default=600)
    parser.add_argument("--test_post_nms_top_n", type=int, default=100)
    parser.add_argument(
        "--fixed_batch_shape",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For the custom model, pad every resized batch to max_size x max_size to reduce CUDA allocator churn.",
    )
    parser.add_argument("--roi_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lr_milestones",
        default="15,25",
        help="Comma-separated epochs at which LR is multiplied by --lr_gamma.",
    )
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument(
        "--lr_scheduler",
        choices=["multistep", "cosine", "plateau"],
        default="multistep",
        help="Learning-rate scheduler. plateau steps from validation mAP@0.5.",
    )
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--plateau_patience", type=int, default=3)
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", type=int, default=None, help="Use one CUDA GPU, e.g. --gpu 0.")
    gpu_group.add_argument("--gpus", default=None, help="Use multiple CUDA GPUs with DDP, e.g. --gpus 0,1.")
    parser.add_argument("--distributed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--pretrained_backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet weights for the selected backbone. Detection heads remain randomly initialized.",
    )
    parser.add_argument("--wandb_project", default="object-detection-final")
    parser.add_argument(
        "--wandb_run_name",
        default=None,
        help="Wandb run name and saved_results subdirectory name.",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--eval_max_images", type=int, default=0)
    parser.add_argument(
        "--full_coco_metrics_interval",
        type=int,
        default=0,
        help="Compute mAP@0.75 and mAP@0.5:0.95 every N epochs. 0 disables during training.",
    )
    parser.add_argument("--log_interval", type=int, default=20, help="Append progress to session log every N batches.")
    parser.add_argument(
        "--empty_cache_interval",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache every N training batches. 0 disables it.",
    )
    parser.add_argument(
        "--augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply conservative detection augmentations to the training dataset.",
    )
    parser.add_argument("--horizontal_flip_probability", type=float, default=0.5)
    parser.add_argument("--color_jitter_probability", type=float, default=0.3)
    parser.add_argument("--grayscale_probability", type=float, default=0.05)
    parser.add_argument(
        "--oversample_class",
        default=None,
        help="Optional class name whose images are sampled more often, e.g. chair.",
    )
    parser.add_argument("--oversample_factor", type=float, default=1.0)
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop training when validation mAP@0.5 does not improve enough.",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=7)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001)
    return parser.parse_args()


def maybe_launch_distributed(args: argparse.Namespace) -> None:
    if not args.gpus or args.distributed:
        return
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpu_ids) < 2:
        raise ValueError("--gpus requires at least two GPU ids, e.g. --gpus 0,1.")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(gpu_ids)),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--distributed",
    ]
    print(f"Launching DDP training on GPUs: {', '.join(gpu_ids)}")
    subprocess.run(command, env=env, check=True)
    raise SystemExit(0)


def parse_lr_milestones(value: str) -> list[int]:
    milestones = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(epoch <= 0 for epoch in milestones):
        raise ValueError("--lr_milestones must contain positive epoch numbers.")
    return sorted(set(milestones))


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


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    lr_milestones: list[int],
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau:
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
            eta_min=args.min_lr,
        )
    if args.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
        )
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=args.lr_gamma,
    )


def set_scheduler_resume_state(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_name: str,
    completed_epoch: int,
    base_lr: float,
    min_lr: float,
    total_epochs: int,
    milestones: list[int],
    gamma: float,
) -> None:
    """Set LR for the next epoch after resuming a checkpoint saved before scheduler.step()."""
    if scheduler_name == "plateau":
        if hasattr(scheduler, "last_epoch"):
            scheduler.last_epoch = completed_epoch
        return
    if scheduler_name == "cosine":
        progress = min(max(completed_epoch, 0), max(total_epochs, 1))
        resume_lr = min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * progress / max(total_epochs, 1))) / 2
    else:
        decay_count = sum(1 for milestone in milestones if milestone <= completed_epoch)
        resume_lr = base_lr * (gamma ** decay_count)
    for group in optimizer.param_groups:
        group["lr"] = resume_lr
    scheduler.last_epoch = completed_epoch
    scheduler._last_lr = [resume_lr for _ in optimizer.param_groups]  # noqa: SLF001


def setup_device(args: argparse.Namespace) -> tuple[torch.device, int, int]:
    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpus requires CUDA-enabled PyTorch and NVIDIA GPUs.")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}"), dist.get_rank(), dist.get_world_size()

    if args.gpu is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu requires CUDA-enabled PyTorch and an NVIDIA GPU.")
        torch.cuda.set_device(args.gpu)
        return torch.device(f"cuda:{args.gpu}"), 0, 1

    return get_device(args.device), 0, 1


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    is_main_process: bool = True,
    log_interval: int = 20,
    empty_cache_interval: int = 0,
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False, disable=not is_main_process)

    for batch_index, (images, targets) in enumerate(progress, start=1):
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(list(targets), device)

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        optimizer.step()

        batch_logs = {"loss": float(losses.detach().cpu())}
        batch_logs.update({k: float(v.detach().cpu()) for k, v in loss_dict.items()})
        for key, value in batch_logs.items():
            totals[key] = totals.get(key, 0.0) + value
        progress.set_postfix(loss=f"{batch_logs['loss']:.4f}")
        should_log = log_callback and (batch_index % log_interval == 0 or batch_index == len(loader))
        log_message = None
        if should_log:
            log_message = (
                f"Epoch {epoch:02d} train batch [{batch_index}/{len(loader)}] "
                f"loss={batch_logs['loss']:.4f} "
                f"avg_loss={totals['loss'] / batch_index:.4f}"
            )

        del images, targets, loss_dict, losses, batch_logs
        if empty_cache_interval and device.type == "cuda" and batch_index % empty_cache_interval == 0:
            torch.cuda.empty_cache()
        if log_message is not None and log_callback is not None:
            log_callback(f"{log_message}. {format_resource_usage(device)}")

    if dist.is_initialized():
        keys = sorted(totals)
        values = torch.tensor([totals[key] for key in keys] + [len(loader)], device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        total_batches = max(float(values[-1]), 1.0)
        return {key: float(values[index]) / total_batches for index, key in enumerate(keys)}
    return {key: value / max(len(loader), 1) for key, value in totals.items()}


@torch.no_grad()
def compute_validation_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_images: int = 0,
) -> dict[str, float]:
    """Compute Faster R-CNN validation losses without optimizer updates."""
    model.train()
    totals: dict[str, float] = {}
    num_batches = 0
    num_images = 0

    progress = tqdm(loader, desc="val loss", leave=False)
    for images, targets in progress:
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(list(targets), device)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        batch_logs = {"loss": float(losses.detach().cpu())}
        batch_logs.update({key: float(value.detach().cpu()) for key, value in loss_dict.items()})
        for key, value in batch_logs.items():
            totals[key] = totals.get(key, 0.0) + value

        num_batches += 1
        num_images += len(images)
        progress.set_postfix(loss=f"{batch_logs['loss']:.4f}")
        if max_images and num_images >= max_images:
            break
        del images, targets, loss_dict, losses, batch_logs

    model.eval()
    return {key: value / max(num_batches, 1) for key, value in totals.items()}


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    dataset: OdDataset,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    max_images: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    model.eval()
    predictions: dict[str, list[dict[str, Any]]] = {}
    image_offset = 0

    progress = tqdm(loader, desc="validate", leave=False)
    for images, _targets in progress:
        images_on_device = [image.to(device) for image in images]
        outputs = model(images_on_device)
        for output in outputs:
            image_info = dataset.images[image_offset]
            image_id = image_info["id"]
            image_predictions = []
            for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
                confidence = float(score.detach().cpu())
                if confidence < score_threshold:
                    continue
                label_id = int(label.detach().cpu())
                image_predictions.append(
                    {
                        "class": dataset.idx_to_class.get(label_id, str(label_id)),
                        "confidence": confidence,
                        "bbox": [float(v) for v in box.detach().cpu().tolist()],
                    }
                )
            predictions[image_id] = image_predictions
            image_offset += 1
            if max_images and image_offset >= max_images:
                del images_on_device, outputs
                return predictions
        del images_on_device, outputs
    return predictions


def ground_truth_from_dataset(dataset: OdDataset, max_images: int = 0) -> dict[str, list[dict[str, Any]]]:
    limit = max_images if max_images else len(dataset.images)
    result: dict[str, list[dict[str, Any]]] = {}
    for image in dataset.images[:limit]:
        image_id = image["id"]
        result[image_id] = [
            {"class": ann["class"], "bbox": [float(v) for v in ann["bbox"]]}
            for ann in dataset.annotations_by_image.get(image_id, [])
        ]
    return result


def append_session_log(path: Path, message: str, timestamp: bool = True) -> None:
    """Append and flush immediately so a running cloud job is observable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " if timestamp else ""
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + message.rstrip() + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_process_memory_mb() -> dict[str, float]:
    rss_mb = 0.0
    peak_rss_mb = 0.0
    try:
        with Path("/proc/self/status").open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_mb = float(line.split()[1]) / 1024
                elif line.startswith("VmHWM:"):
                    peak_rss_mb = float(line.split()[1]) / 1024
    except FileNotFoundError:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux reports KB, macOS reports bytes. The project trains on Linux,
        # but this keeps local syntax checks readable on macOS.
        peak_rss_mb = usage.ru_maxrss / (1024 if sys.platform != "darwin" else 1024**2)
    return {"rss_mb": rss_mb, "peak_rss_mb": peak_rss_mb}


def format_resource_usage(device: torch.device) -> str:
    memory = read_process_memory_mb()
    parts = [
        f"rss={memory['rss_mb']:.1f}MB" if memory["rss_mb"] else "rss=n/a",
        f"peak_rss={memory['peak_rss_mb']:.1f}MB" if memory["peak_rss_mb"] else "peak_rss=n/a",
    ]
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        parts.extend(
            [
                f"cuda_alloc={torch.cuda.memory_allocated(index) / 1024**2:.1f}MB",
                f"cuda_reserved={torch.cuda.memory_reserved(index) / 1024**2:.1f}MB",
                f"cuda_peak_alloc={torch.cuda.max_memory_allocated(index) / 1024**2:.1f}MB",
                f"cuda_free={free_bytes / 1024**2:.1f}MB",
                f"cuda_total={total_bytes / 1024**2:.1f}MB",
            ]
        )
    return "Resources: " + ", ".join(parts)


def release_epoch_memory(device: torch.device) -> None:
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def evaluate_training_metrics(
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    classes: list[str],
    full_metrics: bool,
) -> dict[str, Any]:
    if full_metrics:
        return evaluate_extended_metrics(ground_truth, predictions, classes)

    metrics = evaluate_map(ground_truth, predictions, classes, iou_threshold=0.5)
    metrics["mAP@0.5"] = metrics["mAP@0.5"]
    metrics["mAP@0.75"] = None
    metrics["mAP@0.5:0.95"] = None
    for class_metrics in metrics["per_class"].values():
        class_metrics["ap@0.5"] = class_metrics["ap"]
        class_metrics["ap@0.75"] = None
        class_metrics["ap@0.5:0.95"] = None
    return metrics


def format_metric_value(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def format_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_logs: dict[str, float],
    val_logs: dict[str, float],
    val_metrics: dict[str, Any],
    lr: float,
    elapsed_seconds: float,
) -> str:
    loss_keys = ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    lines = [
        f"Epoch [{epoch:02d}/{total_epochs:02d}]",
        f"├── Train Loss : {train_logs.get('loss', 0.0):.4f}",
    ]
    for index, key in enumerate(loss_keys):
        branch = "└──" if index == len(loss_keys) - 1 else "├──"
        lines.append(f"│   {branch} {key:<17}: {train_logs.get(key, 0.0):.4f}")

    lines.extend(
        [
            f"├── Val Loss   : {val_logs.get('loss', 0.0):.4f}",
            f"├── mAP@0.5    : {val_metrics['mAP@0.5']:.4f}",
            f"├── mAP@0.75   : {format_metric_value(val_metrics.get('mAP@0.75'))}",
            f"├── mAP@0.5:0.95 : {format_metric_value(val_metrics.get('mAP@0.5:0.95'))}",
            f"├── Precision  : {val_metrics['micro_precision']:.4f}",
            f"├── Recall     : {val_metrics['micro_recall']:.4f}",
            f"├── GT Boxes   : {val_metrics['num_ground_truth_boxes']}",
            f"├── Predictions: {val_metrics['num_predictions']}",
            "├── Per-class AP",
        ]
    )
    per_class = val_metrics["per_class"]
    for index, (class_name, metrics) in enumerate(per_class.items()):
        branch = "└──" if index == len(per_class) - 1 else "├──"
        lines.append(
            f"│   {branch} {class_name:<8}: AP50={format_metric_value(metrics.get('ap@0.5'))}, "
            f"AP75={format_metric_value(metrics.get('ap@0.75'))}, "
            f"AP50:95={format_metric_value(metrics.get('ap@0.5:0.95'))}, "
            f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}"
        )
    lines.extend([f"├── LR         : {lr:.6f}", f"└── Time       : {elapsed_seconds:.1f}s"])
    return "\n".join(lines)


def count_dataset_boxes(dataset: OdDataset) -> dict[str, Any]:
    class_counts = Counter()
    boxes_per_image = []
    for image in dataset.images:
        image_id = image["id"]
        anns = dataset.annotations_by_image.get(image_id, [])
        boxes_per_image.append(len(anns))
        class_counts.update(ann["class"] for ann in anns)

    return {
        "num_images": len(dataset),
        "num_boxes": sum(boxes_per_image),
        "images_without_boxes": sum(1 for value in boxes_per_image if value == 0),
        "max_boxes_per_image": max(boxes_per_image, default=0),
        "class_counts": dict(class_counts),
    }


def build_class_oversampling_sampler(
    dataset: OdDataset,
    class_name: str | None,
    factor: float,
) -> tuple[WeightedRandomSampler | None, dict[str, Any]]:
    if not class_name:
        return None, {"enabled": False}
    if class_name not in dataset.classes:
        raise ValueError(f"--oversample_class must be one of {dataset.classes}.")
    if factor < 1.0:
        raise ValueError("--oversample_factor must be greater than or equal to 1.0.")

    weights = []
    num_target_images = 0
    for image in dataset.images:
        image_id = image["id"]
        has_target_class = any(
            ann["class"] == class_name for ann in dataset.annotations_by_image.get(image_id, [])
        )
        if has_target_class:
            num_target_images += 1
        weights.append(factor if has_target_class else 1.0)

    if num_target_images == 0:
        raise ValueError(f"No training images contain oversample class: {class_name}.")

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    return sampler, {
        "enabled": True,
        "class": class_name,
        "factor": factor,
        "target_images": num_target_images,
        "total_images": len(dataset),
        "target_image_ratio": num_target_images / max(len(dataset), 1),
    }


def get_device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "selected_device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if device.type == "cuda":
        index = device.index or torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_total_memory_gb": round(props.total_memory / (1024**3), 2),
            }
        )
    return info


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def format_session_info(info: dict[str, Any]) -> str:
    lines = [
        "========== Training Session ==========",
        f"Started: {info['started_at']}",
        f"Run name: {info['run_name']}",
        f"Device: {info['device']['selected_device']}",
    ]
    if info["distributed"]["world_size"] > 1:
        lines.append(
            f"Distributed: DDP with {info['distributed']['world_size']} processes "
            f"on GPUs {info['distributed']['gpus']}"
        )
    if "cuda_device_name" in info["device"]:
        lines.append(
            "CUDA: "
            f"{info['device']['cuda_device_name']} "
            f"({info['device']['cuda_total_memory_gb']} GB)"
        )
    lines.append(f"Torch: {info['device']['torch_version']}")
    lines.append(f"Classes: {', '.join(info['classes'])}")
    lines.append(
        "Train dataset: "
        f"{info['dataset']['train']['num_images']} images, "
        f"{info['dataset']['train']['num_boxes']} boxes, "
        f"{info['dataset']['train']['images_without_boxes']} empty images"
    )
    lines.append(
        "Val dataset: "
        f"{info['dataset']['val']['num_images']} images, "
        f"{info['dataset']['val']['num_boxes']} boxes, "
        f"{info['dataset']['val']['images_without_boxes']} empty images"
    )
    lines.append(f"Train class counts: {info['dataset']['train']['class_counts']}")
    lines.append(f"Val class counts: {info['dataset']['val']['class_counts']}")
    if info["resume"]["enabled"]:
        lines.append(
            "Resume: "
            f"checkpoint={info['resume']['checkpoint']}, "
            f"checkpoint_epoch={info['resume']['checkpoint_epoch']}, "
            f"start_epoch={info['resume']['start_epoch']}"
        )
    else:
        lines.append("Resume: disabled")
    if info["oversampling"]["enabled"]:
        lines.append(
            "Oversampling: "
            f"class={info['oversampling']['class']}, "
            f"factor={info['oversampling']['factor']}, "
            f"target_images={info['oversampling']['target_images']}/"
            f"{info['oversampling']['total_images']}"
        )
    else:
        lines.append("Oversampling: disabled")
    lines.append(
        f"Model: {info['model']['implementation']} Faster R-CNN {info['model']['backbone']} "
        f"({info['model']['trainable_parameters']:,}/"
        f"{info['model']['total_parameters']:,} trainable/total params)"
    )
    hp = info["hyperparameters"]
    lines.extend(
        [
            "Hyperparameters",
            f"├── Training",
            f"│   ├── epochs             : {hp['epochs']}",
            f"│   ├── batch_size         : {hp['batch_size']}",
            f"│   ├── num_workers        : {hp['num_workers']}",
            f"│   ├── augmentation       : {hp['augmentation']}",
            f"│   ├── oversample_class   : {hp['oversample_class'] or 'disabled'}",
            f"│   └── oversample_factor  : {hp['oversample_factor']}",
            f"├── Optimizer",
            f"│   ├── lr                 : {hp['lr']}",
            f"│   ├── momentum           : {hp['momentum']}",
            f"│   └── weight_decay       : {hp['weight_decay']}",
            f"├── LR Scheduler",
            f"│   ├── type               : {hp['lr_scheduler']}",
            f"│   ├── milestones         : {hp['lr_milestones']}",
            f"│   ├── gamma              : {hp['lr_gamma']}",
            f"│   ├── min_lr             : {hp['min_lr']}",
            f"│   ├── plateau_patience   : {hp['plateau_patience']}",
            f"│   └── plateau_factor     : {hp['plateau_factor']}",
            f"├── Model",
            f"│   ├── implementation     : {'Custom' if hp['custom_model'] else 'Torchvision'}",
            f"│   ├── backbone           : {hp['backbone']}",
            f"│   ├── trainable_layers   : {hp['trainable_backbone_layers']}",
            f"│   ├── pretrained_backbone: {hp['pretrained_backbone']}",
            f"│   ├── roi_dropout        : {hp['roi_dropout']}",
            f"│   ├── min_size           : {hp['min_size']}",
            f"│   ├── max_size           : {hp['max_size']}",
            f"│   ├── anchor_sizes       : {hp['anchor_sizes'] or 'model_default'}",
            f"│   ├── anchor_ratios      : {hp['anchor_ratios'] or 'model_default'}",
            f"│   ├── train_pre_nms_top_n : {hp['train_pre_nms_top_n']}",
            f"│   ├── train_post_nms_top_n: {hp['train_post_nms_top_n']}",
            f"│   ├── test_pre_nms_top_n  : {hp['test_pre_nms_top_n']}",
            f"│   ├── test_post_nms_top_n : {hp['test_post_nms_top_n']}",
            f"│   └── fixed_batch_shape   : {hp['fixed_batch_shape']}",
            f"├── Validation",
            f"│   ├── score_threshold    : {hp['score_threshold']}",
            f"│   ├── eval_max_images    : {hp['eval_max_images'] or 'all'}",
            f"│   └── full_coco_interval : {hp['full_coco_metrics_interval'] or 'disabled'}",
            f"├── Early Stopping",
            f"│   ├── enabled            : {hp['early_stopping']}",
            f"│   ├── patience           : {hp['early_stopping_patience']}",
            f"│   └── min_delta          : {hp['early_stopping_min_delta']}",
            f"└── Logging",
            f"    ├── log_interval       : {hp['log_interval']}",
            f"    ├── empty_cache_interval: {hp['empty_cache_interval'] or 'disabled'}",
            f"    └── use_wandb          : {hp['use_wandb']}",
        ]
    )
    lines.append(f"Saved results dir: {info['paths']['saved_results_dir']}")
    lines.append(f"Checkpoint dir: {info['paths']['checkpoint_dir']}")
    lines.append(f"Best checkpoint: {info['paths']['best_checkpoint']}")
    lines.append(f"Last checkpoint: {info['paths']['last_checkpoint']}")
    lines.append(f"Log dir: {info['paths']['log_dir']}")
    lines.append("=====================================")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.log_interval <= 0:
        raise ValueError("--log_interval must be greater than 0.")
    if args.early_stopping_patience <= 0:
        raise ValueError("--early_stopping_patience must be greater than 0.")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early_stopping_min_delta must be greater than or equal to 0.")
    if args.min_size <= 0 or args.max_size <= 0 or args.min_size > args.max_size:
        raise ValueError("--min_size and --max_size must be positive with min_size <= max_size.")
    if args.min_lr < 0 or args.min_lr > args.lr:
        raise ValueError("--min_lr must be between 0 and --lr.")
    if args.plateau_patience <= 0:
        raise ValueError("--plateau_patience must be greater than 0.")
    if args.plateau_factor <= 0 or args.plateau_factor >= 1:
        raise ValueError("--plateau_factor must be between 0 and 1.")
    if args.full_coco_metrics_interval < 0:
        raise ValueError("--full_coco_metrics_interval must be greater than or equal to 0.")
    if args.empty_cache_interval < 0:
        raise ValueError("--empty_cache_interval must be greater than or equal to 0.")
    lr_milestones = parse_lr_milestones(args.lr_milestones)
    anchor_sizes = parse_optional_int_tuple(args.anchor_sizes)
    anchor_ratios = parse_optional_float_tuple(args.anchor_ratios)
    if anchor_sizes is not None and not anchor_sizes:
        raise ValueError("--anchor_sizes must contain at least one value.")
    if not args.custom and anchor_sizes is not None and len(anchor_sizes) != 5:
        raise ValueError("--anchor_sizes must contain exactly 5 values when using torchvision Faster R-CNN.")
    if args.custom and not 0 <= args.trainable_backbone_layers <= 3:
        raise ValueError("--trainable_backbone_layers must be between 0 and 3 for the custom model.")
    if not args.custom and not 0 <= args.trainable_backbone_layers <= 5:
        raise ValueError("--trainable_backbone_layers must be between 0 and 5 for torchvision.")
    if args.roi_dropout < 0 or args.roi_dropout >= 1:
        raise ValueError("--roi_dropout must be in [0, 1).")
    custom_top_n_values = [
        args.train_pre_nms_top_n,
        args.train_post_nms_top_n,
        args.test_pre_nms_top_n,
        args.test_post_nms_top_n,
    ]
    if any(value <= 0 for value in custom_top_n_values):
        raise ValueError("Custom proposal top-N values must be positive.")
    probabilities = [
        args.horizontal_flip_probability,
        args.color_jitter_probability,
        args.grayscale_probability,
    ]
    if any(value < 0 or value > 1 for value in probabilities):
        raise ValueError("Augmentation probabilities must be between 0 and 1.")
    if args.oversample_factor < 1.0:
        raise ValueError("--oversample_factor must be greater than or equal to 1.0.")
    maybe_launch_distributed(args)
    device, rank, world_size = setup_device(args)
    is_main_process = rank == 0

    started = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.wandb_run_name or f"session-{started}"
    if run_name in {".", ".."} or Path(run_name).name != run_name:
        raise ValueError("--wandb_run_name must be a single folder-safe name.")
    saved_results_root = Path(args.checkpoint_dir or args.saved_results_dir)
    saved_results_dir = saved_results_root / run_name
    checkpoint_dir = saved_results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = saved_results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    train_transforms = (
        build_train_transforms(
            horizontal_flip_probability=args.horizontal_flip_probability,
            color_jitter_probability=args.color_jitter_probability,
            grayscale_probability=args.grayscale_probability,
        )
        if args.augmentation
        else None
    )
    train_dataset = OdDataset(args.train_data, args.image_dir, transforms=train_transforms)
    val_dataset = OdDataset(args.val_data, args.val_image_dir, classes=train_dataset.classes)
    oversampling_sampler, oversampling_info = build_class_oversampling_sampler(
        train_dataset,
        args.oversample_class,
        args.oversample_factor,
    )
    if world_size > 1 and oversampling_sampler is not None:
        raise RuntimeError("--oversample_class is currently supported only for single-GPU training.")
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)
        if world_size > 1
        else oversampling_sampler
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = create_faster_rcnn(
        num_classes=len(train_dataset.classes) + 1,
        backbone_name=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
        trainable_backbone_layers=args.trainable_backbone_layers,
        min_size=args.min_size,
        max_size=args.max_size,
        anchor_sizes=anchor_sizes,
        anchor_ratios=anchor_ratios,
        custom=args.custom,
        train_pre_nms_top_n=args.train_pre_nms_top_n,
        train_post_nms_top_n=args.train_post_nms_top_n,
        test_pre_nms_top_n=args.test_pre_nms_top_n,
        test_post_nms_top_n=args.test_post_nms_top_n,
        fixed_batch_shape=args.fixed_batch_shape,
        roi_dropout=args.roi_dropout,
    ).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index])

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, args, lr_milestones)
    resume_checkpoint = None
    resume_epoch = 0
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume_from checkpoint does not exist: {resume_path}")
        resume_checkpoint = load_checkpoint(resume_path, unwrap_model(model), device, optimizer)
        resume_epoch = int(resume_checkpoint.get("epoch", 0))
        set_scheduler_resume_state(
            optimizer,
            scheduler,
            args.lr_scheduler,
            resume_epoch,
            args.lr,
            args.min_lr,
            args.epochs,
            lr_milestones,
            args.lr_gamma,
        )

    session_info = {
        "started_at": started,
        "run_name": run_name,
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "dataset": {
            "train": count_dataset_boxes(train_dataset),
            "val": count_dataset_boxes(val_dataset),
        },
        "resume": {
            "enabled": resume_checkpoint is not None,
            "checkpoint": str(Path(args.resume_from)) if args.resume_from else None,
            "checkpoint_epoch": resume_epoch if resume_checkpoint is not None else None,
            "start_epoch": resume_epoch + 1 if resume_checkpoint is not None else 1,
        },
        "oversampling": oversampling_info,
        "device": get_device_info(device),
        "distributed": {"world_size": world_size, "rank": rank, "gpus": args.gpus},
        "model": {
            "backbone": args.backbone,
            "implementation": "Custom" if args.custom else "Torchvision",
            **count_parameters(model),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "lr_milestones": lr_milestones,
            "lr_gamma": args.lr_gamma,
            "min_lr": args.min_lr,
            "plateau_patience": args.plateau_patience,
            "plateau_factor": args.plateau_factor,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "score_threshold": args.score_threshold,
            "backbone": args.backbone,
            "trainable_backbone_layers": args.trainable_backbone_layers,
            "custom_model": args.custom,
            "roi_dropout": args.roi_dropout,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "anchor_sizes": anchor_sizes,
            "anchor_ratios": anchor_ratios,
            "train_pre_nms_top_n": args.train_pre_nms_top_n,
            "train_post_nms_top_n": args.train_post_nms_top_n,
            "test_pre_nms_top_n": args.test_pre_nms_top_n,
            "test_post_nms_top_n": args.test_post_nms_top_n,
            "fixed_batch_shape": args.fixed_batch_shape,
            "eval_max_images": args.eval_max_images,
            "full_coco_metrics_interval": args.full_coco_metrics_interval,
            "log_interval": args.log_interval,
            "empty_cache_interval": args.empty_cache_interval,
            "use_wandb": args.use_wandb,
            "pretrained_backbone": args.pretrained_backbone,
            "augmentation": args.augmentation,
            "horizontal_flip_probability": args.horizontal_flip_probability,
            "color_jitter_probability": args.color_jitter_probability,
            "grayscale_probability": args.grayscale_probability,
            "oversample_class": args.oversample_class,
            "oversample_factor": args.oversample_factor,
            "early_stopping": args.early_stopping,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
        },
        "paths": {
            "train_data": str(Path(args.train_data)),
            "val_data": str(Path(args.val_data)),
            "image_dir": str(Path(args.image_dir)),
            "val_image_dir": str(Path(args.val_image_dir)),
            "saved_results_dir": str(saved_results_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "log_dir": str(log_dir),
            "last_checkpoint": str(checkpoint_dir / f"last_model-{started}.pth"),
            "best_checkpoint": str(checkpoint_dir / f"best_model-{started}.pth"),
        },
    }
    text_log_path = log_dir / "session.log"
    if is_main_process:
        session_header = format_session_info(session_info)
        print(f"\n{session_header}\n")
        append_session_log(text_log_path, session_header, timestamp=False)
        append_session_log(text_log_path, "Training session initialized.")

    run = None
    if args.use_wandb and is_main_process:
        if wandb is None:
            print("wandb is not installed; continuing without wandb.")
        else:
            run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args) | session_info,
            )

    last_checkpoint_path = checkpoint_dir / f"last_model-{started}.pth"
    best_checkpoint_path = checkpoint_dir / f"best_model-{started}.pth"
    last_checkpoint_alias = checkpoint_dir / "last_model.pth"
    best_checkpoint_alias = checkpoint_dir / "best_model.pth"
    best_map = -1.0
    if resume_checkpoint is not None:
        best_map = float(
            resume_checkpoint.get("metrics", {}).get(
                "mAP@0.5",
                resume_checkpoint.get("metrics", {}).get("val/mAP@0.5", -1.0),
            )
        )
        if best_checkpoint_alias.exists():
            best_checkpoint = torch.load(best_checkpoint_alias, map_location="cpu")
            best_map = max(
                best_map,
                float(
                    best_checkpoint.get("metrics", {}).get(
                        "mAP@0.5",
                        best_checkpoint.get("metrics", {}).get("val/mAP@0.5", -1.0),
                    )
                ),
            )
    epochs_without_improvement = 0
    model_config = {
        "backbone": args.backbone,
        "custom_model": args.custom,
        "custom_model_version": CUSTOM_MODEL_VERSION if args.custom else None,
        "trainable_backbone_layers": args.trainable_backbone_layers,
        "roi_dropout": args.roi_dropout,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "anchor_sizes": anchor_sizes,
        "anchor_ratios": anchor_ratios,
        "fixed_batch_shape": args.fixed_batch_shape,
    }

    for epoch in range(resume_epoch + 1, args.epochs + 1):
        should_stop = False
        scheduler_metric = -1.0
        epoch_started = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        if is_main_process:
            append_session_log(
                text_log_path,
                f"Epoch [{epoch:02d}/{args.epochs:02d}] started. {format_resource_usage(device)}",
            )
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        train_logs = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            is_main_process,
            log_interval=args.log_interval,
            empty_cache_interval=args.empty_cache_interval,
            log_callback=(
                lambda message: append_session_log(text_log_path, message)
                if is_main_process
                else None
            ),
        )

        if is_main_process:
            append_session_log(
                text_log_path,
                f"Epoch {epoch:02d} training completed. "
                f"{format_resource_usage(device)}",
            )
            release_epoch_memory(device)
            append_session_log(
                text_log_path,
                f"Epoch {epoch:02d} cache released before validation loss. "
                f"{format_resource_usage(device)}",
            )
            val_logs = compute_validation_loss(
                unwrap_model(model),
                val_loader,
                device,
                max_images=args.eval_max_images,
            )
            append_session_log(
                text_log_path,
                f"Epoch {epoch:02d} validation loss completed: {val_logs.get('loss', 0.0):.4f}. "
                f"{format_resource_usage(device)}",
            )
            release_epoch_memory(device)
            append_session_log(
                text_log_path,
                f"Epoch {epoch:02d} cache released before detection metrics. "
                f"{format_resource_usage(device)}",
            )
            val_predictions = predict_dataset(
                unwrap_model(model),
                val_dataset,
                val_loader,
                device,
                score_threshold=args.score_threshold,
                max_images=args.eval_max_images,
            )
            release_epoch_memory(device)
            append_session_log(
                text_log_path,
                f"Epoch {epoch:02d} prediction tensors moved to CPU. Computing metrics. "
                f"{format_resource_usage(device)}",
            )
            val_gt = ground_truth_from_dataset(val_dataset, max_images=args.eval_max_images)
            full_metrics = bool(
                args.full_coco_metrics_interval
                and (epoch % args.full_coco_metrics_interval == 0 or epoch == args.epochs)
            )
            val_metrics = evaluate_training_metrics(
                val_gt,
                val_predictions,
                val_dataset.classes,
                full_metrics=full_metrics,
            )
            metric_mode = "full COCO metrics" if full_metrics else "mAP@0.5 only"
            append_session_log(
                text_log_path,
                f"Epoch {epoch:02d} metrics computed ({metric_mode}). Saving artifacts. "
                f"{format_resource_usage(device)}",
            )
            epoch_seconds = time.perf_counter() - epoch_started

            row = {
                "epoch": epoch,
                "lr": current_lr,
                "time_seconds": epoch_seconds,
                **{f"train/{k}": v for k, v in train_logs.items()},
                **{f"val/{k}": v for k, v in val_logs.items()},
                "val/mAP@0.5": val_metrics["mAP@0.5"],
                "val/micro_precision": val_metrics["micro_precision"],
                "val/micro_recall": val_metrics["micro_recall"],
            }
            if val_metrics.get("mAP@0.75") is not None:
                row["val/mAP@0.75"] = val_metrics["mAP@0.75"]
            if val_metrics.get("mAP@0.5:0.95") is not None:
                row["val/mAP@0.5:0.95"] = val_metrics["mAP@0.5:0.95"]
            if run is not None:
                wandb.log(row, step=epoch)

            epoch_metrics = {"epoch": epoch, "train": train_logs, "val": val_logs, **val_metrics}
            save_checkpoint_with_alias(
                last_checkpoint_path,
                last_checkpoint_alias,
                unwrap_model(model),
                optimizer,
                epoch,
                train_dataset.classes,
                epoch_metrics,
                model_config,
            )
            current_map = val_metrics["mAP@0.5"]
            scheduler_metric = current_map
            if current_map > best_map + args.early_stopping_min_delta:
                best_map = val_metrics["mAP@0.5"]
                epochs_without_improvement = 0
                save_checkpoint_with_alias(
                    best_checkpoint_path,
                    best_checkpoint_alias,
                    unwrap_model(model),
                    optimizer,
                    epoch,
                    train_dataset.classes,
                    epoch_metrics,
                    model_config,
                )
                append_session_log(
                    text_log_path,
                    f"Epoch {epoch:02d} improved mAP@0.5 to {best_map:.4f}. "
                    f"Saved best checkpoint: {best_checkpoint_path.name}. "
                    f"{format_resource_usage(device)}",
                )
            else:
                epochs_without_improvement += 1
                append_session_log(
                    text_log_path,
                    f"Epoch {epoch:02d} did not improve mAP@0.5 enough. "
                    f"Early stopping counter: {epochs_without_improvement}/"
                    f"{args.early_stopping_patience}.",
                )

            epoch_summary = format_epoch_summary(
                epoch,
                args.epochs,
                train_logs,
                val_logs,
                val_metrics,
                current_lr,
                epoch_seconds,
            )
            print(f"\n{epoch_summary}\n")
            append_session_log(text_log_path, epoch_summary, timestamp=False)
            should_stop = (
                args.early_stopping
                and epochs_without_improvement >= args.early_stopping_patience
            )
            if should_stop:
                append_session_log(
                    text_log_path,
                    f"Early stopping triggered at epoch {epoch:02d}. "
                    f"Best mAP@0.5={best_map:.4f}.",
                )
            del val_predictions, val_gt, val_metrics, val_logs, epoch_metrics, row
            release_epoch_memory(device)
            append_session_log(
                text_log_path,
                f"Epoch [{epoch:02d}/{args.epochs:02d}] completed. "
                f"{format_resource_usage(device)}\n",
            )
        if dist.is_initialized():
            metric_tensor = torch.tensor([scheduler_metric], device=device)
            dist.broadcast(metric_tensor, src=0)
            scheduler_metric = float(metric_tensor.item())
        if args.lr_scheduler == "plateau":
            scheduler.step(scheduler_metric)
        else:
            scheduler.step()
        if not is_main_process:
            release_epoch_memory(device)
        if dist.is_initialized():
            stop_tensor = torch.tensor([int(should_stop)], device=device)
            dist.broadcast(stop_tensor, src=0)
            dist.barrier()
            should_stop = bool(stop_tensor.item())
        if should_stop:
            break

    if run is not None:
        run.finish()
    if is_main_process:
        append_session_log(text_log_path, "Training session completed.")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
