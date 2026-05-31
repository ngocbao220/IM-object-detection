from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from models.faster_rcnn import create_faster_rcnn_resnet50
from models.modules import get_device, move_targets_to_device, save_checkpoint_with_alias
from utils.dataset import OdDataset, build_train_transforms, collate_fn
from utils.helper import save_json
from utils.metric import evaluate_extended_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN ResNet-50.")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--val_image_dir", required=True)
    parser.add_argument("--saved_results_dir", default="./saved_results")
    parser.add_argument("--checkpoint_dir", default=None, help="Deprecated alias for --saved_results_dir.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument(
        "--lr_milestones",
        default="15,25",
        help="Comma-separated epochs at which LR is multiplied by --lr_gamma.",
    )
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", type=int, default=None, help="Use one CUDA GPU, e.g. --gpu 0.")
    gpu_group.add_argument("--gpus", default=None, help="Use multiple CUDA GPUs with DDP, e.g. --gpus 0,1.")
    parser.add_argument("--distributed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--pretrained_backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ResNet-50 ImageNet weights. Faster R-CNN detection heads remain randomly initialized.",
    )
    parser.add_argument("--wandb_project", default="object-detection-final")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--eval_max_images", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=20, help="Append progress to session log every N batches.")
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
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False, disable=not is_main_process)

    for batch_index, (images, targets) in enumerate(progress, start=1):
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(list(targets), device)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()

        batch_logs = {"loss": float(losses.detach().cpu())}
        batch_logs.update({k: float(v.detach().cpu()) for k, v in loss_dict.items()})
        for key, value in batch_logs.items():
            totals[key] = totals.get(key, 0.0) + value
        progress.set_postfix(loss=f"{batch_logs['loss']:.4f}")
        if log_callback and (batch_index % log_interval == 0 or batch_index == len(loader)):
            log_callback(
                f"Epoch {epoch:02d} train batch [{batch_index}/{len(loader)}] "
                f"loss={batch_logs['loss']:.4f} "
                f"avg_loss={totals['loss'] / batch_index:.4f}"
            )

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
                return predictions
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


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_session_log(path: Path, message: str, timestamp: bool = True) -> None:
    """Append and flush immediately so a running cloud job is observable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " if timestamp else ""
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + message.rstrip() + "\n")
        f.flush()
        os.fsync(f.fileno())


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
            f"├── mAP@0.75   : {val_metrics['mAP@0.75']:.4f}",
            f"├── mAP@0.5:0.95 : {val_metrics['mAP@0.5:0.95']:.4f}",
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
            f"│   {branch} {class_name:<8}: AP50={metrics['ap@0.5']:.4f}, "
            f"AP75={metrics['ap@0.75']:.4f}, AP50:95={metrics['ap@0.5:0.95']:.4f}, "
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
    lines.append(
        "Model: Faster R-CNN ResNet-50 FPN "
        f"({info['model']['trainable_parameters']:,}/"
        f"{info['model']['total_parameters']:,} trainable/total params)"
    )
    lines.append(
        "Hyperparams: "
        f"epochs={info['hyperparameters']['epochs']}, "
        f"batch_size={info['hyperparameters']['batch_size']}, "
        f"lr={info['hyperparameters']['lr']}, "
        f"lr_milestones={info['hyperparameters']['lr_milestones']}, "
        f"lr_gamma={info['hyperparameters']['lr_gamma']}, "
        f"score_threshold={info['hyperparameters']['score_threshold']}, "
        f"augmentation={info['hyperparameters']['augmentation']}, "
        f"early_stopping={info['hyperparameters']['early_stopping']}, "
        f"early_stopping_patience={info['hyperparameters']['early_stopping_patience']}, "
        f"early_stopping_min_delta={info['hyperparameters']['early_stopping_min_delta']}, "
        f"num_workers={info['hyperparameters']['num_workers']}, "
        f"log_interval={info['hyperparameters']['log_interval']}"
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
    lr_milestones = parse_lr_milestones(args.lr_milestones)
    probabilities = [
        args.horizontal_flip_probability,
        args.color_jitter_probability,
        args.grayscale_probability,
    ]
    if any(value < 0 or value > 1 for value in probabilities):
        raise ValueError("Augmentation probabilities must be between 0 and 1.")
    maybe_launch_distributed(args)
    device, rank, world_size = setup_device(args)
    is_main_process = rank == 0

    saved_results_dir = Path(args.checkpoint_dir or args.saved_results_dir)
    checkpoint_dir = saved_results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = saved_results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = saved_results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

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
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
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

    model = create_faster_rcnn_resnet50(
        num_classes=len(train_dataset.classes) + 1,
        pretrained_backbone=args.pretrained_backbone,
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=args.lr_gamma,
    )

    started = time.strftime("%Y%m%d-%H%M%S")
    session_info = {
        "started_at": started,
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "dataset": {
            "train": count_dataset_boxes(train_dataset),
            "val": count_dataset_boxes(val_dataset),
        },
        "device": get_device_info(device),
        "distributed": {"world_size": world_size, "rank": rank, "gpus": args.gpus},
        "model": count_parameters(model),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "lr_milestones": lr_milestones,
            "lr_gamma": args.lr_gamma,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "score_threshold": args.score_threshold,
            "eval_max_images": args.eval_max_images,
            "log_interval": args.log_interval,
            "pretrained_backbone": args.pretrained_backbone,
            "augmentation": args.augmentation,
            "horizontal_flip_probability": args.horizontal_flip_probability,
            "color_jitter_probability": args.color_jitter_probability,
            "grayscale_probability": args.grayscale_probability,
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
            "metrics_dir": str(metrics_dir),
            "last_checkpoint": str(checkpoint_dir / f"last_model-{started}.pth"),
            "best_checkpoint": str(checkpoint_dir / f"best_model-{started}.pth"),
        },
    }
    session_info_path = log_dir / f"session-{started}.json"
    text_log_path = log_dir / f"session-{started}.log"
    if is_main_process:
        save_json(session_info, session_info_path)
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
                name=args.wandb_run_name,
                config=vars(args) | session_info,
            )

    best_map = -1.0
    epochs_without_improvement = 0
    jsonl_path = log_dir / f"train-{started}.jsonl"
    csv_path = log_dir / f"train-{started}.csv"
    last_checkpoint_path = checkpoint_dir / f"last_model-{started}.pth"
    best_checkpoint_path = checkpoint_dir / f"best_model-{started}.pth"
    last_checkpoint_alias = checkpoint_dir / "last_model.pth"
    best_checkpoint_alias = checkpoint_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        should_stop = False
        epoch_started = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        if is_main_process:
            append_session_log(text_log_path, f"Epoch [{epoch:02d}/{args.epochs:02d}] started.")
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_logs = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            is_main_process,
            log_interval=args.log_interval,
            log_callback=(
                lambda message: append_session_log(text_log_path, message)
                if is_main_process
                else None
            ),
        )

        if is_main_process:
            append_session_log(text_log_path, f"Epoch {epoch:02d} training completed. Computing validation loss.")
            val_logs = compute_validation_loss(
                unwrap_model(model),
                val_loader,
                device,
                max_images=args.eval_max_images,
            )
            append_session_log(
                text_log_path,
                f"Epoch {epoch:02d} validation loss completed: {val_logs.get('loss', 0.0):.4f}. "
                "Computing detection metrics.",
            )
            val_predictions = predict_dataset(
                unwrap_model(model),
                val_dataset,
                val_loader,
                device,
                score_threshold=args.score_threshold,
                max_images=args.eval_max_images,
            )
            val_gt = ground_truth_from_dataset(val_dataset, max_images=args.eval_max_images)
            val_metrics = evaluate_extended_metrics(val_gt, val_predictions, val_dataset.classes)
            append_session_log(text_log_path, f"Epoch {epoch:02d} metrics computed. Saving artifacts.")
            epoch_seconds = time.perf_counter() - epoch_started

            row = {
                "epoch": epoch,
                "lr": current_lr,
                "time_seconds": epoch_seconds,
                **{f"train/{k}": v for k, v in train_logs.items()},
                **{f"val/{k}": v for k, v in val_logs.items()},
                "val/mAP@0.5": val_metrics["mAP@0.5"],
                "val/mAP@0.75": val_metrics["mAP@0.75"],
                "val/mAP@0.5:0.95": val_metrics["mAP@0.5:0.95"],
                "val/micro_precision": val_metrics["micro_precision"],
                "val/micro_recall": val_metrics["micro_recall"],
            }
            append_csv(csv_path, row)
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if run is not None:
                wandb.log(row, step=epoch)

            epoch_metrics = {"epoch": epoch, "train": train_logs, "val": val_logs, **val_metrics}
            save_json(epoch_metrics, metrics_dir / f"epoch_{epoch:03d}.json")
            save_checkpoint_with_alias(
                last_checkpoint_path,
                last_checkpoint_alias,
                unwrap_model(model),
                optimizer,
                epoch,
                train_dataset.classes,
                epoch_metrics,
            )
            current_map = val_metrics["mAP@0.5"]
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
                )
                append_session_log(
                    text_log_path,
                    f"Epoch {epoch:02d} improved mAP@0.5 to {best_map:.4f}. "
                    f"Saved best checkpoint: {best_checkpoint_path.name}.",
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
            append_session_log(text_log_path, f"Epoch [{epoch:02d}/{args.epochs:02d}] completed.\n")
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
        scheduler.step()
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
