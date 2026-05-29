from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from models.faster_rcnn import create_faster_rcnn_resnet50
from models.modules import get_device, move_targets_to_device, save_checkpoint
from utils.dataset import OdDataset, collate_fn
from utils.helper import save_json
from utils.metric import evaluate_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN ResNet-50.")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--val_image_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pretrained_backbone", action="store_true")
    parser.add_argument("--pretrained_coco", action="store_true")
    parser.add_argument("--wandb_project", default="object-detection-final")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--eval_max_images", type=int, default=0)
    return parser.parse_args()


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)

    for images, targets in progress:
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

    return {key: value / max(len(loader), 1) for key, value in totals.items()}


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


def print_session_info(info: dict[str, Any]) -> None:
    print("\n========== Training Session ==========")
    print(f"Started: {info['started_at']}")
    print(f"Device: {info['device']['selected_device']}")
    if "cuda_device_name" in info["device"]:
        print(
            "CUDA: "
            f"{info['device']['cuda_device_name']} "
            f"({info['device']['cuda_total_memory_gb']} GB)"
        )
    print(f"Torch: {info['device']['torch_version']}")
    print(f"Classes: {', '.join(info['classes'])}")
    print(
        "Train dataset: "
        f"{info['dataset']['train']['num_images']} images, "
        f"{info['dataset']['train']['num_boxes']} boxes, "
        f"{info['dataset']['train']['images_without_boxes']} empty images"
    )
    print(
        "Val dataset: "
        f"{info['dataset']['val']['num_images']} images, "
        f"{info['dataset']['val']['num_boxes']} boxes, "
        f"{info['dataset']['val']['images_without_boxes']} empty images"
    )
    print(f"Train class counts: {info['dataset']['train']['class_counts']}")
    print(f"Val class counts: {info['dataset']['val']['class_counts']}")
    print(
        "Model: Faster R-CNN ResNet-50 FPN "
        f"({info['model']['trainable_parameters']:,}/"
        f"{info['model']['total_parameters']:,} trainable/total params)"
    )
    print(
        "Hyperparams: "
        f"epochs={info['hyperparameters']['epochs']}, "
        f"batch_size={info['hyperparameters']['batch_size']}, "
        f"lr={info['hyperparameters']['lr']}, "
        f"num_workers={info['hyperparameters']['num_workers']}"
    )
    print(f"Checkpoint dir: {info['paths']['checkpoint_dir']}")
    print(f"Log dir: {info['paths']['log_dir']}")
    print("=====================================\n")


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = checkpoint_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = OdDataset(args.train_data, args.image_dir)
    val_dataset = OdDataset(args.val_data, args.val_image_dir, classes=train_dataset.classes)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    device = get_device(args.device)
    model = create_faster_rcnn_resnet50(
        num_classes=len(train_dataset.classes) + 1,
        pretrained_backbone=args.pretrained_backbone,
        pretrained_coco=args.pretrained_coco,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

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
        "model": count_parameters(model),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "score_threshold": args.score_threshold,
            "eval_max_images": args.eval_max_images,
            "pretrained_backbone": args.pretrained_backbone,
            "pretrained_coco": args.pretrained_coco,
        },
        "paths": {
            "train_data": str(Path(args.train_data)),
            "val_data": str(Path(args.val_data)),
            "image_dir": str(Path(args.image_dir)),
            "val_image_dir": str(Path(args.val_image_dir)),
            "checkpoint_dir": str(checkpoint_dir),
            "log_dir": str(log_dir),
        },
    }
    session_info_path = log_dir / f"session-{started}.json"
    save_json(session_info, session_info_path)
    print_session_info(session_info)

    run = None
    if args.use_wandb:
        if wandb is None:
            print("wandb is not installed; continuing without wandb.")
        else:
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args) | session_info,
            )

    best_map = -1.0
    jsonl_path = log_dir / f"train-{started}.jsonl"
    csv_path = log_dir / f"train-{started}.csv"

    for epoch in range(1, args.epochs + 1):
        train_logs = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        val_predictions = predict_dataset(
            model,
            val_dataset,
            val_loader,
            device,
            score_threshold=args.score_threshold,
            max_images=args.eval_max_images,
        )
        val_gt = ground_truth_from_dataset(val_dataset, max_images=args.eval_max_images)
        val_metrics = evaluate_map(val_gt, val_predictions, val_dataset.classes)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train/{k}": v for k, v in train_logs.items()},
            "val/mAP@0.5": val_metrics["mAP@0.5"],
            "val/micro_precision": val_metrics["micro_precision"],
            "val/micro_recall": val_metrics["micro_recall"],
        }
        append_csv(csv_path, row)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if run is not None:
            wandb.log(row, step=epoch)

        save_json(val_metrics, log_dir / f"metrics_epoch_{epoch}.json")
        save_checkpoint(
            checkpoint_dir / "last_model.pth",
            model,
            optimizer,
            epoch,
            train_dataset.classes,
            val_metrics,
        )
        if val_metrics["mAP@0.5"] > best_map:
            best_map = val_metrics["mAP@0.5"]
            save_checkpoint(
                checkpoint_dir / "best_model.pth",
                model,
                optimizer,
                epoch,
                train_dataset.classes,
                val_metrics,
            )

        print(
            f"epoch={epoch} loss={train_logs.get('loss', 0):.4f} "
            f"mAP@0.5={val_metrics['mAP@0.5']:.4f} "
            f"precision={val_metrics['micro_precision']:.4f} "
            f"recall={val_metrics['micro_recall']:.4f}"
        )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
