from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


def _none_if_missing(value: Any) -> Any:
    return None if value in {"", "null", "None"} else value


def _gpu_ids(gpus: Any) -> list[str]:
    value = _none_if_missing(gpus)
    if value is None:
        return []
    if isinstance(value, int):
        return [str(value)]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _configured_gpu_ids(cfg: DictConfig) -> list[str]:
    gpu_ids = _gpu_ids(cfg.device.get("gpus"))
    if gpu_ids:
        return gpu_ids

    visible_devices = _gpu_ids(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible_devices:
        return visible_devices

    gpu = _none_if_missing(cfg.device.get("gpu"))
    return [str(gpu)] if gpu is not None else []


def configure_cuda_visible_devices_from_hydra(cfg: DictConfig) -> None:
    if os.environ.get("LOCAL_RANK") is not None:
        return
    gpu_ids = _configured_gpu_ids(cfg)
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        # Keep the common typo unset so CUDA/PyTorch uses the canonical plural env var.
        os.environ.pop("CUDA_VISIBLE_DEVICE", None)


def maybe_launch_distributed_from_hydra(cfg: DictConfig) -> None:
    gpu_ids = _configured_gpu_ids(cfg)
    already_distributed = bool(cfg.device.get("distributed", False))
    if len(gpu_ids) < 2 or already_distributed or os.environ.get("LOCAL_RANK") is not None:
        return

    config_dir = Path(tempfile.mkdtemp(prefix="hydra_ddp_config_"))
    resolved_config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    resolved_config.device.distributed = True
    resolved_config.device.gpus = ",".join(gpu_ids)
    resolved_config.device.gpu = None
    config_path = config_dir / "train.yaml"
    OmegaConf.save(config=resolved_config, f=config_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env.pop("CUDA_VISIBLE_DEVICE", None)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(gpu_ids)),
        str(Path(__file__).resolve()),
        "--config-path",
        str(config_dir),
        "--config-name",
        "train",
    ]
    print(f"Launching Hydra DDP on GPUs: {', '.join(gpu_ids)}")
    subprocess.run(command, env=env, check=True)
    raise SystemExit(0)


def build_train_args(cfg: DictConfig) -> argparse.Namespace:
    data = OmegaConf.to_container(cfg, resolve=True)
    paths = data["paths"]
    run = data["run"]
    model = data["model"]
    optim = data["optim"]
    device = data["device"]
    augmentation = data["augmentation"]
    oversampling = data["oversampling"]
    early_stopping = data["early_stopping"]

    return argparse.Namespace(
        train_data=paths["train_data"],
        val_data=paths["val_data"],
        image_dir=paths["image_dir"],
        val_image_dir=paths["val_image_dir"],
        saved_results_dir=paths["saved_results_dir"],
        checkpoint_dir=_none_if_missing(paths.get("checkpoint_dir")),
        resume_from=_none_if_missing(paths.get("resume_from")),
        epochs=int(optim["epochs"]),
        batch_size=int(optim["batch_size"]),
        num_workers=int(optim["num_workers"]),
        lr=float(optim["lr"]),
        momentum=float(optim["momentum"]),
        weight_decay=float(optim["weight_decay"]),
        score_threshold=float(model["score_threshold"]),
        backbone=model["backbone"],
        custom=bool(model["custom"]),
        min_size=int(model["min_size"]),
        max_size=int(model["max_size"]),
        anchor_sizes=model.get("anchor_sizes", "") or "",
        anchor_ratios=model.get("anchor_ratios", "") or "",
        lr_milestones=str(optim["lr_milestones"]),
        lr_gamma=float(optim["lr_gamma"]),
        device=_none_if_missing(device.get("device")),
        gpu=None if (os.environ.get("LOCAL_RANK") is not None or _gpu_ids(device.get("gpus"))) else device.get("gpu"),
        gpus=_none_if_missing(device.get("gpus")),
        distributed=bool(device.get("distributed", False)) or os.environ.get("LOCAL_RANK") is not None,
        pretrained_backbone=bool(model["pretrained_backbone"]),
        wandb_project=run["wandb_project"],
        wandb_run_name=run["name"],
        use_wandb=bool(run["use_wandb"]),
        eval_max_images=int(run["eval_max_images"]),
        log_interval=int(run["log_interval"]),
        augmentation=bool(augmentation["enabled"]),
        horizontal_flip_probability=float(augmentation["horizontal_flip_probability"]),
        color_jitter_probability=float(augmentation["color_jitter_probability"]),
        grayscale_probability=float(augmentation["grayscale_probability"]),
        oversample_class=_none_if_missing(oversampling.get("class_name")),
        oversample_factor=float(oversampling["factor"]),
        early_stopping=bool(early_stopping["enabled"]),
        early_stopping_patience=int(early_stopping["patience"]),
        early_stopping_min_delta=float(early_stopping["min_delta"]),
    )


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    configure_cuda_visible_devices_from_hydra(cfg)
    maybe_launch_distributed_from_hydra(cfg)
    from train import main as train_main

    train_main(build_train_args(cfg))


if __name__ == "__main__":
    main()
