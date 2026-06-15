from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.custom_faster_rcnn import CUSTOM_MODEL_VERSION, CustomFasterRCNN
from models.modules import BACKBONE_WEIGHTS


def create_custom_faster_rcnn(
    num_classes: int,
    backbone_name: str = "resnet101",
    pretrained_backbone: bool = False,
    trainable_backbone_layers: int = 2,
    min_size: int = 512,
    max_size: int = 768,
    anchor_sizes: tuple[int, ...] | None = None,
    anchor_ratios: tuple[float, ...] | None = None,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    train_pre_nms_top_n: int = 2000,
    train_post_nms_top_n: int = 2000,
    test_pre_nms_top_n: int = 1000,
    test_post_nms_top_n: int = 1000,
    fixed_batch_shape: bool = False,
    roi_dropout: float = 0.0,
    class_loss_weights: torch.Tensor | None = None,
) -> torch.nn.Module:
    return CustomFasterRCNN(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
        anchor_sizes=anchor_sizes,
        anchor_ratios=anchor_ratios,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        train_pre_nms_top_n=train_pre_nms_top_n,
        train_post_nms_top_n=train_post_nms_top_n,
        test_pre_nms_top_n=test_pre_nms_top_n,
        test_post_nms_top_n=test_post_nms_top_n,
        fixed_batch_shape=fixed_batch_shape,
        roi_dropout=roi_dropout,
        class_loss_weights=class_loss_weights,
    )


def create_faster_rcnn(
    num_classes: int,
    backbone_name: str = "resnet101",
    pretrained_backbone: bool = False,
    trainable_backbone_layers: int = 3,
    min_size: int = 512,
    max_size: int = 768,
    anchor_sizes: tuple[int, ...] | None = None,
    anchor_ratios: tuple[float, ...] | None = None,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    train_pre_nms_top_n: int = 2000,
    train_post_nms_top_n: int = 2000,
    test_pre_nms_top_n: int = 1000,
    test_post_nms_top_n: int = 1000,
    fixed_batch_shape: bool = False,
    roi_dropout: float = 0.0,
    class_loss_weights: torch.Tensor | None = None,
) -> torch.nn.Module:
    return create_custom_faster_rcnn(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
        anchor_sizes=anchor_sizes,
        anchor_ratios=anchor_ratios,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        train_pre_nms_top_n=train_pre_nms_top_n,
        train_post_nms_top_n=train_post_nms_top_n,
        test_pre_nms_top_n=test_pre_nms_top_n,
        test_post_nms_top_n=test_post_nms_top_n,
        fixed_batch_shape=fixed_batch_shape,
        roi_dropout=roi_dropout,
        class_loss_weights=class_loss_weights,
    )


def create_faster_rcnn_resnet101(num_classes: int, **kwargs: object) -> torch.nn.Module:
    return create_faster_rcnn(num_classes, backbone_name="resnet101", **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Faster R-CNN model factory.")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--backbone", choices=sorted(BACKBONE_WEIGHTS), default="resnet101")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = create_faster_rcnn(
        args.num_classes,
        backbone_name=args.backbone,
        min_size=128,
        max_size=192,
    )
    model.eval()
    images = [torch.rand(3, 128, 128)]
    with torch.no_grad():
        outputs = model(images)
    print(model.__class__.__name__)
    print({key: tuple(value.shape) for key, value in outputs[0].items()})


if __name__ == "__main__":
    main()
