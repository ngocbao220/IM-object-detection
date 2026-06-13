from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator as TorchvisionAnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN as TorchvisionFasterRCNN

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.custom_faster_rcnn import CUSTOM_MODEL_VERSION, CustomFasterRCNN
from models.modules import BACKBONE_FACTORIES, BACKBONE_WEIGHTS


def create_torchvision_faster_rcnn(
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
) -> torch.nn.Module:
    if backbone_name not in BACKBONE_FACTORIES:
        raise ValueError(
            f"Torchvision Faster R-CNN only supports ResNet backbones here. "
            f"Got {backbone_name}. Use MODEL_IMPL=retina/custom/yolo for non-ResNet backbones."
        )
    weights_backbone = BACKBONE_WEIGHTS[backbone_name] if pretrained_backbone else None
    effective_trainable_layers = trainable_backbone_layers if weights_backbone is not None else 5
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=weights_backbone,
        trainable_layers=effective_trainable_layers,
    )
    anchor_generator = None
    if anchor_sizes is not None or anchor_ratios is not None:
        sizes = anchor_sizes or (32, 64, 128, 256, 512)
        ratios = anchor_ratios or (0.5, 1.0, 2.0)
        if len(sizes) != 5:
            raise ValueError("Torchvision FPN Faster R-CNN expects exactly 5 anchor sizes.")
        anchor_generator = TorchvisionAnchorGenerator(
            sizes=tuple((size,) for size in sizes),
            aspect_ratios=tuple(tuple(ratios) for _ in sizes),
        )
    return TorchvisionFasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        rpn_anchor_generator=anchor_generator,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
    )


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
    custom: bool = False,
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
    if custom:
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
    return create_torchvision_faster_rcnn(
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
    )


def create_faster_rcnn_resnet101(num_classes: int, **kwargs: object) -> torch.nn.Module:
    return create_faster_rcnn(num_classes, backbone_name="resnet101", **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Faster R-CNN model factory.")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--backbone", choices=sorted(BACKBONE_WEIGHTS), default="resnet101")
    parser.add_argument("--custom", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = create_faster_rcnn(
        args.num_classes,
        backbone_name=args.backbone,
        min_size=128,
        max_size=192,
        custom=args.custom,
    )
    model.eval()
    images = [torch.rand(3, 128, 128)]
    with torch.no_grad():
        outputs = model(images)
    print(model.__class__.__name__)
    print({key: tuple(value.shape) for key, value in outputs[0].items()})


if __name__ == "__main__":
    main()
