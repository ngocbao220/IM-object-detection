from __future__ import annotations

import argparse

import torch
from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN


BACKBONE_WEIGHTS = {
    "resnet50": ResNet50_Weights.DEFAULT,
    "resnet101": ResNet101_Weights.DEFAULT,
}


def create_faster_rcnn(
    num_classes: int,
    backbone_name: str = "resnet101",
    pretrained_backbone: bool = False,
    trainable_backbone_layers: int = 3,
    min_size: int = 512,
    max_size: int = 768,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
) -> torch.nn.Module:
    """Create Faster R-CNN with an optional ImageNet-pretrained ResNet backbone.

    Detection components are initialized from scratch. num_classes includes the
    background class at index 0.
    """
    if backbone_name not in BACKBONE_WEIGHTS:
        raise ValueError(f"Unsupported backbone: {backbone_name}. Choose one of {sorted(BACKBONE_WEIGHTS)}.")
    weights_backbone = BACKBONE_WEIGHTS[backbone_name] if pretrained_backbone else None
    effective_trainable_layers = trainable_backbone_layers if weights_backbone is not None else 5
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=weights_backbone,
        trainable_layers=effective_trainable_layers,
    )
    return FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
    )


def create_faster_rcnn_resnet101(num_classes: int, **kwargs: object) -> torch.nn.Module:
    """Backward-compatible ResNet-101 model factory."""
    return create_faster_rcnn(num_classes, backbone_name="resnet101", **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Faster R-CNN model.")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--backbone", choices=sorted(BACKBONE_WEIGHTS), default="resnet101")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = create_faster_rcnn(args.num_classes, backbone_name=args.backbone)
    model.eval()
    images = [torch.rand(3, 256, 256)]
    with torch.no_grad():
        outputs = model(images)
    print(model.__class__.__name__)
    print({key: tuple(value.shape) for key, value in outputs[0].items()})


if __name__ == "__main__":
    main()
