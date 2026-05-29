from __future__ import annotations

import argparse

import torch
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn


def create_faster_rcnn_resnet50(
    num_classes: int,
    pretrained_backbone: bool = False,
    pretrained_coco: bool = False,
    trainable_backbone_layers: int = 3,
    min_size: int = 512,
    max_size: int = 768,
    box_score_thresh: float = 0.05,
) -> torch.nn.Module:
    """Create Faster R-CNN with ResNet-50 FPN backbone.

    num_classes includes the background class at index 0.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_coco else None
    weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone and not pretrained_coco else None
    effective_trainable_layers = (
        trainable_backbone_layers if weights is not None or weights_backbone is not None else None
    )
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=effective_trainable_layers,
        min_size=min_size,
        max_size=max_size,
        box_score_thresh=box_score_thresh,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Faster R-CNN model.")
    parser.add_argument("--num_classes", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = create_faster_rcnn_resnet50(args.num_classes)
    model.eval()
    images = [torch.rand(3, 256, 256)]
    with torch.no_grad():
        outputs = model(images)
    print(model.__class__.__name__)
    print({key: tuple(value.shape) for key, value in outputs[0].items()})


if __name__ == "__main__":
    main()
