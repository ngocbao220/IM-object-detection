from __future__ import annotations

import torch

from models.faster_rcnn import CUSTOM_MODEL_VERSION, create_faster_rcnn
from models.modules import BACKBONE_WEIGHTS
from models.retina import RETINA_MODEL_VERSION, create_retinanet
from models.yolo import YOLO_MODEL_VERSION, create_yolo_like


MODEL_IMPL_CHOICES = ("torchvision", "custom", "retina", "yolo")


def create_detection_model(
    model_impl: str,
    num_classes: int,
    backbone_name: str,
    pretrained_backbone: bool,
    trainable_backbone_layers: int,
    min_size: int,
    max_size: int,
    anchor_sizes: tuple[int, ...] | None,
    anchor_ratios: tuple[float, ...] | None,
    box_score_thresh: float,
    box_nms_thresh: float,
    train_pre_nms_top_n: int,
    train_post_nms_top_n: int,
    test_pre_nms_top_n: int,
    test_post_nms_top_n: int,
    fixed_batch_shape: bool,
    roi_dropout: float,
    retina_topk_candidates: int = 1000,
    retina_max_detections: int = 300,
    yolo_topk_candidates: int = 1000,
    yolo_max_detections: int = 300,
) -> torch.nn.Module:
    if model_impl == "retina":
        return create_retinanet(
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
            topk_candidates=retina_topk_candidates,
            max_detections_per_image=retina_max_detections,
        )
    if model_impl == "yolo":
        return create_yolo_like(
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
            topk_candidates=yolo_topk_candidates,
            max_detections_per_image=yolo_max_detections,
        )
    return create_faster_rcnn(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
        anchor_sizes=anchor_sizes,
        anchor_ratios=anchor_ratios,
        custom=model_impl == "custom",
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        train_pre_nms_top_n=train_pre_nms_top_n,
        train_post_nms_top_n=train_post_nms_top_n,
        test_pre_nms_top_n=test_pre_nms_top_n,
        test_post_nms_top_n=test_post_nms_top_n,
        fixed_batch_shape=fixed_batch_shape,
        roi_dropout=roi_dropout,
    )


def model_version_for_impl(model_impl: str) -> int | None:
    if model_impl == "custom":
        return CUSTOM_MODEL_VERSION
    if model_impl == "retina":
        return RETINA_MODEL_VERSION
    if model_impl == "yolo":
        return YOLO_MODEL_VERSION
    return None
