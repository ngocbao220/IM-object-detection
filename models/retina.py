from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torchvision.ops import batched_nms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.modules import (
    BACKBONE_WEIGHTS,
    CustomAnchorGenerator,
    FeaturePyramidNetwork,
    ResNetBackbone,
    RetinaClassificationHead,
    RetinaRegressionHead,
    sigmoid_focal_loss,
)
from utils.dataset import DetectionModelTransform
from utils.helper import box_iou, clip_boxes_to_image, decode_boxes, encode_boxes, remove_small_boxes


RETINA_MODEL_VERSION = 1


class RetinaNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = False,
        trainable_backbone_layers: int = 2,
        min_size: int = 512,
        max_size: int = 768,
        anchor_sizes: tuple[int, ...] | None = None,
        anchor_ratios: tuple[float, ...] | None = None,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        max_detections_per_image: int = 300,
        topk_candidates: int = 1000,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("RetinaNet expects num_classes >= 2 including background.")
        self.num_classes = num_classes
        self.num_foreground_classes = num_classes - 1
        self.box_score_thresh = box_score_thresh
        self.box_nms_thresh = box_nms_thresh
        self.max_detections_per_image = max_detections_per_image
        self.topk_candidates = topk_candidates
        self.alpha = alpha
        self.gamma = gamma
        self.transform = DetectionModelTransform(min_size=min_size, max_size=max_size)
        self.backbone = ResNetBackbone(
            backbone_name=backbone_name,
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        self.fpn = FeaturePyramidNetwork(self.backbone.out_channels, out_channels=256)
        self.anchor_generator = CustomAnchorGenerator(
            sizes=anchor_sizes or (32, 64, 128, 256, 512),
            ratios=anchor_ratios or (0.5, 1.0, 2.0),
        )
        self.classification_head = RetinaClassificationHead(
            in_channels=256,
            num_anchors=self.anchor_generator.num_anchors,
            num_classes=self.num_foreground_classes,
        )
        self.regression_head = RetinaRegressionHead(
            in_channels=256,
            num_anchors=self.anchor_generator.num_anchors,
        )

    def assign_targets_to_anchors(
        self,
        anchors: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        image_sizes: list[tuple[int, int]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        matched_labels = []
        matched_boxes = []
        for target, image_size in zip(targets, image_sizes):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            labels = torch.full((anchors.shape[0],), -1, dtype=torch.long, device=anchors.device)
            matched_gt_boxes = torch.zeros_like(anchors)
            if gt_boxes.numel() == 0:
                labels[:] = 0
            else:
                valid_anchors = clip_boxes_to_image(anchors, image_size)
                ious = box_iou(valid_anchors, gt_boxes)
                max_iou, matched_idx = ious.max(dim=1)
                labels[max_iou < 0.4] = 0
                positive = max_iou >= 0.5
                labels[positive] = gt_labels[matched_idx[positive]]
                best_per_gt = ious.argmax(dim=0)
                labels[best_per_gt] = gt_labels
                matched_gt_boxes = gt_boxes[matched_idx]
            matched_labels.append(labels)
            matched_boxes.append(matched_gt_boxes)
        return matched_labels, matched_boxes

    def compute_loss(
        self,
        cls_logits: torch.Tensor,
        bbox_regression: torch.Tensor,
        anchors: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        image_sizes: list[tuple[int, int]],
    ) -> dict[str, torch.Tensor]:
        labels, matched_boxes = self.assign_targets_to_anchors(anchors, targets, image_sizes)
        cls_loss = cls_logits.new_zeros(())
        box_loss = bbox_regression.new_zeros(())
        normalizer = 0.0
        for image_index, labels_per_image in enumerate(labels):
            valid = labels_per_image >= 0
            positive = labels_per_image > 0
            targets_one_hot = cls_logits.new_zeros((labels_per_image.shape[0], self.num_foreground_classes))
            if positive.any():
                targets_one_hot[positive, labels_per_image[positive] - 1] = 1.0
            cls_loss = cls_loss + sigmoid_focal_loss(
                cls_logits[image_index][valid],
                targets_one_hot[valid],
                alpha=self.alpha,
                gamma=self.gamma,
                reduction="sum",
            )
            num_positive = int(positive.sum().item())
            normalizer += max(num_positive, 1)
            if num_positive:
                target_deltas = encode_boxes(matched_boxes[image_index][positive], anchors[positive])
                box_loss = box_loss + torch.nn.functional.smooth_l1_loss(
                    bbox_regression[image_index][positive],
                    target_deltas,
                    beta=1 / 9,
                    reduction="sum",
                )
        normalizer = max(normalizer, 1.0)
        return {
            "loss_classifier": cls_loss / normalizer,
            "loss_box_reg": box_loss / normalizer,
        }

    def postprocess_detections(
        self,
        cls_logits: torch.Tensor,
        bbox_regression: torch.Tensor,
        anchors: torch.Tensor,
        image_sizes: list[tuple[int, int]],
        original_sizes: list[tuple[int, int]],
    ) -> list[dict[str, torch.Tensor]]:
        results = []
        scores = cls_logits.sigmoid()
        for image_index, image_size in enumerate(image_sizes):
            scores_per_image = scores[image_index]
            bbox_reg_per_image = bbox_regression[image_index]
            image_boxes = []
            image_scores = []
            image_labels = []
            for class_index in range(self.num_foreground_classes):
                class_scores = scores_per_image[:, class_index]
                keep = torch.where(class_scores > self.box_score_thresh)[0]
                if keep.numel() == 0:
                    continue
                if keep.numel() > self.topk_candidates:
                    _, top_idx = class_scores[keep].topk(self.topk_candidates)
                    keep = keep[top_idx]
                boxes = decode_boxes(bbox_reg_per_image[keep], anchors[keep])
                boxes = clip_boxes_to_image(boxes, image_size)
                size_keep = remove_small_boxes(boxes, min_size=2)
                boxes = boxes[size_keep]
                class_scores_kept = class_scores[keep][size_keep]
                if boxes.numel() == 0:
                    continue
                labels = torch.full(
                    (boxes.shape[0],),
                    class_index + 1,
                    dtype=torch.long,
                    device=boxes.device,
                )
                image_boxes.append(boxes)
                image_scores.append(class_scores_kept)
                image_labels.append(labels)
            if image_boxes:
                boxes = torch.cat(image_boxes, dim=0)
                scores_cat = torch.cat(image_scores, dim=0)
                labels_cat = torch.cat(image_labels, dim=0)
                keep = batched_nms(boxes, scores_cat, labels_cat, self.box_nms_thresh)
                keep = keep[: self.max_detections_per_image]
                boxes = boxes[keep]
                scores_cat = scores_cat[keep]
                labels_cat = labels_cat[keep]
            else:
                boxes = anchors.new_zeros((0, 4))
                scores_cat = anchors.new_zeros((0,))
                labels_cat = anchors.new_zeros((0,), dtype=torch.long)
            resized_h, resized_w = image_size
            orig_h, orig_w = original_sizes[image_index]
            boxes[:, 0::2] *= orig_w / resized_w
            boxes[:, 1::2] *= orig_h / resized_h
            boxes = clip_boxes_to_image(boxes, original_sizes[image_index])
            results.append({"boxes": boxes, "labels": labels_cat, "scores": scores_cat})
        return results

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        if self.training and targets is None:
            raise ValueError("targets are required in training mode.")
        batch, original_sizes, resized_sizes, _scales, resized_targets = self.transform(images, targets)
        features = self.fpn(self.backbone(batch))
        cls_logits = self.classification_head(features)
        bbox_regression = self.regression_head(features)
        anchors = self.anchor_generator(features, batch.shape[-2:])
        if self.training:
            assert resized_targets is not None
            return self.compute_loss(cls_logits, bbox_regression, anchors, resized_targets, resized_sizes)
        return self.postprocess_detections(cls_logits, bbox_regression, anchors, resized_sizes, original_sizes)


def create_retinanet(
    num_classes: int,
    backbone_name: str = "resnet50",
    pretrained_backbone: bool = False,
    trainable_backbone_layers: int = 2,
    min_size: int = 512,
    max_size: int = 768,
    anchor_sizes: tuple[int, ...] | None = None,
    anchor_ratios: tuple[float, ...] | None = None,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    max_detections_per_image: int = 300,
    topk_candidates: int = 1000,
) -> nn.Module:
    return RetinaNet(
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
        max_detections_per_image=max_detections_per_image,
        topk_candidates=topk_candidates,
    )

