from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms as torchvision_nms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.modules import (
    CustomAnchorGenerator,
    FeaturePyramidNetwork,
    ROIHead,
    RPNHead,
    ResNetBackbone,
    assign_roi_targets,
    generate_proposals,
    roi_losses,
    rpn_losses,
)
from utils.dataset import DetectionModelTransform
from utils.helper import clip_boxes_to_image, decode_boxes, remove_small_boxes


CUSTOM_MODEL_VERSION = 5


class CustomFasterRCNN(nn.Module):
    def __init__(
        self,
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
        roi_channels: int = 256,
        train_pre_nms_top_n: int = 2000,
        train_post_nms_top_n: int = 2000,
        test_pre_nms_top_n: int = 1000,
        test_post_nms_top_n: int = 1000,
        fixed_batch_shape: bool = False,
        roi_dropout: float = 0.0,
        class_loss_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.box_score_thresh = box_score_thresh
        self.box_nms_thresh = box_nms_thresh
        self.train_pre_nms_top_n = train_pre_nms_top_n
        self.train_post_nms_top_n = train_post_nms_top_n
        self.test_pre_nms_top_n = test_pre_nms_top_n
        self.test_post_nms_top_n = test_post_nms_top_n
        self.transform = DetectionModelTransform(
            min_size=min_size,
            max_size=max_size,
            fixed_batch_shape=fixed_batch_shape,
        )
        self.backbone = ResNetBackbone(
            backbone_name=backbone_name,
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        self.fpn = FeaturePyramidNetwork(self.backbone.out_channels, out_channels=roi_channels)
        self.anchor_generator = CustomAnchorGenerator(
            sizes=anchor_sizes or (32, 64, 128, 256, 512),
            ratios=anchor_ratios or (0.5, 1.0, 2.0),
        )
        if class_loss_weights is not None:
            if class_loss_weights.numel() != num_classes:
                raise ValueError("class_loss_weights must include background plus all foreground classes.")
            self.register_buffer("class_loss_weights", class_loss_weights.float())
        else:
            self.class_loss_weights = None
        self.rpn_head = RPNHead(roi_channels, self.anchor_generator.num_anchors)
        self.roi_head = ROIHead(roi_channels, num_classes, dropout=roi_dropout)

    def rpn_forward(
        self,
        features: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        objectness_per_level = []
        bbox_deltas_per_level = []
        for feature in features.values():
            objectness, bbox_deltas = self.rpn_head(feature)
            objectness_per_level.append(objectness)
            bbox_deltas_per_level.append(bbox_deltas)
        return torch.cat(objectness_per_level, dim=1), torch.cat(bbox_deltas_per_level, dim=1)

    def postprocess_detections(
        self,
        class_logits: torch.Tensor,
        box_regression: torch.Tensor,
        proposals: list[torch.Tensor],
        image_sizes: list[tuple[int, int]],
        original_sizes: list[tuple[int, int]],
    ) -> list[dict[str, torch.Tensor]]:
        scores = F.softmax(class_logits, dim=-1)
        box_regression = box_regression.reshape(box_regression.shape[0], self.num_classes, 4)
        results = []
        start = 0
        for image_index, boxes in enumerate(proposals):
            num_boxes = boxes.shape[0]
            scores_per_image = scores[start : start + num_boxes]
            deltas_per_image = box_regression[start : start + num_boxes]
            start += num_boxes
            image_boxes = []
            image_scores = []
            image_labels = []
            for class_index in range(1, self.num_classes):
                class_scores = scores_per_image[:, class_index]
                keep = torch.where(class_scores >= self.box_score_thresh)[0]
                if keep.numel() == 0:
                    continue
                decoded = decode_boxes(deltas_per_image[keep, class_index], boxes[keep])
                decoded = clip_boxes_to_image(decoded, image_sizes[image_index])
                keep_size = remove_small_boxes(decoded, min_size=2)
                decoded = decoded[keep_size]
                kept_scores = class_scores[keep][keep_size]
                keep_nms = torchvision_nms(decoded, kept_scores, self.box_nms_thresh)
                image_boxes.append(decoded[keep_nms])
                image_scores.append(kept_scores[keep_nms])
                image_labels.append(
                    torch.full((keep_nms.numel(),), class_index, dtype=torch.long, device=boxes.device)
                )
            if image_boxes:
                final_boxes = torch.cat(image_boxes, dim=0)
                final_scores = torch.cat(image_scores, dim=0)
                final_labels = torch.cat(image_labels, dim=0)
                order = final_scores.argsort(descending=True)[:100]
                final_boxes = final_boxes[order]
                final_scores = final_scores[order]
                final_labels = final_labels[order]
            else:
                final_boxes = boxes.new_zeros((0, 4))
                final_scores = boxes.new_zeros((0,))
                final_labels = boxes.new_zeros((0,), dtype=torch.long)

            resized_h, resized_w = image_sizes[image_index]
            original_h, original_w = original_sizes[image_index]
            final_boxes[:, 0::2] *= original_w / resized_w
            final_boxes[:, 1::2] *= original_h / resized_h
            final_boxes = clip_boxes_to_image(final_boxes, original_sizes[image_index])
            results.append({"boxes": final_boxes, "labels": final_labels, "scores": final_scores})
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
        objectness, pred_bbox_deltas = self.rpn_forward(features)
        anchors = self.anchor_generator(features, batch.shape[-2:])
        num_anchors_per_level = [
            feature.shape[-2] * feature.shape[-1] * self.anchor_generator.num_anchors
            for feature in features.values()
        ]
        with torch.no_grad():
            proposals = generate_proposals(
                objectness.detach(),
                pred_bbox_deltas.detach(),
                anchors.detach(),
                resized_sizes,
                pre_nms_top_n=self.train_pre_nms_top_n if self.training else self.test_pre_nms_top_n,
                post_nms_top_n=self.train_post_nms_top_n if self.training else self.test_post_nms_top_n,
                num_anchors_per_level=num_anchors_per_level,
            )
        if self.training:
            assert resized_targets is not None
            loss_objectness, loss_rpn_box_reg = rpn_losses(
                objectness,
                pred_bbox_deltas,
                anchors,
                resized_targets,
                resized_sizes,
            )
            sampled_proposals, labels, regression_targets = assign_roi_targets(proposals, resized_targets)
            class_logits, box_regression = self.roi_head(features, sampled_proposals, resized_sizes)
            loss_classifier, loss_box_reg = roi_losses(
                class_logits,
                box_regression,
                labels,
                regression_targets,
                self.num_classes,
                self.class_loss_weights,
            )
            return {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }

        class_logits, box_regression = self.roi_head(features, proposals, resized_sizes)
        return self.postprocess_detections(class_logits, box_regression, proposals, resized_sizes, original_sizes)
