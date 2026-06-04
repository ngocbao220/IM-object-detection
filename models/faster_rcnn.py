from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet101_Weights, ResNet50_Weights, resnet101, resnet50
from torchvision.models.detection.anchor_utils import AnchorGenerator as TorchvisionAnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN as TorchvisionFasterRCNN


BACKBONE_WEIGHTS = {
    "resnet50": ResNet50_Weights.DEFAULT,
    "resnet101": ResNet101_Weights.DEFAULT,
}

BACKBONE_FACTORIES = {
    "resnet50": resnet50,
    "resnet101": resnet101,
}

BACKBONE_OUT_CHANNELS = {
    "resnet50": 1024,
    "resnet101": 1024,
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = box_area(boxes1)[:, None] + box_area(boxes2) - inter
    return inter / union.clamp(min=1e-6)


def clip_boxes_to_image(boxes: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)
    return boxes


def remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    return torch.where((widths >= min_size) & (heights >= min_size))[0]


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_empty((0,), dtype=torch.long)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[current].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep) if keep else boxes.new_empty((0,), dtype=torch.long)


def encode_boxes(reference_boxes: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
    widths = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
    heights = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights

    gt_widths = (reference_boxes[:, 2] - reference_boxes[:, 0]).clamp(min=1e-6)
    gt_heights = (reference_boxes[:, 3] - reference_boxes[:, 1]).clamp(min=1e-6)
    gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

    return torch.stack(
        [
            (gt_ctr_x - ctr_x) / widths,
            (gt_ctr_y - ctr_y) / heights,
            torch.log(gt_widths / widths),
            torch.log(gt_heights / heights),
        ],
        dim=1,
    )


def decode_boxes(deltas: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0].clamp(min=-10, max=10)
    dy = deltas[:, 1].clamp(min=-10, max=10)
    dw = deltas[:, 2].clamp(min=-5, max=5)
    dh = deltas[:, 3].clamp(min=-5, max=5)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights
    return torch.stack(
        [
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ],
        dim=1,
    )


def resize_image_and_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor | None,
    min_size: int,
    max_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    _, height, width = image.shape
    short_side = min(height, width)
    long_side = max(height, width)
    scale = min_size / short_side
    if long_side * scale > max_size:
        scale = max_size / long_side
    new_height = int(round(height * scale))
    new_width = int(round(width * scale))
    image = F.interpolate(
        image.unsqueeze(0),
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    if boxes is not None:
        boxes = boxes * scale
    return image, boxes, scale


def pad_images(images: list[torch.Tensor], size_divisible: int = 16) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    image_sizes = [(image.shape[-2], image.shape[-1]) for image in images]
    max_height = max(size[0] for size in image_sizes)
    max_width = max(size[1] for size in image_sizes)
    max_height = int(math.ceil(max_height / size_divisible) * size_divisible)
    max_width = int(math.ceil(max_width / size_divisible) * size_divisible)
    batch = images[0].new_zeros((len(images), 3, max_height, max_width))
    for index, image in enumerate(images):
        _, height, width = image.shape
        batch[index, :, :height, :width] = image
    return batch, image_sizes


class ResNetBackbone(nn.Module):
    """ImageNet-pretrained ResNet feature extractor; detection heads are custom."""

    def __init__(
        self,
        backbone_name: str = "resnet101",
        pretrained_backbone: bool = False,
        trainable_backbone_layers: int = 3,
    ) -> None:
        super().__init__()
        if backbone_name not in BACKBONE_FACTORIES:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose one of {sorted(BACKBONE_FACTORIES)}.")
        weights = BACKBONE_WEIGHTS[backbone_name] if pretrained_backbone else None
        backbone = BACKBONE_FACTORIES[backbone_name](weights=weights)
        self.body = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", backbone.conv1),
                    ("bn1", backbone.bn1),
                    ("relu", backbone.relu),
                    ("maxpool", backbone.maxpool),
                    ("layer1", backbone.layer1),
                    ("layer2", backbone.layer2),
                    ("layer3", backbone.layer3),
                ]
            )
        )
        self.out_channels = BACKBONE_OUT_CHANNELS[backbone_name]
        if pretrained_backbone:
            layers = [backbone.layer1, backbone.layer2, backbone.layer3]
            frozen_layers = layers[: max(0, len(layers) - trainable_backbone_layers)]
            for layer in frozen_layers:
                for parameter in layer.parameters():
                    parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.body(images)


class CustomAnchorGenerator(nn.Module):
    def __init__(
        self,
        sizes: tuple[int, ...] = (64, 128, 192, 256, 512),
        ratios: tuple[float, ...] = (0.33, 0.5, 1.0, 2.0),
    ) -> None:
        super().__init__()
        self.sizes = sizes
        self.ratios = ratios
        anchors = []
        for size in sizes:
            area = float(size * size)
            for ratio in ratios:
                width = math.sqrt(area * ratio)
                height = area / width
                anchors.append([-0.5 * width, -0.5 * height, 0.5 * width, 0.5 * height])
        self.register_buffer("base_anchors", torch.tensor(anchors, dtype=torch.float32))

    @property
    def num_anchors(self) -> int:
        return int(self.base_anchors.shape[0])

    def forward(
        self,
        feature: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        _, _, feature_height, feature_width = feature.shape
        image_height, image_width = image_size
        stride_y = image_height / feature_height
        stride_x = image_width / feature_width
        shifts_x = (torch.arange(feature_width, device=feature.device, dtype=torch.float32) + 0.5) * stride_x
        shifts_y = (torch.arange(feature_height, device=feature.device, dtype=torch.float32) + 0.5) * stride_y
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts = torch.stack(
            [shift_x.reshape(-1), shift_y.reshape(-1), shift_x.reshape(-1), shift_y.reshape(-1)],
            dim=1,
        )
        return (shifts[:, None, :] + self.base_anchors[None, :, :]).reshape(-1, 4)


class RPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.objectness = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        for layer in [self.conv, self.objectness, self.bbox_reg]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.conv(feature))
        objectness = self.objectness(hidden)
        bbox_reg = self.bbox_reg(hidden)
        batch_size, anchors, height, width = objectness.shape
        objectness = objectness.permute(0, 2, 3, 1).reshape(batch_size, -1)
        bbox_reg = bbox_reg.view(batch_size, anchors, 4, height, width)
        bbox_reg = bbox_reg.permute(0, 3, 4, 1, 2).reshape(batch_size, -1, 4)
        return objectness, bbox_reg


class ROIHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, pool_size: int = 7) -> None:
        super().__init__()
        self.pool_size = pool_size
        hidden = 1024
        self.fc1 = nn.Linear(in_channels * pool_size * pool_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.cls_score = nn.Linear(hidden, num_classes)
        self.bbox_pred = nn.Linear(hidden, num_classes * 4)
        for layer in [self.fc1, self.fc2, self.cls_score, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def pool(
        self,
        feature: torch.Tensor,
        proposals: list[torch.Tensor],
        image_sizes: list[tuple[int, int]],
    ) -> torch.Tensor:
        pooled = []
        _, channels, feature_height, feature_width = feature.shape
        for image_index, boxes in enumerate(proposals):
            image_height, image_width = image_sizes[image_index]
            scale_x = feature_width / image_width
            scale_y = feature_height / image_height
            for box in boxes:
                x1, y1, x2, y2 = box
                fx1 = int(torch.floor(x1 * scale_x).clamp(0, feature_width - 1).item())
                fy1 = int(torch.floor(y1 * scale_y).clamp(0, feature_height - 1).item())
                fx2 = int(torch.ceil(x2 * scale_x).clamp(fx1 + 1, feature_width).item())
                fy2 = int(torch.ceil(y2 * scale_y).clamp(fy1 + 1, feature_height).item())
                crop = feature[image_index : image_index + 1, :, fy1:fy2, fx1:fx2]
                pooled.append(F.adaptive_max_pool2d(crop, self.pool_size).squeeze(0))
        if not pooled:
            return feature.new_zeros((0, channels, self.pool_size, self.pool_size))
        return torch.stack(pooled, dim=0)

    def forward(
        self,
        feature: torch.Tensor,
        proposals: list[torch.Tensor],
        image_sizes: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool(feature, proposals, image_sizes)
        if pooled.numel() == 0:
            return pooled.new_zeros((0, self.cls_score.out_features)), pooled.new_zeros((0, self.bbox_pred.out_features))
        x = pooled.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.cls_score(x), self.bbox_pred(x)


class CustomFasterRCNN(nn.Module):
    def __init__(
        self,
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
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.min_size = min_size
        self.max_size = max_size
        self.box_score_thresh = box_score_thresh
        self.box_nms_thresh = box_nms_thresh
        self.backbone = ResNetBackbone(backbone_name, pretrained_backbone, trainable_backbone_layers)
        self.anchor_generator = CustomAnchorGenerator(
            sizes=anchor_sizes or (64, 128, 192, 256, 512),
            ratios=anchor_ratios or (0.33, 0.5, 1.0, 2.0),
        )
        self.rpn_head = RPNHead(self.backbone.out_channels, self.anchor_generator.num_anchors)
        self.roi_head = ROIHead(self.backbone.out_channels, num_classes)

    def transform(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[tuple[int, int]], list[float], list[dict[str, torch.Tensor]] | None]:
        normalized_images = []
        original_sizes = []
        resized_sizes = []
        scales = []
        new_targets = [] if targets is not None else None
        mean = IMAGENET_MEAN.to(images[0].device)
        std = IMAGENET_STD.to(images[0].device)
        for index, image in enumerate(images):
            original_sizes.append((image.shape[-2], image.shape[-1]))
            boxes = targets[index]["boxes"] if targets is not None else None
            resized, resized_boxes, scale = resize_image_and_boxes(image, boxes, self.min_size, self.max_size)
            normalized_images.append((resized - mean) / std)
            resized_sizes.append((resized.shape[-2], resized.shape[-1]))
            scales.append(scale)
            if targets is not None and new_targets is not None:
                target = {key: value for key, value in targets[index].items()}
                target["boxes"] = resized_boxes
                new_targets.append(target)
        batch, _ = pad_images(normalized_images)
        return batch, original_sizes, resized_sizes, scales, new_targets

    def assign_rpn_targets(
        self,
        anchors: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        image_sizes: list[tuple[int, int]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        labels = []
        regression_targets = []
        for target, image_size in zip(targets, image_sizes):
            anchors_in_image = clip_boxes_to_image(anchors, image_size)
            gt_boxes = target["boxes"]
            label = torch.full((anchors.shape[0],), -1.0, device=anchors.device)
            matched_gt = torch.zeros_like(anchors)
            if gt_boxes.numel() == 0:
                label[:] = 0
            else:
                ious = box_iou(anchors_in_image, gt_boxes)
                max_iou, matched_idx = ious.max(dim=1)
                label[max_iou < 0.3] = 0
                label[max_iou >= 0.7] = 1
                best_per_gt = ious.argmax(dim=0)
                label[best_per_gt] = 1
                matched_gt = gt_boxes[matched_idx]
            regression_targets.append(encode_boxes(matched_gt, anchors_in_image))
            labels.append(label)
        return labels, regression_targets

    def sample_labels(self, labels: torch.Tensor, batch_size: int, positive_fraction: float) -> torch.Tensor:
        positive = torch.where(labels == 1)[0]
        negative = torch.where(labels == 0)[0]
        num_positive = min(int(batch_size * positive_fraction), positive.numel())
        num_negative = min(batch_size - num_positive, negative.numel())
        perm_pos = positive[torch.randperm(positive.numel(), device=labels.device)[:num_positive]]
        perm_neg = negative[torch.randperm(negative.numel(), device=labels.device)[:num_negative]]
        return torch.cat([perm_pos, perm_neg], dim=0)

    def rpn_losses(
        self,
        objectness: torch.Tensor,
        pred_bbox_deltas: torch.Tensor,
        anchors: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        image_sizes: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels, regression_targets = self.assign_rpn_targets(anchors, targets, image_sizes)
        sampled_indices = []
        for labels_per_image in labels:
            sampled_indices.append(self.sample_labels(labels_per_image, 256, 0.5))
        objectness_loss = objectness.sum() * 0
        box_loss = pred_bbox_deltas.sum() * 0
        for image_index, sampled in enumerate(sampled_indices):
            labels_per_image = labels[image_index][sampled]
            objectness_loss = objectness_loss + F.binary_cross_entropy_with_logits(
                objectness[image_index][sampled],
                labels_per_image,
            )
            positives = sampled[labels_per_image == 1]
            if positives.numel():
                box_loss = box_loss + F.smooth_l1_loss(
                    pred_bbox_deltas[image_index][positives],
                    regression_targets[image_index][positives],
                    beta=1 / 9,
                    reduction="sum",
                ) / max(positives.numel(), 1)
        normalizer = max(len(sampled_indices), 1)
        return objectness_loss / normalizer, box_loss / normalizer

    def generate_proposals(
        self,
        objectness: torch.Tensor,
        pred_bbox_deltas: torch.Tensor,
        anchors: torch.Tensor,
        image_sizes: list[tuple[int, int]],
    ) -> list[torch.Tensor]:
        proposals = []
        pre_nms_top_n = 2000 if self.training else 1000
        post_nms_top_n = 1000 if self.training else 300
        for image_index, image_size in enumerate(image_sizes):
            scores = objectness[image_index].sigmoid()
            num_top = min(pre_nms_top_n, scores.numel())
            top_scores, top_idx = scores.topk(num_top)
            boxes = decode_boxes(pred_bbox_deltas[image_index][top_idx], anchors[top_idx])
            boxes = clip_boxes_to_image(boxes, image_size)
            keep = remove_small_boxes(boxes, min_size=2)
            boxes, top_scores = boxes[keep], top_scores[keep]
            keep = nms(boxes, top_scores, 0.7)[:post_nms_top_n]
            proposals.append(boxes[keep].detach())
        return proposals

    def assign_roi_targets(
        self,
        proposals: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        sampled_proposals = []
        labels = []
        regression_targets = []
        for proposals_per_image, target in zip(proposals, targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            proposals_per_image = torch.cat([proposals_per_image, gt_boxes], dim=0)
            if gt_boxes.numel() == 0:
                labels_per_image = torch.zeros((proposals_per_image.shape[0],), dtype=torch.long, device=proposals_per_image.device)
                matched_gt = torch.zeros_like(proposals_per_image)
            else:
                ious = box_iou(proposals_per_image, gt_boxes)
                max_iou, matched_idx = ious.max(dim=1)
                labels_per_image = gt_labels[matched_idx]
                labels_per_image[max_iou < 0.5] = 0
                ignore = (max_iou >= 0.0) & (max_iou < 0.1)
                labels_per_image[ignore] = -1
                matched_gt = gt_boxes[matched_idx]
            sampled = self.sample_labels((labels_per_image > 0).float().where(labels_per_image >= 0, torch.tensor(-1.0, device=labels_per_image.device)), 128, 0.25)
            sampled_proposals.append(proposals_per_image[sampled])
            labels.append(labels_per_image[sampled].clamp(min=0))
            regression_targets.append(encode_boxes(matched_gt[sampled], proposals_per_image[sampled]))
        return sampled_proposals, labels, regression_targets

    def roi_losses(
        self,
        class_logits: torch.Tensor,
        box_regression: torch.Tensor,
        labels: list[torch.Tensor],
        regression_targets: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels_cat = torch.cat(labels, dim=0)
        regression_targets_cat = torch.cat(regression_targets, dim=0)
        classification_loss = F.cross_entropy(class_logits, labels_cat)
        positive = torch.where(labels_cat > 0)[0]
        if positive.numel() == 0:
            return classification_loss, box_regression.sum() * 0
        box_regression = box_regression.reshape(box_regression.shape[0], self.num_classes, 4)
        box_loss = F.smooth_l1_loss(
            box_regression[positive, labels_cat[positive]],
            regression_targets_cat[positive],
            beta=1 / 9,
            reduction="sum",
        ) / labels_cat.numel()
        return classification_loss, box_loss

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
                keep_nms = nms(decoded, kept_scores, self.box_nms_thresh)
                image_boxes.append(decoded[keep_nms])
                image_scores.append(kept_scores[keep_nms])
                image_labels.append(torch.full((keep_nms.numel(),), class_index, dtype=torch.long, device=boxes.device))
            if image_boxes:
                final_boxes = torch.cat(image_boxes, dim=0)
                final_scores = torch.cat(image_scores, dim=0)
                final_labels = torch.cat(image_labels, dim=0)
                order = final_scores.argsort(descending=True)[:100]
                final_boxes, final_scores, final_labels = final_boxes[order], final_scores[order], final_labels[order]
            else:
                final_boxes = boxes.new_zeros((0, 4))
                final_scores = boxes.new_zeros((0,))
                final_labels = boxes.new_zeros((0,), dtype=torch.long)
            resized_h, resized_w = image_sizes[image_index]
            original_h, original_w = original_sizes[image_index]
            scale_x = original_w / resized_w
            scale_y = original_h / resized_h
            final_boxes[:, 0::2] *= scale_x
            final_boxes[:, 1::2] *= scale_y
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
        feature = self.backbone(batch)
        objectness, pred_bbox_deltas = self.rpn_head(feature)
        anchors = self.anchor_generator(feature, batch.shape[-2:])
        proposals = self.generate_proposals(objectness, pred_bbox_deltas, anchors, resized_sizes)

        if self.training:
            assert resized_targets is not None
            loss_objectness, loss_rpn_box_reg = self.rpn_losses(
                objectness,
                pred_bbox_deltas,
                anchors,
                resized_targets,
                resized_sizes,
            )
            sampled_proposals, labels, regression_targets = self.assign_roi_targets(proposals, resized_targets)
            class_logits, box_regression = self.roi_head(feature, sampled_proposals, resized_sizes)
            loss_classifier, loss_box_reg = self.roi_losses(class_logits, box_regression, labels, regression_targets)
            return {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }

        class_logits, box_regression = self.roi_head(feature, proposals, resized_sizes)
        return self.postprocess_detections(class_logits, box_regression, proposals, resized_sizes, original_sizes)


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
) -> torch.nn.Module:
    """Create Faster R-CNN.

    By default this returns the torchvision Faster R-CNN implementation. Set
    custom=True to use the detector implemented in this repository. In both
    cases, pretrained_backbone controls only ImageNet backbone weights.
    """
    if not custom:
        if backbone_name not in BACKBONE_WEIGHTS:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose one of {sorted(BACKBONE_WEIGHTS)}.")
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
    )


def create_faster_rcnn_resnet101(num_classes: int, **kwargs: object) -> torch.nn.Module:
    """Backward-compatible ResNet-101 model factory."""
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
