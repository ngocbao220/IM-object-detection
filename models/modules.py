from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet101_Weights, ResNet50_Weights, resnet101, resnet50
from torchvision.ops import nms as torchvision_nms
from torchvision.ops import roi_align

from utils.helper import (
    box_iou,
    clip_boxes_to_image,
    decode_boxes,
    encode_boxes,
    remove_small_boxes,
)


BACKBONE_WEIGHTS = {
    "resnet50": ResNet50_Weights.DEFAULT,
    "resnet101": ResNet101_Weights.DEFAULT,
}

BACKBONE_FACTORIES = {
    "resnet50": resnet50,
    "resnet101": resnet101,
}

BACKBONE_STAGE_CHANNELS = {
    "resnet50": (256, 512, 1024, 2048),
    "resnet101": (256, 512, 1024, 2048),
}


class ResNetBackbone(nn.Module):
    """ImageNet-pretrained ResNet feature extractor for the custom detector."""

    def __init__(
        self,
        backbone_name: str = "resnet101",
        pretrained_backbone: bool = False,
        trainable_backbone_layers: int = 2,
    ) -> None:
        super().__init__()
        if backbone_name not in BACKBONE_FACTORIES:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose one of {sorted(BACKBONE_FACTORIES)}.")
        weights = BACKBONE_WEIGHTS[backbone_name] if pretrained_backbone else None
        backbone = BACKBONE_FACTORIES[backbone_name](weights=weights)
        self.stem = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", backbone.conv1),
                    ("bn1", backbone.bn1),
                    ("relu", backbone.relu),
                    ("maxpool", backbone.maxpool),
                ]
            )
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_channels = BACKBONE_STAGE_CHANNELS[backbone_name]
        if pretrained_backbone:
            for parameter in backbone.conv1.parameters():
                parameter.requires_grad_(False)
            for parameter in backbone.bn1.parameters():
                parameter.requires_grad_(False)

            layers = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
            frozen_layers = layers[: max(0, len(layers) - trainable_backbone_layers)]
            for layer in frozen_layers:
                for parameter in layer.parameters():
                    parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        x = self.stem(images)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return OrderedDict([("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)])


class FeaturePyramidNetwork(nn.Module):
    """Small FPN implementation producing P2-P6 feature maps."""

    def __init__(self, in_channels_list: tuple[int, ...], out_channels: int = 256) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            layer_block = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            nn.init.kaiming_uniform_(inner_block.weight, a=1)
            nn.init.constant_(inner_block.bias, 0)
            nn.init.kaiming_uniform_(layer_block.weight, a=1)
            nn.init.constant_(layer_block.bias, 0)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

    def forward(self, features: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        tensors = list(features.values())
        last_inner = self.inner_blocks[-1](tensors[-1])
        results = [self.layer_blocks[-1](last_inner)]

        for index in range(len(tensors) - 2, -1, -1):
            lateral = self.inner_blocks[index](tensors[index])
            top_down = F.interpolate(last_inner, size=lateral.shape[-2:], mode="nearest")
            last_inner = lateral + top_down
            results.insert(0, self.layer_blocks[index](last_inner))

        output = OrderedDict((f"p{index + 2}", result) for index, result in enumerate(results))
        output["p6"] = F.max_pool2d(results[-1], kernel_size=1, stride=2)
        return output


class CustomAnchorGenerator(nn.Module):
    def __init__(
        self,
        sizes: tuple[int, ...] = (32, 64, 128, 256, 512),
        ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
    ) -> None:
        super().__init__()
        if len(sizes) != 5:
            raise ValueError("Custom FPN anchor generator expects exactly 5 sizes for P2-P6.")
        self.sizes = sizes
        self.ratios = ratios
        anchors_by_level = []
        for size in sizes:
            level_anchors = []
            area = float(size * size)
            for ratio in ratios:
                width = math.sqrt(area * ratio)
                height = area / width
                level_anchors.append([-0.5 * width, -0.5 * height, 0.5 * width, 0.5 * height])
            anchors_by_level.append(level_anchors)
        self.register_buffer("base_anchors", torch.tensor(anchors_by_level, dtype=torch.float32))

    @property
    def num_anchors(self) -> int:
        return int(self.base_anchors.shape[1])

    def _grid_anchors(
        self,
        feature: torch.Tensor,
        image_size: tuple[int, int],
        level_index: int,
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
        base_anchors = self.base_anchors[level_index].to(device=feature.device, dtype=feature.dtype)
        return (shifts[:, None, :] + base_anchors[None, :, :]).reshape(-1, 4)

    def forward(self, features: OrderedDict[str, torch.Tensor], image_size: tuple[int, int]) -> torch.Tensor:
        anchors = [
            self._grid_anchors(feature, image_size, level_index)
            for level_index, feature in enumerate(features.values())
        ]
        return torch.cat(anchors, dim=0)


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
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pool_size: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1).")
        self.pool_size = pool_size
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        hidden = 1024
        self.fc1 = nn.Linear(in_channels * pool_size * pool_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.cls_score = nn.Linear(hidden, num_classes)
        self.bbox_pred = nn.Linear(hidden, num_classes * 4)
        for layer in [self.fc1, self.fc2, self.cls_score, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def _proposal_levels(self, proposals: torch.Tensor) -> torch.Tensor:
        widths = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
        heights = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)
        scales = torch.sqrt(widths * heights)
        levels = torch.floor(4 + torch.log2(scales / 224.0 + 1e-6))
        return levels.clamp(min=2, max=5).to(torch.int64) - 2

    def pool(
        self,
        features: OrderedDict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_sizes: list[tuple[int, int]],
    ) -> torch.Tensor:
        feature_items = [(name, feature) for name, feature in features.items() if name in {"p2", "p3", "p4", "p5"}]
        reference = feature_items[0][1]
        _, channels, _, _ = reference.shape
        if not any(boxes.numel() for boxes in proposals):
            return reference.new_zeros((0, channels, self.pool_size, self.pool_size))

        all_rois = []
        all_levels = []
        roi_offset = 0
        roi_indices = []
        for image_index, boxes in enumerate(proposals):
            if boxes.numel() == 0:
                continue
            levels = self._proposal_levels(boxes)
            batch_indices = torch.full(
                (boxes.shape[0], 1),
                image_index,
                dtype=boxes.dtype,
                device=boxes.device,
            )
            all_rois.append(torch.cat([batch_indices, boxes], dim=1))
            all_levels.append(levels)
            roi_indices.append(torch.arange(roi_offset, roi_offset + boxes.shape[0], device=boxes.device))
            roi_offset += boxes.shape[0]

        rois = torch.cat(all_rois, dim=0)
        levels = torch.cat(all_levels, dim=0)
        original_indices = torch.cat(roi_indices, dim=0)
        pooled = reference.new_zeros((rois.shape[0], channels, self.pool_size, self.pool_size))

        for level_index, (_name, feature) in enumerate(feature_items):
            keep = torch.where(levels == level_index)[0]
            if keep.numel() == 0:
                continue
            _, _, feature_height, feature_width = feature.shape
            level_rois = rois[keep].clone()
            for image_index, image_size in enumerate(image_sizes):
                image_keep = torch.where(level_rois[:, 0] == image_index)[0]
                if image_keep.numel() == 0:
                    continue
                image_height, image_width = image_size
                level_rois[image_keep, 1::2] *= feature_width / image_width
                level_rois[image_keep, 2::2] *= feature_height / image_height
            pooled[original_indices[keep]] = roi_align(
                feature,
                level_rois,
                output_size=(self.pool_size, self.pool_size),
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True,
            )
        return pooled

    def forward(
        self,
        features: OrderedDict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_sizes: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool(features, proposals, image_sizes)
        if pooled.numel() == 0:
            return (
                pooled.new_zeros((0, self.cls_score.out_features)),
                pooled.new_zeros((0, self.bbox_pred.out_features)),
            )
        x = pooled.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.cls_score(x), self.bbox_pred(x)


def sample_labels(labels: torch.Tensor, batch_size: int, positive_fraction: float) -> torch.Tensor:
    positive = torch.where(labels == 1)[0]
    negative = torch.where(labels == 0)[0]
    num_positive = min(int(batch_size * positive_fraction), positive.numel())
    num_negative = min(batch_size - num_positive, negative.numel())
    perm_pos = positive[torch.randperm(positive.numel(), device=labels.device)[:num_positive]]
    perm_neg = negative[torch.randperm(negative.numel(), device=labels.device)[:num_negative]]
    return torch.cat([perm_pos, perm_neg], dim=0)


def assign_rpn_targets(
    anchors: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    image_sizes: list[tuple[int, int]],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    labels = []
    matched_indices = []
    clipped_anchors = []
    for target, image_size in zip(targets, image_sizes):
        anchors_in_image = clip_boxes_to_image(anchors, image_size)
        gt_boxes = target["boxes"]
        label = torch.full((anchors.shape[0],), -1.0, device=anchors.device)
        matched_idx = torch.zeros((anchors.shape[0],), dtype=torch.long, device=anchors.device)
        if gt_boxes.numel() == 0:
            label[:] = 0
        else:
            ious = box_iou(anchors_in_image, gt_boxes)
            max_iou, matched_idx = ious.max(dim=1)
            label[max_iou < 0.3] = 0
            label[max_iou >= 0.7] = 1
            best_per_gt = ious.argmax(dim=0)
            label[best_per_gt] = 1
        labels.append(label)
        matched_indices.append(matched_idx)
        clipped_anchors.append(anchors_in_image)
    return labels, matched_indices, clipped_anchors


def rpn_losses(
    objectness: torch.Tensor,
    pred_bbox_deltas: torch.Tensor,
    anchors: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    image_sizes: list[tuple[int, int]],
    batch_size: int = 256,
    positive_fraction: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        labels, matched_indices, clipped_anchors = assign_rpn_targets(anchors, targets, image_sizes)
    objectness_loss = objectness.new_zeros(())
    box_loss = pred_bbox_deltas.new_zeros(())
    for image_index, labels_per_image in enumerate(labels):
        sampled = sample_labels(labels_per_image, batch_size, positive_fraction)
        sampled_labels = labels_per_image[sampled]
        objectness_loss = objectness_loss + F.binary_cross_entropy_with_logits(
            objectness[image_index][sampled],
            sampled_labels,
        )
        positives = sampled[sampled_labels == 1]
        if positives.numel():
            gt_boxes = targets[image_index]["boxes"]
            target_deltas = encode_boxes(
                gt_boxes[matched_indices[image_index][positives]],
                clipped_anchors[image_index][positives],
            )
            box_loss = box_loss + F.smooth_l1_loss(
                pred_bbox_deltas[image_index][positives],
                target_deltas,
                beta=1 / 9,
                reduction="sum",
            ) / max(positives.numel(), 1)
    normalizer = max(len(targets), 1)
    return objectness_loss / normalizer, box_loss / normalizer


def generate_proposals(
    objectness: torch.Tensor,
    pred_bbox_deltas: torch.Tensor,
    anchors: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    pre_nms_top_n: int,
    post_nms_top_n: int,
    nms_threshold: float = 0.7,
    num_anchors_per_level: list[int] | None = None,
) -> list[torch.Tensor]:
    proposals = []
    for image_index, image_size in enumerate(image_sizes):
        scores = objectness[image_index].sigmoid()
        deltas = pred_bbox_deltas[image_index]
        if num_anchors_per_level is None:
            num_top = min(pre_nms_top_n, scores.numel())
            top_scores, top_idx = scores.topk(num_top)
            boxes = decode_boxes(deltas[top_idx], anchors[top_idx])
        else:
            score_levels = scores.split(num_anchors_per_level, dim=0)
            delta_levels = deltas.split(num_anchors_per_level, dim=0)
            anchor_levels = anchors.split(num_anchors_per_level, dim=0)
            top_scores_per_level = []
            boxes_per_level = []
            for scores_level, deltas_level, anchors_level in zip(score_levels, delta_levels, anchor_levels):
                num_top = min(pre_nms_top_n, scores_level.numel())
                scores_top, top_idx = scores_level.topk(num_top)
                boxes_top = decode_boxes(deltas_level[top_idx], anchors_level[top_idx])
                top_scores_per_level.append(scores_top)
                boxes_per_level.append(boxes_top)
            top_scores = torch.cat(top_scores_per_level, dim=0)
            boxes = torch.cat(boxes_per_level, dim=0)
        boxes = clip_boxes_to_image(boxes, image_size)
        keep = remove_small_boxes(boxes, min_size=2)
        boxes, top_scores = boxes[keep], top_scores[keep]
        keep = torchvision_nms(boxes, top_scores, nms_threshold)[:post_nms_top_n]
        proposals.append(boxes[keep].detach())
    return proposals


def assign_roi_targets(
    proposals: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    batch_size: int = 512,
    positive_fraction: float = 0.25,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    sampled_proposals = []
    labels = []
    regression_targets = []
    for proposals_per_image, target in zip(proposals, targets):
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]
        proposals_per_image = torch.cat([proposals_per_image, gt_boxes], dim=0)
        if gt_boxes.numel() == 0:
            labels_per_image = torch.zeros(
                (proposals_per_image.shape[0],),
                dtype=torch.long,
                device=proposals_per_image.device,
            )
            matched_gt = torch.zeros_like(proposals_per_image)
        else:
            ious = box_iou(proposals_per_image, gt_boxes)
            max_iou, matched_idx = ious.max(dim=1)
            labels_per_image = gt_labels[matched_idx]
            labels_per_image[max_iou < 0.5] = 0
            matched_gt = gt_boxes[matched_idx]
        sampling_labels = (labels_per_image > 0).float().where(
            labels_per_image >= 0,
            torch.tensor(-1.0, device=labels_per_image.device),
        )
        sampled = sample_labels(sampling_labels, batch_size, positive_fraction)
        sampled_proposals.append(proposals_per_image[sampled])
        labels.append(labels_per_image[sampled].clamp(min=0))
        regression_targets.append(encode_boxes(matched_gt[sampled], proposals_per_image[sampled]))
    return sampled_proposals, labels, regression_targets


def roi_losses(
    class_logits: torch.Tensor,
    box_regression: torch.Tensor,
    labels: list[torch.Tensor],
    regression_targets: list[torch.Tensor],
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels_cat = torch.cat(labels, dim=0)
    regression_targets_cat = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels_cat)
    positive = torch.where(labels_cat > 0)[0]
    if positive.numel() == 0:
        return classification_loss, box_regression.new_zeros(())
    box_regression = box_regression.reshape(box_regression.shape[0], num_classes, 4)
    box_loss = F.smooth_l1_loss(
        box_regression[positive, labels_cat[positive]],
        regression_targets_cat[positive],
        beta=1 / 9,
        reduction="sum",
    ) / labels_cat.numel()
    return classification_loss, box_loss
