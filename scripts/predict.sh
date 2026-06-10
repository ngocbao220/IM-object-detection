#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-${WANDB_RUN_NAME:-retina-resnet101-fpn}}"

MODEL_IMPL="${MODEL_IMPL:-retina}" \
WANDB_RUN_NAME="${RUN_NAME}" \
CHECKPOINT="${CHECKPOINT:-./saved_results/${RUN_NAME}/checkpoints/best_model.pth}" \
PREDICT_IMAGE_DIR="${PREDICT_IMAGE_DIR:-./public/val/images}" \
PREDICTIONS_OUTPUT="${PREDICTIONS_OUTPUT:-./saved_results/${RUN_NAME}/predictions.json}" \
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.05}" \
NMS_THRESHOLD="${NMS_THRESHOLD:-0.5}" \
RETINA_TOPK_CANDIDATES="${RETINA_TOPK_CANDIDATES:-2000}" \
RETINA_MAX_DETECTIONS="${RETINA_MAX_DETECTIONS:-300}" \
YOLO_TOPK_CANDIDATES="${YOLO_TOPK_CANDIDATES:-2000}" \
YOLO_MAX_DETECTIONS="${YOLO_MAX_DETECTIONS:-300}" \
GPU="${GPU:-0}" \
bash script.sh predict
