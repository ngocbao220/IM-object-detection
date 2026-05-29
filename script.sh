#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash script.sh install
#   bash script.sh download
#   bash script.sh train
#   bash script.sh predict
#   bash script.sh evaluate
#   bash script.sh all

KAGGLE_DATASET_SLUG="${KAGGLE_DATASET_SLUG:-ngocbao/object_detection/final_public.zip}"

TRAIN_DATA="${TRAIN_DATA:-./public/annotations/train.json}"
VAL_DATA="${VAL_DATA:-./public/annotations/val.json}"
TRAIN_IMAGE_DIR="${TRAIN_IMAGE_DIR:-./public/train/images}"
VAL_IMAGE_DIR="${VAL_IMAGE_DIR:-./public/val/images}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./models}"
CHECKPOINT="${CHECKPOINT:-./models/best_model.pth}"

PREDICT_IMAGE_DIR="${PREDICT_IMAGE_DIR:-./public/val/images}"
PREDICTIONS_OUTPUT="${PREDICTIONS_OUTPUT:-./predictions.json}"
EVAL_OUTPUT="${EVAL_OUTPUT:-./evaluation.json}"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-0.005}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.5}"

install() {
  python -m pip install -r requirements.txt
}

download() {
  python utils/helper.py \
    --download_dataset \
    --dataset_slug "${KAGGLE_DATASET_SLUG}"
}

train() {
  python train.py \
    --train_data "${TRAIN_DATA}" \
    --val_data "${VAL_DATA}" \
    --image_dir "${TRAIN_IMAGE_DIR}" \
    --val_image_dir "${VAL_IMAGE_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --lr "${LR}"
}

predict() {
  python predict.py \
    --image_dir "${PREDICT_IMAGE_DIR}" \
    --output "${PREDICTIONS_OUTPUT}" \
    --checkpoint "${CHECKPOINT}" \
    --score_threshold "${SCORE_THRESHOLD}"
}

evaluate() {
  python public/tools/evaluate_predictions.py \
    --ground_truth "${VAL_DATA}" \
    --predictions "${PREDICTIONS_OUTPUT}" \
    --output "${EVAL_OUTPUT}"
}

self_test() {
  python utils/metric.py
  python utils/dataset.py \
    --annotation "${TRAIN_DATA}" \
    --image_dir "${TRAIN_IMAGE_DIR}"
  python models/faster_rcnn.py --num_classes 6
}

case "${1:-help}" in
  install)
    install
    ;;
  download)
    download
    ;;
  train)
    train
    ;;
  predict)
    predict
    ;;
  evaluate)
    evaluate
    ;;
  test)
    self_test
    ;;
  all)
    install
    download
    train
    predict
    evaluate
    ;;
  help|--help|-h)
    sed -n '1,22p' "$0"
    ;;
  *)
    echo "Unknown command: $1"
    echo "Run: bash script.sh help"
    exit 1
    ;;
esac
