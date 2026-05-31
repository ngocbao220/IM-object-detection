#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash script.sh install
#   bash script.sh download
#   bash script.sh train
#   bash script.sh predict
#   bash script.sh predict-raw
#   bash script.sh evaluate
#   bash script.sh analyze
#   bash script.sh tune-thresholds
#   bash script.sh augment-ablation
#   bash script.sh augment-summary
#   bash script.sh all

KAGGLE_DATASET_SLUG="${KAGGLE_DATASET_SLUG:-ngocbaotrinhtuan/object-detection/final_public.zip}"
LOCAL_DATASET_ZIP="${LOCAL_DATASET_ZIP:-}"

TRAIN_DATA="${TRAIN_DATA:-./public/annotations/train.json}"
VAL_DATA="${VAL_DATA:-./public/annotations/val.json}"
TRAIN_IMAGE_DIR="${TRAIN_IMAGE_DIR:-./public/train/images}"
VAL_IMAGE_DIR="${VAL_IMAGE_DIR:-./public/val/images}"
SAVED_RESULTS_DIR="${SAVED_RESULTS_DIR:-./saved_results}"
CHECKPOINT="${CHECKPOINT:-${SAVED_RESULTS_DIR}/checkpoints/best_model.pth}"

PREDICT_IMAGE_DIR="${PREDICT_IMAGE_DIR:-./public/val/images}"
PREDICTIONS_OUTPUT="${PREDICTIONS_OUTPUT:-${SAVED_RESULTS_DIR}/predictions.json}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${SAVED_RESULTS_DIR}/evaluation.json}"
RAW_PREDICTIONS_OUTPUT="${RAW_PREDICTIONS_OUTPUT:-${SAVED_RESULTS_DIR}/predictions_raw.json}"
ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-${SAVED_RESULTS_DIR}/analysis}"
THRESHOLD_TUNING_OUTPUT="${THRESHOLD_TUNING_OUTPUT:-${SAVED_RESULTS_DIR}/threshold_tuning.json}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
LR="${LR:-0.005}"
LR_MILESTONES="${LR_MILESTONES:-15,25}"
LR_GAMMA="${LR_GAMMA:-0.1}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.5}"
NMS_THRESHOLD="${NMS_THRESHOLD:-0.5}"
CONFIDENCE_THRESHOLDS="${CONFIDENCE_THRESHOLDS:-0.2,0.3,0.4,0.5,0.6,0.7}"
NMS_THRESHOLDS="${NMS_THRESHOLDS:-0.3,0.4,0.5,0.6,0.7}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
GPU="${GPU:-}"
GPUS="${GPUS:-}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
PRETRAINED_BACKBONE="${PRETRAINED_BACKBONE:-1}"
AUGMENTATION="${AUGMENTATION:-1}"
HORIZONTAL_FLIP_PROBABILITY="${HORIZONTAL_FLIP_PROBABILITY:-0.5}"
COLOR_JITTER_PROBABILITY="${COLOR_JITTER_PROBABILITY:-0.3}"
GRAYSCALE_PROBABILITY="${GRAYSCALE_PROBABILITY:-0.05}"
EARLY_STOPPING="${EARLY_STOPPING:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-7}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.001}"
ABLATION_RESULTS_DIR="${ABLATION_RESULTS_DIR:-./saved_results/augmentation_ablation}"
ABLATION_EPOCHS="${ABLATION_EPOCHS:-30}"

install() {
  python -m pip install --upgrade pip
  python -m pip install torch torchvision --index-url "${PYTORCH_INDEX_URL}"
  python -m pip install -r requirements.txt
}

download() {
  if [[ -n "${LOCAL_DATASET_ZIP}" ]]; then
    python utils/helper.py \
      --download_dataset \
      --local_zip "${LOCAL_DATASET_ZIP}"
  else
    python utils/helper.py \
      --download_dataset \
      --dataset_slug "${KAGGLE_DATASET_SLUG}"
  fi
}

train() {
  train_args=(
    --train_data "${TRAIN_DATA}"
    --val_data "${VAL_DATA}"
    --image_dir "${TRAIN_IMAGE_DIR}"
    --val_image_dir "${VAL_IMAGE_DIR}"
    --saved_results_dir "${SAVED_RESULTS_DIR}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --log_interval "${LOG_INTERVAL}"
    --lr "${LR}"
    --lr_milestones "${LR_MILESTONES}"
    --lr_gamma "${LR_GAMMA}"
    --score_threshold "${SCORE_THRESHOLD}"
    --horizontal_flip_probability "${HORIZONTAL_FLIP_PROBABILITY}"
    --color_jitter_probability "${COLOR_JITTER_PROBABILITY}"
    --grayscale_probability "${GRAYSCALE_PROBABILITY}"
    --early_stopping_patience "${EARLY_STOPPING_PATIENCE}"
    --early_stopping_min_delta "${EARLY_STOPPING_MIN_DELTA}"
  )

  if [[ -n "${GPUS}" ]]; then
    train_args+=(--gpus "${GPUS}")
  elif [[ -n "${GPU}" ]]; then
    train_args+=(--gpu "${GPU}")
  fi

  if [[ "${USE_WANDB}" == "1" ]]; then
    train_args+=(--use_wandb)
    if [[ -n "${WANDB_RUN_NAME}" ]]; then
      train_args+=(--wandb_run_name "${WANDB_RUN_NAME}")
    fi
  fi

  if [[ "${PRETRAINED_BACKBONE}" == "1" ]]; then
    train_args+=(--pretrained_backbone)
  else
    train_args+=(--no-pretrained_backbone)
  fi

  if [[ "${AUGMENTATION}" == "1" ]]; then
    train_args+=(--augmentation)
  else
    train_args+=(--no-augmentation)
  fi

  if [[ "${EARLY_STOPPING}" == "1" ]]; then
    train_args+=(--early_stopping)
  else
    train_args+=(--no-early_stopping)
  fi

  python train.py "${train_args[@]}"
}

predict() {
  python predict.py \
    --image_dir "${PREDICT_IMAGE_DIR}" \
    --output "${PREDICTIONS_OUTPUT}" \
    --checkpoint "${CHECKPOINT}" \
    --score_threshold "${SCORE_THRESHOLD}" \
    --nms_threshold "${NMS_THRESHOLD}"
}

predict_raw() {
  python predict.py \
    --image_dir "${PREDICT_IMAGE_DIR}" \
    --output "${RAW_PREDICTIONS_OUTPUT}" \
    --checkpoint "${CHECKPOINT}" \
    --score_threshold 0.01 \
    --nms_threshold 1.0
}

evaluate() {
  python public/tools/evaluate_predictions.py \
    --ground_truth "${VAL_DATA}" \
    --predictions "${PREDICTIONS_OUTPUT}" \
    --output "${EVAL_OUTPUT}"
}

analyze() {
  python -m utils.analyze_predictions \
    --ground_truth "${VAL_DATA}" \
    --predictions "${PREDICTIONS_OUTPUT}" \
    --image_dir "${VAL_IMAGE_DIR}" \
    --output_dir "${ANALYSIS_OUTPUT_DIR}" \
    --max_visualizations 50
}

tune_thresholds() {
  python -m utils.tune_thresholds \
    --ground_truth "${VAL_DATA}" \
    --predictions "${RAW_PREDICTIONS_OUTPUT}" \
    --output "${THRESHOLD_TUNING_OUTPUT}" \
    --confidence_thresholds "${CONFIDENCE_THRESHOLDS}" \
    --nms_thresholds "${NMS_THRESHOLDS}"
}

run_augmentation_experiment() {
  experiment_name="$1"
  augmentation="$2"
  flip_probability="$3"
  jitter_probability="$4"
  grayscale_probability="$5"

  echo "============================================================"
  echo "Running augmentation experiment: ${experiment_name}"
  echo "augmentation=${augmentation} flip=${flip_probability} jitter=${jitter_probability} grayscale=${grayscale_probability}"
  echo "results=${ABLATION_RESULTS_DIR}/${experiment_name}"
  echo "============================================================"

  SAVED_RESULTS_DIR="${ABLATION_RESULTS_DIR}/${experiment_name}" \
  EPOCHS="${ABLATION_EPOCHS}" \
  AUGMENTATION="${augmentation}" \
  HORIZONTAL_FLIP_PROBABILITY="${flip_probability}" \
  COLOR_JITTER_PROBABILITY="${jitter_probability}" \
  GRAYSCALE_PROBABILITY="${grayscale_probability}" \
  EARLY_STOPPING=0 \
  WANDB_RUN_NAME="${experiment_name}" \
  train
}

augment_ablation() {
  run_augmentation_experiment "00_no_augmentation" 0 0.0 0.0 0.0
  run_augmentation_experiment "01_horizontal_flip" 1 0.5 0.0 0.0
  run_augmentation_experiment "02_color_jitter" 1 0.0 0.3 0.0
  run_augmentation_experiment "03_grayscale" 1 0.0 0.0 0.05
  run_augmentation_experiment "04_all_augmentations" 1 0.5 0.3 0.05
  summarize_augmentation_ablation
}

summarize_augmentation_ablation() {
  python utils/summarize_augmentation_ablation.py \
    --results_dir "${ABLATION_RESULTS_DIR}"
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
  predict-raw)
    predict_raw
    ;;
  evaluate)
    evaluate
    ;;
  analyze)
    analyze
    ;;
  tune-thresholds)
    tune_thresholds
    ;;
  augment-ablation)
    augment_ablation
    ;;
  augment-summary)
    summarize_augmentation_ablation
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
    analyze
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
