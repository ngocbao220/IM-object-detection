#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1
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

# =========================
# 1. Dataset
# =========================
KAGGLE_DATASET_SLUG="${KAGGLE_DATASET_SLUG:-ngocbaotrinhtuan/object-detection/final_public.zip}"
LOCAL_DATASET_ZIP="${LOCAL_DATASET_ZIP:-}"

TRAIN_DATA="${TRAIN_DATA:-./public/annotations/train.json}"
VAL_DATA="${VAL_DATA:-./public/annotations/val.json}"
TRAIN_IMAGE_DIR="${TRAIN_IMAGE_DIR:-./public/train/images}"
VAL_IMAGE_DIR="${VAL_IMAGE_DIR:-./public/val/images}"

# =========================
# 2. Run Outputs
# =========================
SAVED_RESULTS_DIR="${SAVED_RESULTS_DIR:-./saved_results}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-faster-custom}"
RUN_RESULTS_DIR="${SAVED_RESULTS_DIR}/${WANDB_RUN_NAME}"

CHECKPOINT="${CHECKPOINT:-${RUN_RESULTS_DIR}/checkpoints/best_model.pth}"
RESUME_FROM="${RESUME_FROM:-}"
RESUME_LAST="${RESUME_LAST:-0}"

PREDICT_IMAGE_DIR="${PREDICT_IMAGE_DIR:-./public/val/images}"
PREDICTIONS_OUTPUT="${PREDICTIONS_OUTPUT:-${RUN_RESULTS_DIR}/predictions.json}"
RAW_PREDICTIONS_OUTPUT="${RAW_PREDICTIONS_OUTPUT:-${RUN_RESULTS_DIR}/predictions_raw.json}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${RUN_RESULTS_DIR}/evaluation.json}"
ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-${RUN_RESULTS_DIR}/analysis}"
THRESHOLD_TUNING_OUTPUT="${THRESHOLD_TUNING_OUTPUT:-${RUN_RESULTS_DIR}/threshold_tuning.json}"

# =========================
# 3. Important Model Params
# =========================
# MODEL_IMPL accepts: torchvision, custom, retina, yolo. CUSTOM_MODEL=0/1 is kept for compatibility.
MODEL_IMPL="${MODEL_IMPL:-custom}"
CUSTOM_MODEL="${CUSTOM_MODEL:-1}"
BACKBONE="${BACKBONE:-resnet50}"
PRETRAINED_BACKBONE="${PRETRAINED_BACKBONE:-1}"
TRAINABLE_BACKBONE_LAYERS="${TRAINABLE_BACKBONE_LAYERS:-2}"
MIN_SIZE="${MIN_SIZE:-512}"
MAX_SIZE="${MAX_SIZE:-768}"
ROI_DROPOUT="${ROI_DROPOUT:-0.2}"
RETINA_TOPK_CANDIDATES="${RETINA_TOPK_CANDIDATES:-1000}"
RETINA_MAX_DETECTIONS="${RETINA_MAX_DETECTIONS:-300}"
YOLO_TOPK_CANDIDATES="${YOLO_TOPK_CANDIDATES:-1000}"
YOLO_MAX_DETECTIONS="${YOLO_MAX_DETECTIONS:-300}"
# Empty anchor values use model defaults. Custom FPN default: sizes=32,64,128,256,512 and ratios=0.5,1.0,2.0.
ANCHOR_SIZES="${ANCHOR_SIZES:-}"
ANCHOR_RATIOS="${ANCHOR_RATIOS:-}"
TRAIN_PRE_NMS_TOP_N="${TRAIN_PRE_NMS_TOP_N:-2000}"
TRAIN_POST_NMS_TOP_N="${TRAIN_POST_NMS_TOP_N:-2000}"
TEST_PRE_NMS_TOP_N="${TEST_PRE_NMS_TOP_N:-1000}"
TEST_POST_NMS_TOP_N="${TEST_POST_NMS_TOP_N:-1000}"
CUSTOM_FIXED_BATCH_SHAPE="${CUSTOM_FIXED_BATCH_SHAPE:-1}"

# =========================
# 4. Training Params
# =========================
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-0.001}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"

# LR_SCHEDULER accepts: multistep, cosine, plateau.
LR_SCHEDULER="${LR_SCHEDULER:-plateau}"
LR_MILESTONES="${LR_MILESTONES:-15,25}"
LR_GAMMA="${LR_GAMMA:-0.1}"
MIN_LR="${MIN_LR:-0.00001}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-3}"
PLATEAU_FACTOR="${PLATEAU_FACTOR:-0.5}"

EARLY_STOPPING="${EARLY_STOPPING:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.01}"

# =========================
# 5. Data Sampling/Augment
# =========================
AUGMENTATION="${AUGMENTATION:-1}"
HORIZONTAL_FLIP_PROBABILITY="${HORIZONTAL_FLIP_PROBABILITY:-0.5}"
COLOR_JITTER_PROBABILITY="${COLOR_JITTER_PROBABILITY:-0.2}"
GRAYSCALE_PROBABILITY="${GRAYSCALE_PROBABILITY:-0.03}"
SCALE_JITTER_PROBABILITY="${SCALE_JITTER_PROBABILITY:-0.4}"
SCALE_JITTER_MIN="${SCALE_JITTER_MIN:-0.9}"
SCALE_JITTER_MAX="${SCALE_JITTER_MAX:-1.15}"
SAFE_CROP_PROBABILITY="${SAFE_CROP_PROBABILITY:-0.2}"
SAFE_CROP_MIN_SCALE="${SAFE_CROP_MIN_SCALE:-0.75}"
SAFE_CROP_MIN_VISIBILITY="${SAFE_CROP_MIN_VISIBILITY:-0.5}"
BLUR_PROBABILITY="${BLUR_PROBABILITY:-0.08}"
BLUR_KERNEL_SIZE="${BLUR_KERNEL_SIZE:-5}"
NOISE_PROBABILITY="${NOISE_PROBABILITY:-0.08}"
NOISE_STD="${NOISE_STD:-0.02}"

SAMPLER_STRATEGY="${SAMPLER_STRATEGY:-class_small_balanced}"
SMALL_OBJECT_BOOST="${SMALL_OBJECT_BOOST:-1.5}"
SMALL_OBJECT_THRESHOLD="${SMALL_OBJECT_THRESHOLD:-0.01}"
EMPTY_IMAGE_WEIGHT="${EMPTY_IMAGE_WEIGHT:-0.5}"
OVERSAMPLE_CLASS="${OVERSAMPLE_CLASS:-}"
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-1.0}"

# =========================
# 6. Inference/Evaluation
# =========================
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.5}"
NMS_THRESHOLD="${NMS_THRESHOLD:-0.5}"
CONFIDENCE_THRESHOLDS="${CONFIDENCE_THRESHOLDS:-0.2,0.3,0.4,0.5,0.6,0.7}"
NMS_THRESHOLDS="${NMS_THRESHOLDS:-0.3,0.4,0.5,0.6,0.7}"

# =========================
# 7. Runtime/Logging
# =========================
GPU="${GPU:-0}"
GPUS="${GPUS:-}"
USE_WANDB="${USE_WANDB:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
FULL_COCO_METRICS_INTERVAL="${FULL_COCO_METRICS_INTERVAL:-0}"
EMPTY_CACHE_INTERVAL="${EMPTY_CACHE_INTERVAL:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
export PYTORCH_CUDA_ALLOC_CONF

# =========================
# 8. Experiment Suites
# =========================
ABLATION_RESULTS_DIR="${ABLATION_RESULTS_DIR:-./saved_results/augmentation_ablation}"
ABLATION_EPOCHS="${ABLATION_EPOCHS:-30}"

if [[ -n "${MODEL_IMPL}" ]]; then
  case "${MODEL_IMPL}" in
    custom)
      CUSTOM_MODEL=1
      ;;
    torchvision)
      CUSTOM_MODEL=0
      ;;
    retina)
      CUSTOM_MODEL=0
      ;;
    yolo)
      CUSTOM_MODEL=0
      ;;
    *)
      echo "MODEL_IMPL must be one of 'torchvision', 'custom', 'retina', 'yolo'. Got: ${MODEL_IMPL}"
      exit 1
      ;;
  esac
fi

install() {
  echo "============ Dependency Installation ============"
  echo "pytorch_index_url: ${PYTORCH_INDEX_URL}"
  echo "requirements: requirements.txt"
  echo "================================================="
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
  resume_checkpoint="${RESUME_FROM}"
  resolved_model_impl="${MODEL_IMPL}"
  if [[ -z "${resolved_model_impl}" ]]; then
    if [[ "${CUSTOM_MODEL}" == "1" ]]; then
      resolved_model_impl="custom"
    else
      resolved_model_impl="torchvision"
    fi
  fi
  if [[ "${RESUME_LAST}" == "1" && -z "${resume_checkpoint}" ]]; then
    resume_checkpoint="${RUN_RESULTS_DIR}/checkpoints/last_model.pth"
  fi

  train_args=(
    --train_data "${TRAIN_DATA}"
    --val_data "${VAL_DATA}"
    --image_dir "${TRAIN_IMAGE_DIR}"
    --val_image_dir "${VAL_IMAGE_DIR}"
    --saved_results_dir "${SAVED_RESULTS_DIR}"
    --wandb_run_name "${WANDB_RUN_NAME}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --log_interval "${LOG_INTERVAL}"
    --full_coco_metrics_interval "${FULL_COCO_METRICS_INTERVAL}"
    --empty_cache_interval "${EMPTY_CACHE_INTERVAL}"
    --lr "${LR}"
    --momentum "${MOMENTUM}"
    --weight_decay "${WEIGHT_DECAY}"
    --grad_clip_norm "${GRAD_CLIP_NORM}"
    --lr_scheduler "${LR_SCHEDULER}"
    --lr_milestones "${LR_MILESTONES}"
    --lr_gamma "${LR_GAMMA}"
    --min_lr "${MIN_LR}"
    --plateau_patience "${PLATEAU_PATIENCE}"
    --plateau_factor "${PLATEAU_FACTOR}"
    --score_threshold "${SCORE_THRESHOLD}"
    --model_impl "${resolved_model_impl}"
    --backbone "${BACKBONE}"
    --trainable_backbone_layers "${TRAINABLE_BACKBONE_LAYERS}"
    --min_size "${MIN_SIZE}"
    --max_size "${MAX_SIZE}"
    --roi_dropout "${ROI_DROPOUT}"
    --retina_topk_candidates "${RETINA_TOPK_CANDIDATES}"
    --retina_max_detections "${RETINA_MAX_DETECTIONS}"
    --yolo_topk_candidates "${YOLO_TOPK_CANDIDATES}"
    --yolo_max_detections "${YOLO_MAX_DETECTIONS}"
    --train_pre_nms_top_n "${TRAIN_PRE_NMS_TOP_N}"
    --train_post_nms_top_n "${TRAIN_POST_NMS_TOP_N}"
    --test_pre_nms_top_n "${TEST_PRE_NMS_TOP_N}"
    --test_post_nms_top_n "${TEST_POST_NMS_TOP_N}"
    --horizontal_flip_probability "${HORIZONTAL_FLIP_PROBABILITY}"
    --color_jitter_probability "${COLOR_JITTER_PROBABILITY}"
    --grayscale_probability "${GRAYSCALE_PROBABILITY}"
    --scale_jitter_probability "${SCALE_JITTER_PROBABILITY}"
    --scale_jitter_min "${SCALE_JITTER_MIN}"
    --scale_jitter_max "${SCALE_JITTER_MAX}"
    --safe_crop_probability "${SAFE_CROP_PROBABILITY}"
    --safe_crop_min_scale "${SAFE_CROP_MIN_SCALE}"
    --safe_crop_min_visibility "${SAFE_CROP_MIN_VISIBILITY}"
    --blur_probability "${BLUR_PROBABILITY}"
    --blur_kernel_size "${BLUR_KERNEL_SIZE}"
    --noise_probability "${NOISE_PROBABILITY}"
    --noise_std "${NOISE_STD}"
    --sampler_strategy "${SAMPLER_STRATEGY}"
    --small_object_boost "${SMALL_OBJECT_BOOST}"
    --small_object_threshold "${SMALL_OBJECT_THRESHOLD}"
    --empty_image_weight "${EMPTY_IMAGE_WEIGHT}"
    --oversample_factor "${OVERSAMPLE_FACTOR}"
    --early_stopping_patience "${EARLY_STOPPING_PATIENCE}"
    --early_stopping_min_delta "${EARLY_STOPPING_MIN_DELTA}"
  )

  if [[ -n "${GPUS}" ]]; then
    train_args+=(--gpus "${GPUS}")
  elif [[ -n "${GPU}" ]]; then
    train_args+=(--gpu "${GPU}")
  fi

  if [[ -n "${OVERSAMPLE_CLASS}" ]]; then
    train_args+=(--oversample_class "${OVERSAMPLE_CLASS}")
  fi

  if [[ -n "${resume_checkpoint}" ]]; then
    train_args+=(--resume_from "${resume_checkpoint}")
  fi

  if [[ "${CUSTOM_MODEL}" == "1" ]]; then
    train_args+=(--custom)
    if [[ "${CUSTOM_FIXED_BATCH_SHAPE}" == "1" ]]; then
      train_args+=(--fixed_batch_shape)
    else
      train_args+=(--no-fixed_batch_shape)
    fi
  fi

  if [[ -n "${ANCHOR_SIZES}" ]]; then
    train_args+=(--anchor_sizes "${ANCHOR_SIZES}")
  fi

  if [[ -n "${ANCHOR_RATIOS}" ]]; then
    train_args+=(--anchor_ratios "${ANCHOR_RATIOS}")
  fi

  if [[ "${USE_WANDB}" == "1" ]]; then
    train_args+=(--use_wandb)
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
  resolved_model_impl="${MODEL_IMPL}"
  if [[ -z "${resolved_model_impl}" ]]; then
    if [[ "${CUSTOM_MODEL}" == "1" ]]; then
      resolved_model_impl="custom"
    else
      resolved_model_impl="torchvision"
    fi
  fi
  predict_args=(
    --image_dir "${PREDICT_IMAGE_DIR}"
    --output "${PREDICTIONS_OUTPUT}"
    --checkpoint "${CHECKPOINT}"
    --score_threshold "${SCORE_THRESHOLD}"
    --nms_threshold "${NMS_THRESHOLD}"
    --model_impl "${resolved_model_impl}"
    --backbone "${BACKBONE}"
    --min_size "${MIN_SIZE}"
    --max_size "${MAX_SIZE}"
    --retina_topk_candidates "${RETINA_TOPK_CANDIDATES}"
    --retina_max_detections "${RETINA_MAX_DETECTIONS}"
    --yolo_topk_candidates "${YOLO_TOPK_CANDIDATES}"
    --yolo_max_detections "${YOLO_MAX_DETECTIONS}"
  )

  if [[ -n "${ANCHOR_SIZES}" ]]; then
    predict_args+=(--anchor_sizes "${ANCHOR_SIZES}")
  fi

  if [[ -n "${ANCHOR_RATIOS}" ]]; then
    predict_args+=(--anchor_ratios "${ANCHOR_RATIOS}")
  fi

  if [[ "${CUSTOM_MODEL}" == "1" ]]; then
    predict_args+=(--custom)
  fi

  python predict.py "${predict_args[@]}"
}

predict_raw() {
  resolved_model_impl="${MODEL_IMPL}"
  if [[ -z "${resolved_model_impl}" ]]; then
    if [[ "${CUSTOM_MODEL}" == "1" ]]; then
      resolved_model_impl="custom"
    else
      resolved_model_impl="torchvision"
    fi
  fi

  raw_score_threshold="${RAW_SCORE_THRESHOLD:-0.0}"
  raw_nms_threshold="${RAW_NMS_THRESHOLD:-0.99}"
  raw_retina_max_detections="${RAW_RETINA_MAX_DETECTIONS:-1000}"
  raw_retina_topk_candidates="${RAW_RETINA_TOPK_CANDIDATES:-2000}"
  raw_yolo_max_detections="${RAW_YOLO_MAX_DETECTIONS:-1000}"
  raw_yolo_topk_candidates="${RAW_YOLO_TOPK_CANDIDATES:-2000}"

  echo "============ Raw Prediction Export ============"
  echo "image_dir: ${PREDICT_IMAGE_DIR}"
  echo "checkpoint: ${CHECKPOINT}"
  echo "output: ${RAW_PREDICTIONS_OUTPUT}"
  echo "model_impl: ${resolved_model_impl}"
  echo "score_threshold: ${raw_score_threshold}"
  echo "nms_threshold: ${raw_nms_threshold}"
  echo "retina_topk_candidates: ${raw_retina_topk_candidates}"
  echo "retina_max_detections: ${raw_retina_max_detections}"
  echo "yolo_topk_candidates: ${raw_yolo_topk_candidates}"
  echo "yolo_max_detections: ${raw_yolo_max_detections}"
  echo "==============================================="

  predict_args=(
    --image_dir "${PREDICT_IMAGE_DIR}"
    --output "${RAW_PREDICTIONS_OUTPUT}"
    --checkpoint "${CHECKPOINT}"
    --score_threshold "${raw_score_threshold}"
    --nms_threshold "${raw_nms_threshold}"
    --model_impl "${resolved_model_impl}"
    --backbone "${BACKBONE}"
    --min_size "${MIN_SIZE}"
    --max_size "${MAX_SIZE}"
    --retina_topk_candidates "${raw_retina_topk_candidates}"
    --retina_max_detections "${raw_retina_max_detections}"
    --yolo_topk_candidates "${raw_yolo_topk_candidates}"
    --yolo_max_detections "${raw_yolo_max_detections}"
  )

  if [[ -n "${ANCHOR_SIZES}" ]]; then
    predict_args+=(--anchor_sizes "${ANCHOR_SIZES}")
  fi

  if [[ -n "${ANCHOR_RATIOS}" ]]; then
    predict_args+=(--anchor_ratios "${ANCHOR_RATIOS}")
  fi

  if [[ "${CUSTOM_MODEL}" == "1" ]]; then
    predict_args+=(--custom)
  fi

  python predict.py "${predict_args[@]}"
}

evaluate() {
  echo "============ Prediction Evaluation ============"
  echo "ground_truth: ${VAL_DATA}"
  echo "predictions: ${PREDICTIONS_OUTPUT}"
  echo "output: ${EVAL_OUTPUT}"
  echo "iou_threshold: 0.5"
  echo "==============================================="
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

  SAVED_RESULTS_DIR="${ABLATION_RESULTS_DIR}" \
  EPOCHS="${ABLATION_EPOCHS}" \
  AUGMENTATION="${augmentation}" \
  HORIZONTAL_FLIP_PROBABILITY="${flip_probability}" \
  COLOR_JITTER_PROBABILITY="${jitter_probability}" \
  GRAYSCALE_PROBABILITY="${grayscale_probability}" \
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
  echo "============ Augmentation Summary ============"
  echo "results_dir: ${ABLATION_RESULTS_DIR}"
  echo "=============================================="
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
