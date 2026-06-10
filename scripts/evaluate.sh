#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-${WANDB_RUN_NAME:-retina-resnet101-fpn}}"

WANDB_RUN_NAME="${RUN_NAME}" \
VAL_DATA="${VAL_DATA:-./public/annotations/val.json}" \
PREDICTIONS_OUTPUT="${PREDICTIONS_OUTPUT:-./saved_results/${RUN_NAME}/predictions.json}" \
EVAL_OUTPUT="${EVAL_OUTPUT:-./saved_results/${RUN_NAME}/evaluation.json}" \
bash script.sh evaluate
