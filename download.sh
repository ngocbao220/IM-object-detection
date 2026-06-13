#!/usr/bin/env bash
set -euo pipefail

# Simple download helper.
#
# Model checkpoint:
#   bash download.sh --model MODEL_NAME=retina-baseline
#   bash download.sh --model MODEL_NAME=retina-baseline MODEL_VERSION=3
#
# Public dataset:
#   bash download.sh --dataset

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
SAVED_RESULTS_DIR="${SAVED_RESULTS_DIR:-${PROJECT_ROOT}/saved_results}"
KAGGLE_OWNER="${KAGGLE_OWNER:-${KAGGLE_USERNAME:-ngocbaotrinhtuan}}"
HF_REPO_ID="${HF_REPO_ID:-ngocbao05/object-detection-checkpoints}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
KAGGLE_MODEL_SLUG="${KAGGLE_MODEL_SLUG:-object-detection-checkpoints}"
KAGGLE_FRAMEWORK="${KAGGLE_FRAMEWORK:-PyTorch}"
KAGGLE_DATASET_SLUG="${KAGGLE_DATASET_SLUG:-${KAGGLE_OWNER}/object-detection-public}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_VERSION="${MODEL_VERSION:-latest}"
MODE=""

usage() {
  cat <<'EOF'
Usage:
  bash download.sh --model MODEL_NAME=<run_name> [MODEL_VERSION=<version|latest>]
  bash download.sh --dataset

Examples:
  bash download.sh --model MODEL_NAME=retina-baseline
  bash download.sh --model MODEL_NAME=retina-baseline MODEL_VERSION=2
  bash download.sh --dataset

Optional env:
  HF_REPO_ID, HF_REPO_TYPE, HF_TOKEN, HUGGINGFACE_TOKEN,
  KAGGLE_OWNER, KAGGLE_DATASET_SLUG, SAVED_RESULTS_DIR
EOF
}

slugify() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//'
}

require_kaggle() {
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "kaggle CLI is not installed. Run: pip install kaggle" >&2
    exit 1
  fi
}

require_huggingface_hub() {
  if ! python -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "huggingface_hub is not installed. Run: pip install huggingface-hub" >&2
    exit 1
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        MODE="model"
        shift
        ;;
      --model=*)
        MODE="model"
        MODEL_NAME="${1#--model=}"
        shift
        ;;
      --dataset)
        MODE="dataset"
        shift
        ;;
      MODEL_NAME=*)
        MODEL_NAME="${1#MODEL_NAME=}"
        shift
        ;;
      MODEL_VERSION=*)
        MODEL_VERSION="${1#MODEL_VERSION=}"
        shift
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage
        exit 1
        ;;
    esac
  done
}

download_model() {
  if [[ -z "${MODEL_NAME}" ]]; then
    echo "Missing MODEL_NAME. Example: bash download.sh --model MODEL_NAME=retina-baseline" >&2
    exit 1
  fi

  require_huggingface_hub
  mkdir -p "${SAVED_RESULTS_DIR}"

  local instance_slug revision tmp_dir target_dir source_dir
  instance_slug="$(slugify "${MODEL_NAME}")"
  revision="${MODEL_VERSION}"
  if [[ "${revision}" == "latest" ]]; then
    revision="main"
  fi

  tmp_dir="${SAVED_RESULTS_DIR}/.hf_model_download/${instance_slug}"
  target_dir="${SAVED_RESULTS_DIR}/${MODEL_NAME}/checkpoints"

  rm -rf "${tmp_dir}"
  mkdir -p "${tmp_dir}" "${target_dir}"

  echo "============ Download Hugging Face Model ============"
  echo "repo_id: ${HF_REPO_ID}"
  echo "path_in_repo: ${instance_slug}"
  echo "revision: ${revision}"
  echo "target: ${target_dir}"
  echo "====================================================="

  HF_REPO_ID="${HF_REPO_ID}" \
  HF_REPO_TYPE="${HF_REPO_TYPE}" \
  HF_MODEL_PATH="${instance_slug}" \
  HF_REVISION="${revision}" \
  HF_LOCAL_DIR="${tmp_dir}" \
  python -c '
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ.get("HF_REPO_TYPE", "model")
path = os.environ["HF_MODEL_PATH"].strip("/")
revision = os.environ.get("HF_REVISION") or "main"
local_dir = os.environ["HF_LOCAL_DIR"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    revision=revision,
    local_dir=local_dir,
    allow_patterns=[f"{path}/**"],
    token=token,
)
print(f"Downloaded {repo_id}/{path} to {local_dir}")
'

  source_dir="${tmp_dir}/${instance_slug}/checkpoints"
  if [[ -d "${source_dir}" ]]; then
    cp -f "${source_dir}/"*.pth "${target_dir}/" 2>/dev/null || true
  else
    find "${tmp_dir}/${instance_slug}" -maxdepth 4 -type f -name '*.pth' -exec cp -f {} "${target_dir}/" \;
  fi

  if [[ ! -f "${target_dir}/best_model.pth" && ! -f "${target_dir}/last_model.pth" ]]; then
    echo "Downloaded files, but no best_model.pth/last_model.pth found under ${tmp_dir}/${instance_slug}" >&2
    exit 1
  fi

  echo "Done. Checkpoints are in ${target_dir}"
}

download_dataset() {
  require_kaggle
  local tmp_dir="${PROJECT_ROOT}/.kaggle_download"
  rm -rf "${tmp_dir}"
  mkdir -p "${tmp_dir}"

  echo "============ Download Kaggle Dataset ============"
  echo "dataset: ${KAGGLE_DATASET_SLUG}"
  echo "target: ${PROJECT_ROOT}"
  echo "================================================="

  kaggle datasets download -d "${KAGGLE_DATASET_SLUG}" -p "${tmp_dir}" --unzip

  if [[ -d "${tmp_dir}/public" ]]; then
    rm -rf "${PROJECT_ROOT}/public"
    mv "${tmp_dir}/public" "${PROJECT_ROOT}/public"
  elif [[ -d "${tmp_dir}/extracted/public" ]]; then
    rm -rf "${PROJECT_ROOT}/public"
    mv "${tmp_dir}/extracted/public" "${PROJECT_ROOT}/public"
  else
    echo "Could not find public/ after dataset download under ${tmp_dir}" >&2
    exit 1
  fi

  rm -rf "${tmp_dir}"
  echo "Done. Dataset is in ${PROJECT_ROOT}/public"
}

parse_args "$@"

case "${MODE}" in
  model)
    download_model
    ;;
  dataset)
    download_dataset
    ;;
  *)
    usage
    exit 1
    ;;
esac
