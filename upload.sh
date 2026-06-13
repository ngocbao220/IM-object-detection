#!/usr/bin/env bash
set -euo pipefail

# Simple upload helper.
#
# Model checkpoint:
#   bash upload.sh --model MODEL_NAME=retina-baseline
#
# Public dataset:
#   bash upload.sh --dataset

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
SAVED_RESULTS_DIR="${SAVED_RESULTS_DIR:-${PROJECT_ROOT}/saved_results}"
EXPORT_ROOT="${EXPORT_ROOT:-${SAVED_RESULTS_DIR}/upload_export}"
KAGGLE_OWNER="${KAGGLE_OWNER:-${KAGGLE_USERNAME:-ngocbaotrinhtuan}}"
HF_REPO_ID="${HF_REPO_ID:-ngocbao05/object-detection-checkpoints}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
HF_REPO_PRIVATE="${HF_REPO_PRIVATE:-0}"

KAGGLE_MODEL_SLUG="${KAGGLE_MODEL_SLUG:-object-detection-checkpoints}"
KAGGLE_MODEL_TITLE="${KAGGLE_MODEL_TITLE:-Object Detection Checkpoints}"
KAGGLE_MODEL_SUBTITLE="${KAGGLE_MODEL_SUBTITLE:-PyTorch checkpoints for object detection experiments}"
KAGGLE_MODEL_PRIVATE="${KAGGLE_MODEL_PRIVATE:-1}"
KAGGLE_FRAMEWORK="${KAGGLE_FRAMEWORK:-PyTorch}"
KAGGLE_MODEL_LICENSE="${KAGGLE_MODEL_LICENSE:-Apache 2.0}"

KAGGLE_DATASET_SLUG="${KAGGLE_DATASET_SLUG:-${KAGGLE_OWNER}/object-detection-public}"
KAGGLE_DATASET_TITLE="${KAGGLE_DATASET_TITLE:-Object Detection Public Dataset}"
KAGGLE_DATASET_LICENSE="${KAGGLE_DATASET_LICENSE:-CC0-1.0}"
PUBLIC_DIR="${PUBLIC_DIR:-${PROJECT_ROOT}/public}"

MODEL_NAME="${MODEL_NAME:-}"
MODE=""

usage() {
  cat <<'EOF'
Usage:
  bash upload.sh --model MODEL_NAME=<run_name>
  bash upload.sh --dataset

Examples:
  bash upload.sh --model MODEL_NAME=retina-baseline
  bash upload.sh --dataset

Optional env:
  HF_REPO_ID, HF_REPO_PRIVATE, HF_TOKEN, HUGGINGFACE_TOKEN,
  KAGGLE_OWNER, KAGGLE_DATASET_SLUG, SAVED_RESULTS_DIR, PUBLIC_DIR
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

run_allow_exists() {
  local output status
  echo "$ $*"
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  printf '%s\n' "${output}"
  if printf '%s' "${output}" | tr '[:upper:]' '[:lower:]' | grep -Eq 'already|exist|409|duplicate'; then
    echo "Resource already exists; continuing."
    return 0
  fi
  if printf '%s' "${output}" | tr '[:upper:]' '[:lower:]' | grep -Eq 'creation error|not found|please upload at least one file'; then
    return 1
  fi
  if [[ "${status}" -eq 0 ]]; then
    return 0
  fi
  return "${status}"
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

latest_file() {
  local directory="$1"
  local glob_pattern="$2"
  local fallback="$3"
  local matches=()
  shopt -s nullglob
  matches=("${directory}"/${glob_pattern})
  shopt -u nullglob
  if [[ "${#matches[@]}" -gt 0 ]]; then
    ls -t "${matches[@]}" | head -n 1
  elif [[ -f "${fallback}" ]]; then
    printf '%s\n' "${fallback}"
  fi
}

write_model_metadata() {
  local dir="$1"
  mkdir -p "${dir}"
  cat > "${dir}/model-metadata.json" <<EOF
{
  "ownerSlug": "${KAGGLE_OWNER}",
  "title": "${KAGGLE_MODEL_TITLE}",
  "slug": "${KAGGLE_MODEL_SLUG}",
  "subtitle": "${KAGGLE_MODEL_SUBTITLE}",
  "isPrivate": $([[ "${KAGGLE_MODEL_PRIVATE}" == "1" ]] && echo true || echo false),
  "description": "# Model Summary\\n\\nObject detection checkpoint collection.\\n\\n# Model Characteristics\\n\\nPyTorch checkpoints exported from saved_results/<run_name>/checkpoints.\\n",
  "publishTime": "",
  "provenanceSources": ""
}
EOF
}

write_model_instance_metadata() {
  local dir="$1"
  local instance_slug="$2"
  mkdir -p "${dir}"
  cat > "${dir}/model-instance-metadata.json" <<EOF
{
  "ownerSlug": "${KAGGLE_OWNER}",
  "modelSlug": "${KAGGLE_MODEL_SLUG}",
  "instanceSlug": "${instance_slug}",
  "framework": "${KAGGLE_FRAMEWORK}",
  "overview": "Checkpoint for ${MODEL_NAME}.",
  "usage": "# Model Format\\n\\nPyTorch .pth checkpoint files.\\n\\n# Model Usage\\n\\nPlace files under saved_results/${MODEL_NAME}/checkpoints/.\\n",
  "licenseName": "${KAGGLE_MODEL_LICENSE}",
  "fineTunable": false,
  "trainingData": [],
  "modelInstanceType": "Unspecified",
  "baseModelInstanceId": 0,
  "externalBaseModelUrl": ""
}
EOF
  cat > "${dir}/README.md" <<EOF
# ${MODEL_NAME}

This Kaggle Model instance stores checkpoint versions for \`${MODEL_NAME}\`.

Actual checkpoint files are uploaded as model instance versions under
\`checkpoints/best_model.pth\` and \`checkpoints/last_model.pth\`.
EOF
}

upload_model() {
  if [[ -z "${MODEL_NAME}" ]]; then
    echo "Missing MODEL_NAME. Example: bash upload.sh --model MODEL_NAME=retina-baseline" >&2
    exit 1
  fi

  require_huggingface_hub

  local checkpoint_dir best_src last_src instance_slug version_dir
  checkpoint_dir="${SAVED_RESULTS_DIR}/${MODEL_NAME}/checkpoints"
  best_src="$(latest_file "${checkpoint_dir}" 'best_model-*.pth' "${checkpoint_dir}/best_model.pth" || true)"
  last_src="$(latest_file "${checkpoint_dir}" 'last_model-*.pth' "${checkpoint_dir}/last_model.pth" || true)"

  if [[ -z "${best_src}" && -z "${last_src}" ]]; then
    echo "No checkpoint found in ${checkpoint_dir}" >&2
    exit 1
  fi

  instance_slug="$(slugify "${MODEL_NAME}")"
  version_dir="${EXPORT_ROOT}/hf_model_${instance_slug}"

  rm -rf "${version_dir}"
  mkdir -p "${version_dir}/checkpoints"
  [[ -n "${best_src}" ]] && cp -f "${best_src}" "${version_dir}/checkpoints/best_model.pth"
  [[ -n "${last_src}" ]] && cp -f "${last_src}" "${version_dir}/checkpoints/last_model.pth"
  cat > "${version_dir}/checkpoint-manifest.json" <<EOF
{
  "model_name": "${MODEL_NAME}",
  "best_source": "${best_src}",
  "last_source": "${last_src}"
}
EOF
  cat > "${version_dir}/README.md" <<EOF
# ${MODEL_NAME}

Object detection checkpoint exported from \`saved_results/${MODEL_NAME}/checkpoints\`.

Files:
- \`checkpoints/best_model.pth\`
- \`checkpoints/last_model.pth\`
- \`checkpoint-manifest.json\`
EOF

  echo "============ Upload Hugging Face Model ============"
  echo "model_name: ${MODEL_NAME}"
  echo "repo_id: ${HF_REPO_ID}"
  echo "path_in_repo: ${instance_slug}"
  echo "checkpoint_dir: ${checkpoint_dir}"
  echo "version_dir: ${version_dir}"
  echo "==================================================="

  HF_REPO_ID="${HF_REPO_ID}" \
  HF_REPO_TYPE="${HF_REPO_TYPE}" \
  HF_REPO_PRIVATE="${HF_REPO_PRIVATE}" \
  HF_MODEL_PATH="${instance_slug}" \
  HF_UPLOAD_DIR="${version_dir}" \
  HF_COMMIT_MESSAGE="Update checkpoint for ${MODEL_NAME}" \
  python -c '
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ.get("HF_REPO_TYPE", "model")
private = os.environ.get("HF_REPO_PRIVATE", "0") == "1"
path_in_repo = os.environ["HF_MODEL_PATH"]
folder_path = os.environ["HF_UPLOAD_DIR"]
commit_message = os.environ.get("HF_COMMIT_MESSAGE", "Upload checkpoint")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type=repo_type,
    folder_path=folder_path,
    path_in_repo=path_in_repo,
    commit_message=commit_message,
)
print(f"Uploaded to https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")
'
}

upload_dataset() {
  require_kaggle
  if [[ ! -d "${PUBLIC_DIR}" ]]; then
    echo "Public dataset directory not found: ${PUBLIC_DIR}" >&2
    exit 1
  fi

  local dataset_dir="${EXPORT_ROOT}/public_dataset"
  rm -rf "${dataset_dir}"
  mkdir -p "${dataset_dir}"
  cp -R "${PUBLIC_DIR}" "${dataset_dir}/public"
  cat > "${dataset_dir}/dataset-metadata.json" <<EOF
{
  "title": "${KAGGLE_DATASET_TITLE}",
  "id": "${KAGGLE_DATASET_SLUG}",
  "licenses": [{"name": "${KAGGLE_DATASET_LICENSE}"}]
}
EOF

  echo "============ Upload Kaggle Dataset ============"
  echo "dataset: ${KAGGLE_DATASET_SLUG}"
  echo "source: ${PUBLIC_DIR}"
  echo "stage: ${dataset_dir}"
  echo "================================================"

  if ! kaggle datasets version -p "${dataset_dir}" --dir-mode zip -m "Update public dataset"; then
    echo "Dataset version failed; trying create..."
    kaggle datasets create -p "${dataset_dir}" --dir-mode zip
  fi
}

parse_args "$@"

case "${MODE}" in
  model)
    upload_model
    ;;
  dataset)
    upload_dataset
    ;;
  *)
    usage
    exit 1
    ;;
esac
