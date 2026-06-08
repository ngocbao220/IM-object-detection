#!/usr/bin/env bash
set -euo pipefail

SOURCE="${SOURCE:-kaggle}"
RUN_NAME="${RUN_NAME:-custom-baseline}"
RUN_NAMES="${RUN_NAMES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./saved_result/${RUN_NAME}/checkpoints}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
KAGGLE_DATASET_SLUG="${KAGGLE_DATASET_SLUG:-ngocbaotrinhtuan/object-detection}"

mkdir -p "${OUTPUT_DIR}"

RUN_NAME_LIST=()
if [[ -n "${RUN_NAMES}" ]]; then
  IFS=',' read -r -a _raw_run_names <<< "${RUN_NAMES}"
  for item in "${_raw_run_names[@]}"; do
    trimmed="$(echo "${item}" | xargs)"
    if [[ -n "${trimmed}" ]]; then
      RUN_NAME_LIST+=("${trimmed}")
    fi
  done
else
  RUN_NAME_LIST=("${RUN_NAME}")
fi

latest_file() {
  local pattern="$1"
  local directory="$2"
  local fallback="$3"
  local found
  found="$(find "${directory}" -maxdepth 1 -type f -name "${pattern}" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
  if [[ -n "${found}" ]]; then
    printf '%s\n' "${found}"
  elif [[ -f "${fallback}" ]]; then
    printf '%s\n' "${fallback}"
  fi
}

copy_local_checkpoints() {
  local found_any=0
  local checkpoint_dir best_src last_src target_dir

  for run_name in "${RUN_NAME_LIST[@]}"; do
    checkpoint_dir="${PROJECT_ROOT}/saved_results/${run_name}/checkpoints"
    best_src="$(latest_file 'best_model-*.pth' "${checkpoint_dir}" "${checkpoint_dir}/best_model.pth" || true)"
    last_src="$(latest_file 'last_model-*.pth' "${checkpoint_dir}" "${checkpoint_dir}/last_model.pth" || true)"

    if [[ -z "${best_src}" && -z "${last_src}" ]]; then
      echo "No checkpoint found under ${checkpoint_dir}; skipping ${run_name}" >&2
      continue
    fi

    if [[ ${#RUN_NAME_LIST[@]} -gt 1 ]]; then
      target_dir="${OUTPUT_DIR}/${run_name}"
    else
      target_dir="${OUTPUT_DIR}"
    fi
    mkdir -p "${target_dir}"

    if [[ -n "${best_src}" ]]; then
      cp -f "${best_src}" "${target_dir}/best_model.pth"
      echo "Copied best checkpoint: ${best_src} -> ${target_dir}/best_model.pth"
    fi
    if [[ -n "${last_src}" ]]; then
      cp -f "${last_src}" "${target_dir}/last_model.pth"
      echo "Copied last checkpoint: ${last_src} -> ${target_dir}/last_model.pth"
    fi
    found_any=1
  done

  if [[ "${found_any}" -ne 1 ]]; then
    echo "No checkpoint found for all requested run names: ${RUN_NAME_LIST[*]}" >&2
    exit 1
  fi
}

copy_kaggle_checkpoints() {
  if [[ -z "${KAGGLE_DATASET_SLUG}" ]]; then
    echo "Set KAGGLE_DATASET_SLUG=owner/dataset-name" >&2
    exit 1
  fi
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "kaggle CLI is not installed. Run: pip install kaggle" >&2
    exit 1
  fi

  local tmp_dir="${OUTPUT_DIR}/.kaggle_download"
  rm -rf "${tmp_dir}"
  mkdir -p "${tmp_dir}"
  kaggle datasets download -d "${KAGGLE_DATASET_SLUG}" -p "${tmp_dir}"
  local zip_file
  zip_file="$(find "${tmp_dir}" -maxdepth 1 -name '*.zip' | head -n1)"
  if [[ -z "${zip_file}" ]]; then
    echo "Kaggle download did not produce a zip file." >&2
    exit 1
  fi
  unzip -o "${zip_file}" -d "${OUTPUT_DIR}" >/dev/null
  rm -rf "${tmp_dir}"
}

case "${SOURCE}" in
  local)
    copy_local_checkpoints
    ;;
  kaggle)
    copy_kaggle_checkpoints
    ;;
  *)
    echo "SOURCE must be one of: local, kaggle" >&2
    exit 1
    ;;
esac

echo "Done. Files are in ${OUTPUT_DIR}"
