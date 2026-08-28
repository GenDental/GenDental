#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

visible_gpu_count() {
  awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}"
}

run_gendental() {
  local mode="$1"
  local config="$2"
  shift 2
  (
    cd "${REPO_ROOT}"
    python main.py --mode "$mode" --config "$config" \
      --num_gpus "$(visible_gpu_count)" "$@"
  )
}
