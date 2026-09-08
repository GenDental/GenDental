#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,6}"
TASK_MODE="${TASK_MODE:-target}"
case "$TASK_MODE" in
  target|motion) ;;
  *) echo "TASK_MODE must be target or motion" >&2; exit 2 ;;
esac
if [[ "$TASK_MODE" == "target" ]]; then
  DEFAULT_OUTPUT_DIR="/data3/leics/dataset/checkpoints/Gendental/stage_two"
  DEFAULT_CKPT_PATH=""
  DEFAULT_INDEX_PATH="files/zj"
else
  DEFAULT_OUTPUT_DIR="/data3/leics/dataset/checkpoints/ToothWise/motion_transfer_smooth3"
  DEFAULT_CKPT_PATH="/data3/leics/dataset/checkpoints/ToothWise/motion_transfer_smooth2/ckpt/last.ckpt"
  DEFAULT_INDEX_PATH="files/motion"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
CKPT_PATH="${CKPT_PATH:-$DEFAULT_CKPT_PATH}"
DATA_PATH="${DATA_PATH:-/data3/leics/dataset/teeth/sample512_merged}"
INDEX_PATH="${INDEX_PATH:-$DEFAULT_INDEX_PATH}"
EXTRA_ARGS=()
if [[ -n "$CKPT_PATH" ]]; then
  EXTRA_ARGS+=(--ckpt_path "$CKPT_PATH")
fi
run_gendental train configs/stage_two.yaml \
  --output_dir "$OUTPUT_DIR" \
  --epochs 500 --base_lr 1e-4 --monitor val_total_loss --fast \
  --set "model.params.task_mode=$TASK_MODE" \
  --set "dataset.params.data_path=$DATA_PATH" \
  --set "dataset.params.index_path=$INDEX_PATH" \
  "${EXTRA_ARGS[@]}"
