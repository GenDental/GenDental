#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-/data3/leics/dataset/checkpoints/Gendental/TAlig0x}"
CKPT_PATH="${CKPT_PATH:-}"
INIT_CKPT="${INIT_CKPT:-}"
DATA_PATH="${DATA_PATH:-/data3/leics/dataset/teeth/merged_alignment_5x}"
INDEX_PATH="${INDEX_PATH:-files/alignment5x}"
EPOCHS="${EPOCHS:-500}"
BASE_LR="${BASE_LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"

if [[ -n "$CKPT_PATH" && -n "$INIT_CKPT" ]]; then
  echo "Set only one of CKPT_PATH (resume) or INIT_CKPT (weights only)." >&2
  exit 2
fi
EXTRA_ARGS=()
[[ -n "$CKPT_PATH" ]] && EXTRA_ARGS+=(--ckpt_path "$CKPT_PATH")
[[ -n "$INIT_CKPT" ]] && EXTRA_ARGS+=(--init_ckpt "$INIT_CKPT")

run_gendental train configs/TAlig.yaml \
  --output_dir "$OUTPUT_DIR" --epochs "$EPOCHS" \
  "${EXTRA_ARGS[@]}" \
  --base_lr "$BASE_LR" --fast \
  --set "dataset.params.data_path=$DATA_PATH" \
  --set "dataset.params.index_path=$INDEX_PATH" \
  --set "dataset.params.batch_size=$BATCH_SIZE"
