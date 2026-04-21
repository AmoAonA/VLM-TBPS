#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PSD2_DATA_ROOT:-}}"
TEXT_ROOT="${TEXT_ROOT:-${PSD2_TEXT_DATA_ROOT:-${PROJECT_ROOT}/data_text}}"
CONFIG_FILE="${CONFIG_FILE:-configs/open_source/clipviper_prw_stage1_baseline.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smoke/prw_one_iter}"
IMS_PER_BATCH="${IMS_PER_BATCH:-1}"
ROI_BATCH_SIZE="${ROI_BATCH_SIZE:-32}"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT or PSD2_DATA_ROOT must be set." >&2
  exit 1
fi

export PSD2_DATA_ROOT="${DATA_ROOT}"
export PSD2_TEXT_DATA_ROOT="${TEXT_ROOT}"

PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" tools/train_ps_net.py \
    --config-file "${CONFIG_FILE}" \
    --num-gpus=1 \
    --dist-url auto \
    DATALOADER.NUM_WORKERS 0 \
    SOLVER.MAX_ITER 1 \
    SOLVER.STEPS "()" \
    SOLVER.WARMUP_ITERS 0 \
    SOLVER.IMS_PER_BATCH "${IMS_PER_BATCH}" \
    MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE "${ROI_BATCH_SIZE}" \
    TEST.EVAL_PERIOD 0 \
    OUTPUT_DIR "${OUTPUT_DIR}"

echo "train_one_iter_ok"
