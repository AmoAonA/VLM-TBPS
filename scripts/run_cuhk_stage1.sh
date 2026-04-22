#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PSD2_DATA_ROOT:-}}"
NUM_GPUS="${NUM_GPUS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/viper/CUHK-SYSU/clipviper_cuhk_stage1_baseline}"
CONFIG_FILE="${CONFIG_FILE:-configs/open_source/clipviper_cuhk_stage1_baseline.yaml}"
RUN_EVAL_AFTER="${RUN_EVAL_AFTER:-1}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}_eval}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT or PSD2_DATA_ROOT must be set." >&2
  exit 1
fi

export PSD2_DATA_ROOT="${DATA_ROOT}"

PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" tools/train_ps_net.py \
    --config-file "${CONFIG_FILE}" \
    --num-gpus="${NUM_GPUS}" \
    --dist-url auto \
    OUTPUT_DIR "${OUTPUT_DIR}"

if [[ "${RUN_EVAL_AFTER}" == "1" ]]; then
  PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" tools/train_ps_net.py \
      --config-file "${CONFIG_FILE}" \
      --eval-only \
      --num-gpus=1 \
      --dist-url auto \
      MODEL.WEIGHTS "${OUTPUT_DIR}/model_final.pth" \
      DATALOADER.NUM_WORKERS "${EVAL_NUM_WORKERS}" \
      OUTPUT_DIR "${EVAL_OUTPUT_DIR}"
fi
