#!/usr/bin/env bash
set -euo pipefail

OFFICIAL_VIPER_ROOT="${OFFICIAL_VIPER_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PSD2_DATA_ROOT:-}}"
NUM_GPUS="${NUM_GPUS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/viper/clipviper_cuhk_official_clean}"
RUN_EVAL_AFTER="${RUN_EVAL_AFTER:-1}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}_eval}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"

if [[ -z "${OFFICIAL_VIPER_ROOT}" ]]; then
  echo "OFFICIAL_VIPER_ROOT must be set." >&2
  exit 1
fi

if [[ -z "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT or PSD2_DATA_ROOT must be set." >&2
  exit 1
fi

cd "${OFFICIAL_VIPER_ROOT}"
ln -sfn "${DATA_ROOT}" Data

PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" tools/train_ps_net.py \
    --config-file configs/person_search/viper/clipviper_cuhk.yaml \
    --num-gpus="${NUM_GPUS}" \
    --dist-url auto \
    OUTPUT_DIR "${OUTPUT_DIR}"

if [[ "${RUN_EVAL_AFTER}" == "1" ]]; then
  PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" tools/train_ps_net.py \
      --config-file configs/person_search/viper/clipviper_cuhk.yaml \
      --eval-only \
      --num-gpus=1 \
      --dist-url auto \
      MODEL.WEIGHTS "${OUTPUT_DIR}/model_final.pth" \
      DATALOADER.NUM_WORKERS "${EVAL_NUM_WORKERS}" \
      OUTPUT_DIR "${EVAL_OUTPUT_DIR}"
fi
