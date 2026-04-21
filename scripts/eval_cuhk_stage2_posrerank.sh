#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PSD2_DATA_ROOT:-}}"
TEXT_ROOT="${TEXT_ROOT:-${PSD2_TEXT_DATA_ROOT:-${PROJECT_ROOT}/data_text}}"
CKPT="${CKPT:-${MODEL_WEIGHTS:-}}"
CONFIG_FILE="${CONFIG_FILE:-configs/open_source/viper_semantic_cerberus_cuhk_stage2_posrerank_eval.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/viper/CUHK-SYSU/cuhk_stage2_attr_posrerank_eval}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT or PSD2_DATA_ROOT must be set." >&2
  exit 1
fi

if [[ -z "${CKPT}" ]]; then
  echo "CKPT or MODEL_WEIGHTS must be set." >&2
  exit 1
fi

export PSD2_DATA_ROOT="${DATA_ROOT}"
export PSD2_TEXT_DATA_ROOT="${TEXT_ROOT}"

PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" tools/train_ps_net.py \
    --config-file "${CONFIG_FILE}" \
    --eval-only \
    --num-gpus=1 \
    --dist-url auto \
    MODEL.WEIGHTS "${CKPT}" \
    PERSON_SEARCH.REID.SCHEMA_PATH "${TEXT_ROOT}/cuhk_sysu/generated_schema.json" \
    INPUT.ATTRIBUTE_ANNOTATIONS_PATH "${TEXT_ROOT}/cuhk_sysu/CUHK_SYSU_Final_Complete_train_cleaned.json" \
    INPUT.TEST_ATTRIBUTE_ANNOTATIONS_PATH "${TEXT_ROOT}/cuhk_sysu/CUHK_SYSU_Final_Complete_test_cleaned.json" \
    DATALOADER.NUM_WORKERS "${EVAL_NUM_WORKERS}" \
    OUTPUT_DIR "${OUTPUT_DIR}"
