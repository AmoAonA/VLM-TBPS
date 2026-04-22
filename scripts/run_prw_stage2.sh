#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PSD2_DATA_ROOT:-}}"
TEXT_ROOT="${TEXT_ROOT:-${PSD2_TEXT_DATA_ROOT:-${PROJECT_ROOT}/data_text}}"
NUM_GPUS="${NUM_GPUS:-1}"
CONFIG_FILE="${CONFIG_FILE:-configs/open_source/viper_semantic_cerberus_stage2_attr_from_stage1_bnneckfalse.yaml}"
STAGE1_CKPT="${STAGE1_CKPT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/viper/PRW/viper_semantic_cerberus_stage2_attr_from_stage1_bnneckfalse}"
RUN_EVAL_AFTER="${RUN_EVAL_AFTER:-0}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}_eval}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
EVAL_FUSION_MODE="${EVAL_FUSION_MODE:-avg}"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT or PSD2_DATA_ROOT must be set." >&2
  exit 1
fi

if [[ -z "${STAGE1_CKPT}" ]]; then
  echo "STAGE1_CKPT must be set." >&2
  exit 1
fi

export PSD2_DATA_ROOT="${DATA_ROOT}"
export PSD2_TEXT_DATA_ROOT="${TEXT_ROOT}"

PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" tools/train_ps_net.py \
    --config-file "${CONFIG_FILE}" \
    --num-gpus="${NUM_GPUS}" \
    --dist-url auto \
    MODEL.WEIGHTS "${STAGE1_CKPT}" \
    PERSON_SEARCH.REID.SCHEMA_PATH "${TEXT_ROOT}/PRW/generated_schema.json" \
    INPUT.ATTRIBUTE_ANNOTATIONS_PATH "${TEXT_ROOT}/PRW/prw_Final_Complete_cleaned_train.json" \
    INPUT.TEST_ATTRIBUTE_ANNOTATIONS_PATH "${TEXT_ROOT}/PRW/prw_Final_Complete_cleaned_test.json" \
    OUTPUT_DIR "${OUTPUT_DIR}"

if [[ "${RUN_EVAL_AFTER}" == "1" ]]; then
  PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" tools/train_ps_net.py \
      --config-file "${CONFIG_FILE}" \
      --eval-only \
      --num-gpus=1 \
      --dist-url auto \
      MODEL.WEIGHTS "${OUTPUT_DIR}/model_final.pth" \
      PERSON_SEARCH.REID.SCHEMA_PATH "${TEXT_ROOT}/PRW/generated_schema.json" \
      INPUT.ATTRIBUTE_ANNOTATIONS_PATH "${TEXT_ROOT}/PRW/prw_Final_Complete_cleaned_train.json" \
      INPUT.TEST_ATTRIBUTE_ANNOTATIONS_PATH "${TEXT_ROOT}/PRW/prw_Final_Complete_cleaned_test.json" \
      TEST.SEARCH_FUSION_MODE "${EVAL_FUSION_MODE}" \
      DATALOADER.NUM_WORKERS "${EVAL_NUM_WORKERS}" \
      OUTPUT_DIR "${EVAL_OUTPUT_DIR}"
fi
