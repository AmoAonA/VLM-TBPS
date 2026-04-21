#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PSD2_DATA_ROOT:-}}"
TEXT_ROOT="${TEXT_ROOT:-${PSD2_TEXT_DATA_ROOT:-${PROJECT_ROOT}/data_text}}"
NUM_GPUS="${NUM_GPUS:-1}"
STAGE1_CKPT="${STAGE1_CKPT:-}"
CONFIG_FILE="${CONFIG_FILE:-configs/open_source/viper_semantic_cerberus_cuhk_stage2_attr_from_stage1_bnneckfalse.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/viper/CUHK-SYSU/viper_semantic_cerberus_cuhk_stage2_attr_from_stage1_bnneckfalse}"
RUN_EVAL_AFTER="${RUN_EVAL_AFTER:-1}"
EVAL_MODE="${EVAL_MODE:-posrerank}"
EVAL_CONFIG_FILE="${EVAL_CONFIG_FILE:-configs/open_source/viper_semantic_cerberus_cuhk_stage2_posrerank_eval.yaml}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}_eval_${EVAL_MODE}}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"

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
    PERSON_SEARCH.REID.SCHEMA_PATH "${TEXT_ROOT}/cuhk_sysu/generated_schema.json" \
    INPUT.ATTRIBUTE_ANNOTATIONS_PATH "${TEXT_ROOT}/cuhk_sysu/CUHK_SYSU_Final_Complete_train_cleaned.json" \
    INPUT.TEST_ATTRIBUTE_ANNOTATIONS_PATH "${TEXT_ROOT}/cuhk_sysu/CUHK_SYSU_Final_Complete_test_cleaned.json" \
    OUTPUT_DIR "${OUTPUT_DIR}"

if [[ "${RUN_EVAL_AFTER}" == "1" ]]; then
  if [[ "${EVAL_MODE}" == "global" ]]; then
    EVAL_CONFIG_FILE="${CONFIG_FILE}"
    EXTRA_EVAL_OPTS=(TEST.SEARCH_FUSION_MODE global)
  elif [[ "${EVAL_MODE}" == "posrerank" ]]; then
    EXTRA_EVAL_OPTS=()
  else
    echo "EVAL_MODE must be 'posrerank' or 'global'." >&2
    exit 1
  fi

  PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" tools/train_ps_net.py \
      --config-file "${EVAL_CONFIG_FILE}" \
      --eval-only \
      --num-gpus=1 \
      --dist-url auto \
      MODEL.WEIGHTS "${OUTPUT_DIR}/model_final.pth" \
      PERSON_SEARCH.REID.SCHEMA_PATH "${TEXT_ROOT}/cuhk_sysu/generated_schema.json" \
      INPUT.ATTRIBUTE_ANNOTATIONS_PATH "${TEXT_ROOT}/cuhk_sysu/CUHK_SYSU_Final_Complete_train_cleaned.json" \
      INPUT.TEST_ATTRIBUTE_ANNOTATIONS_PATH "${TEXT_ROOT}/cuhk_sysu/CUHK_SYSU_Final_Complete_test_cleaned.json" \
      DATALOADER.NUM_WORKERS "${EVAL_NUM_WORKERS}" \
      "${EXTRA_EVAL_OPTS[@]}" \
      OUTPUT_DIR "${EVAL_OUTPUT_DIR}"
fi
