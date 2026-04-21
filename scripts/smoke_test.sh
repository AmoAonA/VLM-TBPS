#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PSD2_DATA_ROOT:-}}"

if [[ -n "${DATA_ROOT}" ]]; then
  export PSD2_DATA_ROOT="${DATA_ROOT}"
  bash scripts/check_dataset_layout.sh
else
  echo "DATA_ROOT not set; skip dataset layout check."
fi

"${PYTHON_BIN}" tools/train_ps_net.py --help >/dev/null
"${PYTHON_BIN}" tools/eval_search_from_cache.py --help >/dev/null

echo "smoke_test_ok"
