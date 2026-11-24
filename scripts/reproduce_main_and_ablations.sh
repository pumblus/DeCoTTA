#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_ROOT}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT_DIR="${OUT_DIR:-results/reproduce_decotta}"
DEVICE="${DEVICE:-cuda:0}"

PRESETS=(
  pd_dual_tta
  shallow_only
  no_rescue
  no_prev_match
  no_supp_only
  no_shift_only
  no_supp_shift_gate
  deep_inst_only
  weight_only_sf
)

python -m evaluate.run_benchmark \
  --suite "${1:-all}" \
  --presets "${PRESETS[@]}" \
  --out_dir "${OUT_DIR}" \
  --device "${DEVICE}"
