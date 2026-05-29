#!/usr/bin/env bash
# Smoke: full fine-tune + QC + MMS-FA (no LoRA, no train_weights). Runs in ndizi_mlops.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BUNDLE_ROOT}"

export HF_HOME="${HF_HOME:-${BUNDLE_ROOT}/.cache/huggingface}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_ndizi}"
export LIBROSA_CACHE_DIR="${LIBROSA_CACHE_DIR:-/tmp/librosa_cache_ndizi}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_cache_ndizi}"
mkdir -p "${HF_HOME}" "${NUMBA_CACHE_DIR}" "${LIBROSA_CACHE_DIR}" "${MPLCONFIGDIR}"

CONFIG="${BUNDLE_ROOT}/config_files/whisper/merged_full_mms_fa_smoke.json"
echo "BUNDLE_ROOT=${BUNDLE_ROOT}"
echo "CONFIG=${CONFIG}"

exec python3 "${BUNDLE_ROOT}/scripts/train_whisper.py" --config "${CONFIG}" "$@"
