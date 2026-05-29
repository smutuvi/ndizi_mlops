#!/usr/bin/env bash
# Set caches, then run training from a config file (JSON or YAML).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BUNDLE_ROOT}"

export HF_HOME="${HF_HOME:-${BUNDLE_ROOT}/.cache/huggingface}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_ndizi}"
export LIBROSA_CACHE_DIR="${LIBROSA_CACHE_DIR:-/tmp/librosa_cache_ndizi}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_cache_ndizi}"
mkdir -p "${HF_HOME}" "${NUMBA_CACHE_DIR}" "${LIBROSA_CACHE_DIR}" "${MPLCONFIGDIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "BUNDLE_ROOT=${BUNDLE_ROOT}"
echo "HF_HOME=${HF_HOME}"

CONFIG="${1:-${BUNDLE_ROOT}/config_files/w2vbert/ndizi_w2vbert_merged.json}"
echo "CONFIG=${CONFIG}"

exec python3 "${BUNDLE_ROOT}/scripts/train_model.py" --config "${CONFIG}"
