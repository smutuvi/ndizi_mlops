#!/usr/bin/env bash
# 25-epoch MMS-FA: skip pre-chunk QC, post-chunk audio-only QC, early stopping, Hub push.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BUNDLE_ROOT}"

export HF_HOME="${HF_HOME:-${BUNDLE_ROOT}/.cache/huggingface}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_ndizi}"
export LIBROSA_CACHE_DIR="${LIBROSA_CACHE_DIR:-/tmp/librosa_cache_ndizi}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_cache_ndizi}"
mkdir -p "${HF_HOME}" "${NUMBA_CACHE_DIR}" "${LIBROSA_CACHE_DIR}" "${MPLCONFIGDIR}"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_API_KEY:-}" ]]; then
  echo "WARN: No HF_TOKEN set; push_to_hub may fail." >&2
fi

CONFIG="${BUNDLE_ROOT}/config_files/whisper/merged_full_mms_fa_25epoch_hub_skip_pre_qc.json"
echo "CONFIG=${CONFIG}"

exec python3 "${BUNDLE_ROOT}/scripts/train_whisper.py" --config "${CONFIG}" "$@"
