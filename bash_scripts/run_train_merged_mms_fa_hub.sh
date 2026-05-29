#!/usr/bin/env bash
# Full fine-tune 25 epochs: QC + MMS-FA chunking, push to smutuvi/ndizi_whisper_large_v3_turbo_merged_mms_fa
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
  echo "WARN: No HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HF_API_KEY set; push_to_hub may fail." >&2
fi

CONFIG="${BUNDLE_ROOT}/config_files/whisper/merged_full_mms_fa_25epoch_hub.json"
echo "BUNDLE_ROOT=${BUNDLE_ROOT}"
echo "CONFIG=${CONFIG}"

exec python3 "${BUNDLE_ROOT}/scripts/train_whisper.py" --config "${CONFIG}" "$@"
