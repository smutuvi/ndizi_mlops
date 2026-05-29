#!/usr/bin/env bash
# Transcribe one local WAV (CTC or Whisper). Forwards all args to scripts/evaluate_asr_single.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BUNDLE_ROOT}"

export HF_HOME="${HF_HOME:-${BUNDLE_ROOT}/.cache/huggingface}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_ndizi}"
export LIBROSA_CACHE_DIR="${LIBROSA_CACHE_DIR:-/tmp/librosa_cache_ndizi}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_cache_ndizi}"
mkdir -p "${HF_HOME}" "${NUMBA_CACHE_DIR}" "${LIBROSA_CACHE_DIR}" "${MPLCONFIGDIR}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 --model_path DIR --audio_path FILE.wav [optional flags]" >&2
  exit 1
fi

exec python3 "${BUNDLE_ROOT}/scripts/evaluate_asr_single.py" "$@"
