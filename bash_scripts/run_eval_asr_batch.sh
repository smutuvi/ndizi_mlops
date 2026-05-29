#!/usr/bin/env bash
# Run batched Hub eval for CTC or Whisper checkpoints (same Python driver).
#
# Usage (from bundle root or any cwd):
#   bash_scripts/run_eval_asr_batch.sh \
#     --model_path inprogress/.../checkpoint-dir \
#     --output_dir eval/my_run \
#     --test_datasets smutuvi/ndizi-1:test smutuvi/ndizi-1-2025:test
#
# Backend: default --backend auto (uses training_config_resolved.json next to the
# checkpoint: stack whisper → Whisper generate, else CTC greedy). Override with
# --backend ctc or --backend whisper. Optional: --whisper_language sw --whisper_task transcribe
#
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
  echo "Pass arguments to scripts/evaluate_asr_batch.py (at least --model_path, --output_dir, --test_datasets ...)." >&2
  exit 1
fi

exec python3 "${BUNDLE_ROOT}/scripts/evaluate_asr_batch.py" "$@"
