#!/usr/bin/env bash
# Smoke Whisper train: full merged train set, 1 epoch (formatting + QC + MMS-FA).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_train_ndizi_whisper.sh" \
  "${SCRIPT_DIR}/../config_files/whisper/ndizi_whisper_smoke_format_pipeline.json"
