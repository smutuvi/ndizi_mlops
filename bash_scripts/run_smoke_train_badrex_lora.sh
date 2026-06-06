#!/usr/bin/env bash
# Smoke badrex domain adapt: keep hub Swahili vocab, frozen encoder, train CTC head, 512 rows, 1 epoch.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_train_ndizi_w2vbert.sh" \
  "${SCRIPT_DIR}/../config_files/w2vbert/ndizi_w2vbert_badrex_lora_smoke_1epoch.json" \
  "$@"
