#!/usr/bin/env bash
# Full Ndizi LoRA on badrex: custom punct vocab, oral norm, QC on by default (--no-apply-data-qc to skip).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_train_ndizi_w2vbert.sh" \
  "${SCRIPT_DIR}/../config_files/w2vbert/ndizi_w2vbert_badrex_lora_10epoch_punct.json" \
  "$@"
