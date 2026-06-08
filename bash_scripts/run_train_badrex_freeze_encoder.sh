#!/usr/bin/env bash
# Badrex domain adapt: freeze full encoder, train CTC head only (preserves Swahili acoustics).
# 3 epochs, QC on, hub badrex vocab. A/B vs run_train_badrex_lora_punct.sh (trainable_scope=full).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_train_ndizi_w2vbert.sh" \
  "${SCRIPT_DIR}/../config_files/w2vbert/ndizi_w2vbert_badrex_domain_3epoch_freeze_encoder.json" \
  "$@"
