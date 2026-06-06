#!/usr/bin/env bash
# Badrex domain adapt quick run: 3 epochs, hub vocab, frozen encoder, train CTC head on Ndizi data.
# QC on by default; pass --no-apply-data-qc to skip.
# For 10 epochs use config_files/w2vbert/ndizi_w2vbert_badrex_domain_10epoch.json instead.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_train_ndizi_w2vbert.sh" \
  "${SCRIPT_DIR}/../config_files/w2vbert/ndizi_w2vbert_badrex_domain_3epoch.json" \
  "$@"
