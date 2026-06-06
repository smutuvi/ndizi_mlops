#!/usr/bin/env bash
# Full badrex domain adapt (Ethio-ASR style): hub vocab, frozen encoder, train head on Ndizi field data.
# QC on by default; pass --no-apply-data-qc to skip.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_train_ndizi_w2vbert.sh" \
  "${SCRIPT_DIR}/../config_files/w2vbert/ndizi_w2vbert_badrex_domain_10epoch.json" \
  "$@"
