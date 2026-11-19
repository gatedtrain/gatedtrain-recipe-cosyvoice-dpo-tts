#!/usr/bin/env bash
# Apply the registered gate (results/2026-11-17/gate/registration-v2.yaml of this repository) to the scored test arms
# with analyze_gate.py; the vendor's rl weights are reported descriptively. Writes $OUT/gate-test-v2/gate_result.json.
# usage: run_gate.sh [registration.yaml]
set -euo pipefail
A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # this runtime directory
OUT="${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_OUT:-$A/out}"                 # run outputs
REC="${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RECORDS:-$OUT/records}"        # per-step logs
mkdir -p "$OUT" "$REC"
REG="${1:-$A/../results/2026-11-17/gate/registration-v2.yaml}"
"$A/eval/.venv/bin/python" "$A/analyze_gate.py" --registration "$REG" --baseline "$OUT/test-base-score-v2" \
  --candidate "$OUT/test-dpo-train-k6-final-score-v2" --reference "$OUT/test-rl-score-v2" --out "$OUT/gate-test-v2"
