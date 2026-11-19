#!/usr/bin/env bash
# Hard-text pilot: base and upstream-rl LLM on items-hard-pilot, then scoring (no real recordings: the hard texts
# have none). Each step prints UTC start and end times for the GPU log.
# usage: run_pilot_hard.sh
set -euo pipefail
A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # this runtime directory
OUT="${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_OUT:-$A/out}"                 # run outputs
REC="${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RECORDS:-$OUT/records}"        # per-step logs
mkdir -p "$OUT" "$REC"
for llm in base rl; do
  echo "gen-$llm start $(date -u +%FT%TZ)"
  env -u HF_TOKEN HF_HUB_OFFLINE=1 "$A/.venv/bin/python" "$A/gen_tts.py" --model-dir "$A/models/fun-cosyvoice3-0.5b-2512" \
    --items "$A/items-hard-pilot" --out "$OUT/pilot-hard-raw-gen-$llm" --llm "$llm" --seed 11 > "$REC/pilot-hard-raw-gen-$llm.log" 2>&1
  echo "gen-$llm end $(date -u +%FT%TZ)"
done
for llm in base rl; do
  echo "score-$llm start $(date -u +%FT%TZ)"
  env -u HF_TOKEN HF_HUB_OFFLINE=1 "$A/eval/.venv/bin/python" "$A/eval/score_tts.py" --gen "$OUT/pilot-hard-raw-gen-$llm" \
    --items "$A/items-hard-pilot" --out "$OUT/pilot-hard-raw-score-$llm" --no-real > "$REC/pilot-hard-raw-score-$llm.log" 2>&1
  echo "score-$llm end $(date -u +%FT%TZ)"
done
