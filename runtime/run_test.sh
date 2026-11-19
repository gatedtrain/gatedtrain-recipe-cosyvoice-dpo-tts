#!/usr/bin/env bash
# Test-set runs of the gate: base, the selected DPO weights and the vendor's rl weights generate items-hard-test with
# seed 11, then each is scored (no real recordings). Each step prints UTC start and end times for the GPU log.
# In the published run the first scoring (<tag>-score) used the earlier score_tts.py of registration v1;
# run_test_rescore_v2.sh re-scored the same audio with the current score_tts.py (registration v2), and the gate read
# <tag>-score-v2. With the current score_tts.py both steps give the v2 scoring.
# usage: run_test.sh
# The published run's layout (llm_dpo.step<N>.pt, llm_dpo.pt) is that of train_dpo.py in this repository's first
# commit; the current train_dpo.py writes checkpoint-<N>/llm.pt for gatedtrain (launch/tts.sh is the rerun).
set -euo pipefail
A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # this runtime directory
OUT="${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_OUT:-$A/out}"                 # run outputs
REC="${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RECORDS:-$OUT/records}"        # per-step logs
mkdir -p "$OUT" "$REC"
declare -A LLM=([test-base]=base [test-dpo-train-k6-final]="$OUT/dpo-train-k6/llm_dpo.pt" [test-rl]=rl)
for tag in test-base test-dpo-train-k6-final test-rl; do
  if [ ! -e "$OUT/$tag-gen/gen_meta.json" ]; then
    echo "gen $tag start $(date -u +%FT%TZ)"
    env -u HF_TOKEN HF_HUB_OFFLINE=1 "$A/.venv/bin/python" "$A/gen_tts.py" --model-dir "$A/models/fun-cosyvoice3-0.5b-2512" \
      --items "$A/items-hard-test" --out "$OUT/$tag-gen" --llm "${LLM[$tag]}" --seed 11 > "$REC/$tag-gen.log" 2>&1
    echo "gen $tag end $(date -u +%FT%TZ)"
  fi
  if [ ! -e "$OUT/$tag-score/score_meta.json" ]; then
    echo "score $tag start $(date -u +%FT%TZ)"
    env -u HF_TOKEN HF_HUB_OFFLINE=1 "$A/eval/.venv/bin/python" "$A/eval/score_tts.py" --gen "$OUT/$tag-gen" \
      --items "$A/items-hard-test" --out "$OUT/$tag-score" --no-real > "$REC/$tag-score.log" 2>&1
    echo "score $tag end $(date -u +%FT%TZ)"
  fi
done
