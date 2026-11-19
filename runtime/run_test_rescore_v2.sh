#!/usr/bin/env bash
# Re-score the three test arms (registration v2): the same generated audio, score_tts.py with Whisper long-form for
# audio over 30 s. Outputs <tag>-score-v2; the first scoring stays as it was. Prints UTC start and end.
# usage: run_test_rescore_v2.sh
set -euo pipefail
A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # this runtime directory
OUT="${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_OUT:-$A/out}"                 # run outputs
REC="${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RECORDS:-$OUT/records}"        # per-step logs
mkdir -p "$OUT" "$REC"
for tag in test-base test-dpo-train-k6-final test-rl; do
  if [ ! -e "$OUT/$tag-score-v2/score_meta.json" ]; then
    echo "score-v2 $tag start $(date -u +%FT%TZ)"
    env -u HF_TOKEN HF_HUB_OFFLINE=1 "$A/eval/.venv/bin/python" "$A/eval/score_tts.py" --gen "$OUT/$tag-gen" \
      --items "$A/items-hard-test" --out "$OUT/$tag-score-v2" --no-real > "$REC/$tag-score-v2.log" 2>&1
    echo "score-v2 $tag end $(date -u +%FT%TZ)"
  fi
done
