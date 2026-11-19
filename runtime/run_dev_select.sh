#!/usr/bin/env bash
# Checkpoint selection on the dev set (train templates, dev_clean voices): base, every saved DPO checkpoint and the
# final weights generate items-hard-dev with the same seed, then each is scored (no real recordings). Selection uses
# wer_qwen_canon. Each step prints UTC start and end times for the GPU log.
# usage: run_dev_select.sh <dpo-out-dir-name>   (e.g. dpo-train-k6)
# The published run's layout (llm_dpo.step<N>.pt, llm_dpo.pt) is that of train_dpo.py in this repository's first
# commit; the current train_dpo.py writes checkpoint-<N>/llm.pt for gatedtrain (launch/tts.sh is the rerun).
set -euo pipefail
A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # this runtime directory
OUT="${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_OUT:-$A/out}"                 # run outputs
REC="${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RECORDS:-$OUT/records}"        # per-step logs
mkdir -p "$OUT" "$REC"
RUN="$1"
declare -A LLM=([dev-base]=base ["dev-$RUN-final"]="$OUT/$RUN/llm_dpo.pt")
for f in "$OUT/$RUN"/llm_dpo.step*.pt; do
  name=$(basename "$f" .pt)
  LLM["dev-$RUN-${name#llm_dpo.}"]="$f"
done
for tag in "${!LLM[@]}"; do
  if [ ! -e "$OUT/$tag-gen/gen_meta.json" ]; then
    echo "gen $tag start $(date -u +%FT%TZ)"
    env -u HF_TOKEN HF_HUB_OFFLINE=1 "$A/.venv/bin/python" "$A/gen_tts.py" --model-dir "$A/models/fun-cosyvoice3-0.5b-2512" \
      --items "$A/items-hard-dev" --out "$OUT/$tag-gen" --llm "${LLM[$tag]}" --seed 11 > "$REC/$tag-gen.log" 2>&1
    echo "gen $tag end $(date -u +%FT%TZ)"
  fi
  if [ ! -e "$OUT/$tag-score/score_meta.json" ]; then
    echo "score $tag start $(date -u +%FT%TZ)"
    env -u HF_TOKEN HF_HUB_OFFLINE=1 "$A/eval/.venv/bin/python" "$A/eval/score_tts.py" --gen "$OUT/$tag-gen" \
      --items "$A/items-hard-dev" --out "$OUT/$tag-score" --no-real > "$REC/$tag-score.log" 2>&1
    echo "score $tag end $(date -u +%FT%TZ)"
  fi
done
