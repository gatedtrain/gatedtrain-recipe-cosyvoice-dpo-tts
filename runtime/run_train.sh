#!/usr/bin/env bash
# Training pipeline of the recipe: K=6 candidates per training item with the base LLM, Whisper-large-v3-turbo WER
# rewards, chosen/rejected pairs, DPO on the LLM speech tokens. The commands and arguments are those of the published
# run (dpo-train-k6). Each step prints UTC start and end times for the GPU log. Steps whose output exists are skipped.
# The published run passed --ref-cache with the reference log-probs of its 20-step smoke on the same pairs
# (byte-identical inputs); without it train_dpo.py computes them itself before the first update.
# usage: run_train.sh
# The published run's layout (llm_dpo.step<N>.pt, llm_dpo.pt) is that of train_dpo.py in this repository's first
# commit; the current train_dpo.py writes checkpoint-<N>/llm.pt for gatedtrain (launch/tts.sh is the rerun).
set -euo pipefail
A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # this runtime directory
OUT="${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_OUT:-$A/out}"                 # run outputs
REC="${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RECORDS:-$OUT/records}"        # per-step logs
mkdir -p "$OUT" "$REC"
M="$A/models/fun-cosyvoice3-0.5b-2512"
step() { echo "$1 start $(date -u +%FT%TZ)"; shift; "$@"; echo "end $(date -u +%FT%TZ)"; }
if [ ! -e "$OUT/cand-train-k6/candidates_meta.json" ]; then
  step cand env -u HF_TOKEN HF_HUB_OFFLINE=1 "$A/.venv/bin/python" "$A/sample_candidates.py" --model-dir "$M" \
    --items "$A/items-hard-train" --out "$OUT/cand-train-k6" --k 6 --seed 11 --llm base > "$REC/cand-train-k6.log" 2>&1
fi
if [ ! -e "$OUT/reward-train-k6/reward_meta.json" ]; then
  step reward env -u HF_TOKEN HF_HUB_OFFLINE=1 "$A/eval/.venv/bin/python" "$A/eval/score_candidates.py" \
    --candidates "$OUT/cand-train-k6" --items "$A/items-hard-train" --out "$OUT/reward-train-k6" --batch-size 16 \
    > "$REC/reward-train-k6.log" 2>&1
fi
if [ ! -e "$OUT/pairs-train-k6/pairs_meta.json" ]; then
  "$A/.venv/bin/python" "$A/make_pairs.py" --candidates "$OUT/cand-train-k6" --rewards "$OUT/reward-train-k6" \
    --items "$A/items-hard-train" --out "$OUT/pairs-train-k6" --margin 0.05 --min-ratio 0.5 --max-ratio 2.5
fi
if [ ! -e "$OUT/dpo-train-k6/train_meta.json" ]; then
  step dpo env -u HF_TOKEN HF_HUB_OFFLINE=1 "$A/.venv/bin/python" "$A/train_dpo.py" --model-dir "$M" \
    --candidates "$OUT/cand-train-k6" --pairs "$OUT/pairs-train-k6" --out "$OUT/dpo-train-k6" --beta 0.1 --lr 1e-6 \
    --epochs 2 --micro-batch 2 --grad-accum 8 --warmup-steps 10 --clip 1.0 --seed 11 --amp bf16 --save-every 26 \
    > "$REC/dpo-train-k6.log" 2>&1
fi
