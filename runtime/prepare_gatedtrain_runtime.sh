#!/usr/bin/env bash
# Make this runtime directory launchable under `gatedtrain ext-launch` (read-only mount, no network), after the setup of
# README.md (uv sync of .venv and eval/.venv, fetch_cosyvoice.sh, fetch_models.py). Idempotent; host side, no GPU.
# - eval/ecapa-cache/: score_tts.py lets SpeechBrain link the speaker model's files there on first use, which a
#   read-only mount forbids; the links are made here, relative, so they resolve at any mount point;
# - src/CosyVoice/third_party/Matcha-TTS/.env.example: removed (fetch_cosyvoice.sh does it too), see there;
# - then checks what an arm and a training launch read: both venvs, the base and evaluator models, the prompt audio.
# usage: prepare_gatedtrain_runtime.sh
set -euo pipefail
A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E="$A/eval/ecapa-cache"
mkdir -p "$E"
while read -r dst src; do
  if [ ! -e "$E/$dst" ]; then
    ln -sfn "../../models/spkrec-ecapa-voxceleb/$src" "$E/$dst"
  fi
done <<'LINKS'
classifier.ckpt classifier.ckpt
embedding_model.ckpt embedding_model.ckpt
hyperparams.yaml hyperparams.yaml
label_encoder.ckpt label_encoder.txt
mean_var_norm_emb.ckpt mean_var_norm_emb.ckpt
LINKS
rm -f "$A/src/CosyVoice/third_party/Matcha-TTS/.env.example"
missing=0
for p in .venv/bin/python eval/.venv/bin/python src/CosyVoice/cosyvoice/cli/cosyvoice.py \
         models/fun-cosyvoice3-0.5b-2512/llm.pt models/qwen3-asr-1.7b models/whisper-large-v3-turbo/normalizer.json \
         models/spkrec-ecapa-voxceleb/embedding_model.ckpt data-source/unpacked-test_clean data-source/unpacked-dev_clean \
         eval/ecapa-cache/embedding_model.ckpt; do
  if [ ! -e "$A/$p" ]; then echo "missing: $A/$p" >&2; missing=1; fi
done
if [ -e "$A/models/fun-cosyvoice3-0.5b-2512/.cache" ]; then
  echo "models/fun-cosyvoice3-0.5b-2512/.cache exists (download metadata); ext-collect refuses a base with it" >&2
  missing=1
fi
exit "$missing"
