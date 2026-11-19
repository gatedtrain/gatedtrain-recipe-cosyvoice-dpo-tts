#!/usr/bin/env bash
# Clone the CosyVoice source at the pinned commit into src/CosyVoice (with its Matcha-TTS submodule at the commit the
# CosyVoice tree records). gen_tts.py, sample_candidates.py and train_dpo.py import it from there; it is not part of
# this repository (Apache-2.0, see the README "Data and licenses"; Matcha-TTS is MIT). Network.
# usage: fetch_cosyvoice.sh
set -euo pipefail
A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COSYVOICE_URL=https://github.com/QwenAudio/CosyVoice.git
COSYVOICE_COMMIT=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
MATCHA_COMMIT=dd9105b34bf2be2230f4aa1e4769fb586a3c824e
D="$A/src/CosyVoice"
if [ ! -d "$D/.git" ]; then
  mkdir -p "$A/src"
  git clone --no-checkout "$COSYVOICE_URL" "$D"
fi
git -C "$D" checkout --detach "$COSYVOICE_COMMIT"
git -C "$D" submodule update --init third_party/Matcha-TTS
test "$(git -C "$D" rev-parse HEAD)" = "$COSYVOICE_COMMIT"
test "$(git -C "$D/third_party/Matcha-TTS" rev-parse HEAD)" = "$MATCHA_COMMIT"
# Matcha-TTS ships an unused template `.env.example`; gatedtrain ext-launch refuses a read-only mount holding a
# credential-like file name (`.env.*`, EXT-80), so it is removed from the working tree (nothing imports it).
rm -f "$D/third_party/Matcha-TTS/.env.example"
echo "src/CosyVoice at $COSYVOICE_COMMIT, Matcha-TTS at $MATCHA_COMMIT"
