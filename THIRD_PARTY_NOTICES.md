# Third-party notices

This repository redistributes the following third-party material. Everything else in it is covered by `LICENSE`
(Apache License 2.0).

## LibriTTS-R (transcripts)

- **What:** utterance transcripts with their LibriTTS-R utterance ids and speaker ids, in the item sets
  `runtime/items-hard-train/items.jsonl` (2,000 rows), `runtime/items-hard-test/items.jsonl` (450),
  `runtime/items-hard-dev/items.jsonl` (150), `runtime/items-hard-pilot/items.jsonl` (64) and
  `runtime/items-smoke/items.jsonl` (14): every row's voice-prompt transcript (`prompt_text`, `prompt_id`, `speaker`,
  and the relative path of the prompt recording `prompt_wav`), and the target text of the `natural_long` rows and of
  the smoke rows. The other target texts are generated from the recipe's own templates (`runtime/make_hard_items.py`).
  The test fixtures `tests/fixtures/area11/*.jsonl` (byte copies of the first rows of the published run's generation
  and scoring outputs) carry LibriTTS-R speaker ids (`speaker`) and ASR transcripts (`asr_qwen`, `asr_whisper`) of speech
  synthesized in the voices of LibriTTS-R speakers; `runtime/tests/test_eval_canon.py` copies items, their target and
  reference texts and such transcripts verbatim from the pilot scoring outputs. The target texts of those rows are
  template texts of this recipe, not LibriTTS-R transcripts, but the voices, speaker ids and therefore the transcripts
  derive from LibriTTS-R; they are covered by this notice. No audio is redistributed.
- **Source:** LibriTTS-R, https://www.openslr.org/141/ (subsets `train_clean_100`, `dev_clean`, `test_clean`; archive
  md5 sums in `runtime/data-source/DATA.json`). Y. Koizumi, H. Zen, S. Karita, Y. Ding, K. Yatabe, N. Morioka,
  M. Bacchiani, Y. Zhang, W. Han, A. Bapna, "LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus",
  Interspeech 2023. LibriTTS-R restores the audio of LibriTTS (Zen et al., 2019), which is derived from LibriSpeech
  and the LibriVox audiobooks.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/legalcode. The corpus's `LICENSE.txt` states: "The LibriTTS-R corpus is
  made available by Google LLC under a Creative Commons Attribution 4.0 International License."
- **Changes:** utterances selected by `runtime/make_hard_items.py` and `runtime/make_items.py` (seeded draws by
  speaker and split, see `runtime/README.md` "Items"); each row reshaped to `{item, category, speaker, prompt_id,
  prompt_text, prompt_wav, text, ...}`. The transcript text itself is unchanged.
- **No endorsement:** the LibriTTS-R authors and Google LLC do not endorse this repository or its results.

## CosyVoice (Fun-CosyVoice3-0.5B-2512)

- **What:** no code or weights are redistributed. `runtime/fetch_cosyvoice.sh` clones CosyVoice (`github.com/QwenAudio/CosyVoice`) at commit
  `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` (Apache-2.0, its `LICENSE`) and `runtime/fetch_models.py` downloads
  `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` at a pinned revision; the README lists their terms and the "Responsible use"
  section.
