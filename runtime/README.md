# Runtime: Fun-CosyVoice3-0.5B DPO with a Whisper WER reward (prototype)

Two environments, both CPython 3.12 on Linux x86_64, each built from its own lock:

| Environment | Lock | Main packages | Scripts |
|---|---|---|---|
| main, `.venv` | `uv.lock` (`pyproject.toml`, `exclude-newer` 2026-11-01) | torch 2.5.1 + CUDA 12.4 wheels, torchaudio 2.5.1, transformers 4.51.3, onnxruntime-gpu 1.20.0, the CosyVoice requirements without serving parts | `gen_tts.py`, `sample_candidates.py`, `make_pairs.py`, `train_dpo.py`, `make_hard_items.py`, `make_items.py`, `analyze_gate.py`, `fetch_models.py` |
| evaluator, `eval/.venv` | `eval/uv.lock` (`eval/pyproject.toml`) | torch 2.8.0 + CUDA 12.8 wheels, qwen-asr 0.0.6 (transformers 4.57.6), speechbrain 1.1.2, jiwer 3.1.0 | `eval/score_tts.py`, `eval/score_candidates.py`, `eval/wer_canon.py` |

The two cannot share one environment (qwen-asr pins transformers 4.57.6; speechbrain needs torch >= 2.7.1; CosyVoice
runs on transformers 4.51.3 and torch 2.5.1). The CosyVoice source is imported from `src/CosyVoice`
(`fetch_cosyvoice.sh`: commit `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`, Matcha-TTS submodule
`dd9105b34bf2be2230f4aa1e4769fb586a3c824e`). Models and data: `fetch_models.py` (anonymous downloads; a set
`HF_TOKEN` stops it) into `models/` and `data-source/`, every file checked against the Hub's LFS SHA-256 or OpenSLR's
md5 and listed in `models/MODELS.json` and `data-source/DATA.json`. All GPU steps ran on one RTX 4090 (24 GB) with
`HF_HUB_OFFLINE=1` and `HF_TOKEN` unset.

## Model and decoding

Fun-CosyVoice3-0.5B-2512 (revision `29e01c4e…`): a Qwen2-based LLM maps text plus a voice prompt (transcript and
speech tokens) to speech tokens; a flow-matching decoder and a HiFT vocoder turn them into 24 kHz audio. Only the
LLM (`llm.pt`, sha256 `69f43bd5…`) is trained; the vendor's `llm.rl.pt` (`74d34b01…`) is a reference arm, reported and
never deciding. Text goes to the LLM raw (`text_frontend=False`): the CosyVoice text normaliser fell back to a
number speller that mangles thousands separators and decimals when its resources could not be downloaded, so the
gate measures the model on written-form text. Zero-shot, non-streaming, seed 11 per item (`gen_tts.py`).

## Items

`make_hard_items.py` builds programmatic hard sentences (numbers, dates, times, money, units, initialisms,
homographs, tongue twisters, rare names, long multi-clause sentences) from disjoint template sets for training and
evaluation, plus long natural LibriTTS-R sentences, each paired with a voice prompt of a LibriTTS-R speaker:

| Set | Rule | Rows | Speakers | SHA-256 of `items.jsonl` |
|---|---|---|---|---|
| `items-hard-train` | train templates, voices and natural sentences from train_clean_100 | 2,000 (1,600 hard, 400 natural) | 243 | `bcd79094…4488` |
| `items-hard-dev` | train templates (own RNG stream, every train text rejected), voices and sentences from dev_clean | 150 (120, 30) | 38 | `b9ac4220…b1a4` |
| `items-hard-test` | test templates and lexicon, voices and sentences from test_clean | 450 (360, 90) | 38 | `33cf0b87…098d090` |
| `items-hard-pilot` | test templates, dev_clean voices (base-model pilot only) | 64 | 38 | `c68e12fa…7e07c` |
| `items-smoke` | `make_items.py`, plain dev_clean utterances with real recordings | 14 | 7 | `6136abb5…4e46` |

The test set shares no speaker, prompt, text, reference text or natural source with the training set (checked at
build, `item_set_meta.json` `checked_disjoint_from`); the dev set is disjoint from both.

## Pipeline (`run_train.sh`)

1. `sample_candidates.py`: K = 6 zero-shot candidates per training item with the base LLM (seed 11), keeping the LLM
   tokens and the sampled stop token (12,000 candidates, 9.87 GPU-h, the bulk of the cost).
2. `eval/score_candidates.py`: Whisper-large-v3-turbo WER (canon-1, `wer_canon.py`) of every candidate; a failed
   batch is retried one item at a time; audio over 30 s is left unscored (3 candidates).
3. `make_pairs.py`: per item, the lowest-WER candidate against the highest, if the WER margin is at least 0.05 and
   both durations are within 0.5-2.5 times the expected length: 1,034 pairs from 2,000 items (chosen WER 0.045,
   rejected 0.274).
4. `train_dpo.py`: sigmoid DPO (beta 0.1) on the summed log-probabilities of the speech tokens, built exactly as
   zero-shot inference conditions the LLM; frozen reference = the initial LLM, its log-probabilities cached once;
   full fine-tune of the LLM, AdamW lr 1e-6, warm-up 10, micro-batch 2 pairs x accumulation 8, 2 epochs = 130
   steps, bf16 autocast, clip 1.0, seed 11, checkpoints every 26 steps and at the end. 190 s wall, peak 15.5 GiB.
   The published run wrote `llm_dpo.step<N>.pt` and `llm_dpo.pt` (train_dpo.py `sha256:e19f6cc8…`, a pre-release
   version of this repository that is not in its published history); the current train_dpo.py writes the gatedtrain layout instead ("Under gatedtrain" below), with the same
   training computation (checked on CPU: identical weights and logs, fp32 and bf16, tiny model).

## Selection and test (`run_dev_select.sh`, `run_test.sh`, `run_test_rescore_v2.sh`, `run_gate.sh`)

`run_dev_select.sh dpo-train-k6` generates and scores the dev set with the base and every checkpoint; the lowest dev
`wer_qwen_canon` selects the candidate. `run_test.sh` generates and scores the test set with base, candidate and the
vendor's rl weights; `run_gate.sh` applies the registration with `analyze_gate.py`. Each step logs to
`$GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RECORDS` (default `out/records/`) and prints its UTC start and end for the GPU log.

## Metrics (`eval/score_tts.py`)

Per item: `wer_qwen_canon` (deciding): WER of Qwen3-ASR-1.7B's transcript against the item's reference (`ref_text`
when present, else `text`) after Whisper's English normaliser and canon-1 (`wer_canon.py`: times to one token, runs
of single letters joined); `sim_prompt`: cosine similarity of SpeechBrain ECAPA embeddings of the output and the
prompt recording (16 kHz); `wer_whisper_canon`: the same WER with the reward's ASR, reported only. An item whose
audio is missing or fails to score is an error row, never a score. No MOS predictor is installed (`mos_scores`
raises), so naturalness is not measured.

## GPU cost (measured, RTX 4090)

| Step | GPU-h |
|---|---|
| smoke and pilots (generation, scoring, frontend choice) | 0.70 |
| pipeline smokes (candidates, rewards, DPO exact and bf16) | 0.13 |
| candidates, 2,000 items x 6 | 9.87 |
| rewards (first attempt with batch failures, then per-item retry) | 0.47 |
| DPO (20-step smoke on the full pairs, then 130 steps) | 0.07 |
| dev selection (6 arms x 150 items) | 1.13 |
| test runs (3 arms x 450 items, generation and first scoring) | 1.67 |
| scorer fix smoke and test re-scoring v2 | 0.91 |
| **total** | **14.95** |

Per test arm: generation about 25 minutes (0.42 GPU-h), scoring 10 to 26 minutes.

## Under gatedtrain (adapter `cosyvoice_dpo`)

The gate is rerun through `gatedtrain ext-launch` and collected by `gatedtrain ext-collect` with this recipe's adapter
(`src/gatedtrain_recipe_cosyvoice_dpo_tts/`: `rows.py`, `adapter.py`). The reused inputs are the published run's pairs
(`pairs-train-k6`, 1,034 pairs), the candidates they were made from (`cand-train-k6`, tokens only) and the reference
log-probs of the base on those pairs (`dpo-train-k6/ref_logps.jsonl`); no candidate is sampled again. The new DPO
weights are a new candidate.

### Outputs of `train_dpo.py`

`run_args.json`, `env-freeze.txt`, `env-runtime.json`, `samples.jsonl` (one line per pair of each optimizer step,
before the step), `checkpoint-<N>/` (`llm.pt`, `base_ref.json`, `cut.json` written last, atomic rename),
`train_log.jsonl`, `ref_logps.jsonl`, `train_meta.json`; `--out` is a new or empty directory and `--profile` (smoke,
full) is recorded. The adapter re-derives the pair order (`schedule`) from `run_args.json` and requires `samples.jsonl`
to be exactly it (consumption precision `row`, observed); a cut's model is the base directory with its `llm.pt`
replaced, and its model hash is that directory's manifest hash.

### Preparing the runtime

After the setup above: `prepare_gatedtrain_runtime.sh` makes the speaker model's links in `eval/ecapa-cache/` (score_tts.py
cannot create them in a read-only mount), removes Matcha-TTS's unused `.env.example` (ext-launch refuses a credential-like
file name in a mount) and checks that both venvs, the models and the prompt audio are in place. The base directory must
hold no `.cache/`.

### Launch request facts

| Host | Container | Mode |
|---|---|---|
| the interpreter tree's directory (`.venv/pyvenv.cfg` `home`, alias and resolved) | same | ro |
| `<runtime>` (scripts, both venvs, models, data-source, item sets, src/CosyVoice) | same | ro |
| training: `<out>/inputs/pairs`, `<out>/inputs/cand`, `<out>/inputs/ref` | `/inputs/pairs`, `/inputs/cand`, `/inputs/ref` | ro |
| evaluation of a cut: `<out>/<run>/checkpoint-<N>` | `/inputs/cut` | ro |
| `<out>/<step>` (new, empty, owned by the launching user) | `/out` | rw |
| tmpfs | `/tmp` | |

`<out>` is `{output_root}/cosyvoice-dpo-tts` (`GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_OUT`). The image (`GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_IMAGE`) must
provide the `ffmpeg` binary, which score_tts.py's Whisper pipeline uses to decode audio (a plain CUDA image fails every
scoring row); `launch/image/Dockerfile` builds it from a pinned CUDA base and a pinned apt snapshot. Environment: `HF_HUB_OFFLINE=1`,
`HOME=/tmp/home`, `XDG_CACHE_HOME=/tmp/cache`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`,
`TOKENIZERS_PARALLELISM=false`. Network `none`, `gpus: true`, argv[0] `<runtime>/.venv/bin/python`, framework
`{name: torch, version: <the venv's torch, 2.5.1+cu124>}`, training seed 11. Requests are written by
`python -m gatedtrain_recipe_cosyvoice_dpo_tts.launch` and the steps by `launch/tts.sh`:

```
train smoke: train_dpo.py --model-dir <runtime>/models/fun-cosyvoice3-0.5b-2512 --candidates /inputs/cand
             --pairs /inputs/pairs --ref-cache /inputs/ref/ref_logps.jsonl --out /out --profile smoke --beta 0.1
             --lr 1e-6 --epochs 2 --micro-batch 2 --grad-accum 8 --warmup-steps 10 --clip 1.0 --seed 11 --amp bf16
             --max-steps 6 --save-every 3
train full:  the same with --profile full --save-every 26 (130 steps, cuts 26, 52, 78, 104, 130)
evaluation:  eval_arm.py --llm base|rl|/inputs/cut --item-set items-hard-test|items-hard-dev --out /out --seed 11
             [--limit 20, smoke only]
```

`eval_arm.py` runs gen_tts.py (this venv) and then eval/score_tts.py (eval/.venv) in one container; the arms are made
on the host by `python -m gatedtrain_recipe_cosyvoice_dpo_tts.arms` (one per metric, `wer_qwen_canon` and `sim_prompt`) and
their hashes filled by `gatedtrain ext-arm-fill` (`--base` for a cut).

### Steps and GPU cost (estimates from the published run's measurements)

| Step (`launch/tts.sh`) | What | GPU-h |
|---|---|---|
| `stage-inputs` | copy pairs, candidate tokens and reference log-probs into `<out>/inputs` (host) | 0 |
| `gate-init`, `gate-freeze` | registration to `<out>/gate/`; after `created_at`: spec, trainer config, `gatedtrain validate` (host) | 0 |
| `train-smoke` | 6 steps, cuts 3 and 6 (model load, two 2 GB cuts) | 0.05 |
| `eval-smoke-base` | base, first 20 test items, generation and scoring | 0.05 |
| `smoke-record` | arms of the smoke baseline (both metrics), `smoke.json` (host) | 0 |
| `train-full` | 130 steps (190 s of steps, loads and five cuts) | 0.08 |
| `dev-base`, `dev-cut <N>` x 5 | 150 dev items per arm (1.13 GPU-h for six arms in the published run) | 1.13 |
| `select` | `selection.json` (host) | 0 |
| `train-retrain` | only if the selected cut is not the last | 0.08 |
| `test-arm base`, `cand`, `rl` | 450 test items per arm: generation 0.42 + scoring 0.17-0.43 | 1.8 |
| `arms`, `collect` | arms, ext-arm-fill, ext-collect (host) | 0 |
| **total** | without retraining | **3.1** |

Measured in the 2026-11-18 run (`RESULT.md`): train-full 0.08, six dev arms 1.22, train-retrain 0.07, three test arms
1.45 (0.47-0.49 each), smokes 0.18 including one repeated pair; 3.00 in all.

### Checkpoint selection (pre-registered; `select.py`)

The rule of the published run, written out: every cut of the full run generates and scores `items-hard-dev` (150
items) with seed 11 (the base too, reported only); the statistic is the cut's mean `wer_qwen_canon` from score_tts.py's
`summary.json` (error rows skipped); a cut with an error row is eligible only if every cut has one (fewest first); the
lowest statistic is selected, ties to the later step. gatedtrain gates the last cut of a run, so a selected cut before the
last is not collected from this run: a new full launch with `--max-steps N` (warm-up then constant LR, so the same
schedule up to N; not bit-identical under bf16) gives the candidate, its last cut, which is not dev-selected again.

### The smoke record

`python -m gatedtrain_recipe_cosyvoice_dpo_tts.smoke` computes `smoke.json` from the training smoke (peak memory,
seconds per step, projection for 130 steps) and the smoke baseline arms of both clause evaluators (20 base scores each
of `wer_qwen_canon` and `sim_prompt`, `baseline_scores_by_evaluator`, design revision 18, R18-8). Each is judged with
its own row and direction: a score on the metric's best bound (WER 0) is left out of the one-value rule, at least a
quarter of the scores (and at least 2) must lie off it, and the share of the most common other value is taken over the
whole sample. The published base's first 20 test items (14 at WER 0, six distinct positive WERs) pass that rule.

### Gate files (`launch/gate/`)

`registration.yaml` is the rerun's pre-registration in the design revision 18 form: clauses `quality`
(`wer_qwen_canon`, superiority, margin 0, alpha 0.025, power 0.8, planning effect 0.02, sigma_plan 0.12) and
`speaker_guard` (`sim_prompt`, non-inferiority, margin 0.01, alpha 0.025, power 0.8, sigma_plan 0.03), both blocking,
`multiplicity: {plan: all_clauses_must_pass}`, `cluster_key: speaker`, `bootstrap_variant: percentile`, 10,000
replicates, seed 20261117, max_unpaired 0, K 1, eval seed 11, the item set by its manifest hash, the sampler
`rows.SAMPLER`. gatedtrain plans 283 items for `quality` and 71 for `speaker_guard` (registration-v2's comment said 63).
Its comments register the dev-selection and retraining rules and disclose that this test set was already scored for an
earlier candidate. It has no `created_at`: `launch/tts.sh gate-init` copies it to `<out>/gate/`, `created_at` is added
there right before the first GPU step, and `gate-freeze` writes `<out>/gate/spec.yaml` from `spec.template.yaml` (the
gate's `evaluator` the list of both clause evaluators, the registration pinned by its SHA-256), `trainer.yaml` and the
`runtime` link, then runs `gatedtrain validate`.
