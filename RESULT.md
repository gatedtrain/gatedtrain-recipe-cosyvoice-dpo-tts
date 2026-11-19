Prototype result of a single registered run; not a benchmark claim; not affiliated with or endorsed by the owners of the models, datasets, or frameworks used.

# Result (2026-11-18): gate PASS, collected by ptc-hub 0.2.0, re-judged by gatedtrain 0.3.0

ptc-hub (also called ptc) and ptc-skel are earlier names of gatedtrain; see "Rename to gatedtrain 0.3.0".

Fun-CosyVoice3-0.5B, DPO on the LLM speech tokens with the published run's Whisper-WER-ranked pairs, launched with
`ptc-hub ext-launch` and collected with `ptc-hub ext-collect` (adapter `cosyvoice_dpo`, ptc-hub 0.2.0, design
revision 18). Candidate: the last cut of the run `train-retrain` (104 optimizer steps, `checkpoint-104/llm.pt` sha256
`caf2a750…ce11`; ptc model hash `d4352805…413a`); baseline: the base `llm.pt` (ptc model hash `97e23ca8…4e41`).
450 held-out items of `runtime/items-hard-test` (38 test_clean voices), one generation per item with seed 11.

| Clause (both blocking) | Metric | Baseline | DPO | Improvement, 95 % interval | Verdict |
|---|---|---|---|---|---|
| quality: superiority, margin 0, alpha 0.025 | `wer_qwen_canon` (Qwen3-ASR-1.7B, lower is better) | 0.1078 | 0.0708 | 0.0370 [0.0250, 0.0496] | PASS (achieved power 0.86) |
| speaker_guard: non-inferiority, margin 0.01, alpha 0.025 | `sim_prompt` (ECAPA cosine to the voice prompt, higher is better) | 0.8378 | 0.8393 | +0.0015 [-0.0028, +0.0081] | PASS (lower bound > -0.01) |

Interval: cluster bootstrap over speakers (38 clusters), percentile, 10,000 replicates, seed 20261117; 450 paired
items, no unpaired or error row (max_unpaired 0). Both clauses must pass (`all_clauses_must_pass`, each at its own
alpha; joint planned power 0.64). Tags: prototype, unpromotable, registration unsigned. Registration
`mm-a11-tts-ptc-rerun-v1` (sha256 `522c3f0c…0988`, `created_at` 2026-11-18T00:18:38Z, before the first GPU step), no
amendment, `post_hoc: false`. Files: `results/2026-11-18/` (`gate/` registration and spec; `ptc-run/` the gate report,
plan, lockfile, smoke record, consumption ledger and lineage edges of the ptc run; `arms-test-base/`,
`arms-test-cand/` the per-item arms; `records/` the dev selection and the rl summary).

Descriptive (never deciding), `wer_qwen_canon` per category, base to DPO: number 0.227 to 0.084, money 0.240 to
0.122, date 0.048 to 0.021, unit 0.092 to 0.079, long 0.054 to 0.046, name 0.239 to 0.234, natural_long 0.019 to
0.014, abbr 0.035 to 0.026 (homograph and twister 0 in both, n 4 each). The vendor's RL weights `llm.rl.pt` (reference
arm, outside ptc-hub): `wer_qwen_canon` 0.0980, `sim_prompt` 0.8343. Naturalness (MOS) was not measured.

What a pass shows is the registration's construct statement: fewer ASR-judged reading errors on these categories that
transfer to an independent ASR (the reward used Whisper, the gate Qwen3-ASR) on held-out texts and voices, not a
general naturalness gain; the two ASRs can share biases; Qwen3-ASR was used on the dev set for checkpoint selection.

### How the candidate was chosen

`train-full` (130 steps, cuts 26, 52, 78, 104, 130) and the base each generated and scored the 150 dev items
(`records/selection.json`): mean dev `wer_qwen_canon` base 0.0668, step 26 0.0524, 52 0.0373, 78 0.0404, 104 0.0366,
130 0.0383; `sim_prompt` 0.830 to 0.832 for every arm. Step 104 had the lowest dev WER. ptc-hub gates the last cut of a
run, so, by the registered rule, a new training launch with `--max-steps 104` (same settings, seed and inputs) was the
run and its last cut the candidate, without another dev selection. No test item was used for selection.

### Disclosures

- **The test set was already scored.** The published run (below) generated and scored these 450 items for the base,
  an earlier DPO candidate of the same recipe and the vendor's rl weights, and its result is published. The rerun's
  registration states this; its values are those of registration v2, fixed before that test set was first observed.
  The base arm is the same model with the same seeds, and its means equal the earlier ones (0.1078, 0.8378).
- **Evaluation image.** The first evaluation smoke ran in a plain CUDA image without the `ffmpeg` binary, which
  score_tts.py's Whisper pipeline needs to decode audio: all 20 rows were scoring errors. The image
  `launch/image/Dockerfile` (the same CUDA base plus `ffmpeg` from a pinned apt snapshot) was built and both smoke
  steps were repeated; the scoring code and the registration did not change. The aborted smoke outputs are kept
  outside this repository.
- **Smoke memory.** The smoke record's host RAM and swap values are readings after the run, not sampled peaks. During
  the full steps host memory was sampled every 15 s: peak RAM 12.5 GB, swap 7.2 GB.
- **Planned sample size of the guard.** ptc-hub plans 71 items for `speaker_guard` (non-inferiority at d = 0.01,
  sigma 0.03); registration v2's comment said 63, an arithmetic slip in a comment. 450 items cover both.
- **Not bit-reproducible.** bf16 training gives other weights on a rerun (the two training smokes differed at step 6);
  the candidate is identified by its hash.

### GPU hours

3.00 GPU-h on one RTX 4090 over 15 guarded steps: the first train and eval smokes (0.08, the eval smoke without
ffmpeg), the repeated smokes 0.10, `train-full` 0.08, dev arms 1.22 (6 x 150 items), `train-retrain` 0.07, test arms
1.45 (base 0.49, candidate 0.49, rl 0.47).

## Rename to gatedtrain 0.3.0 (2026-11-18)

The project was renamed from ptc / ptc-hub to gatedtrain before its first publication: distribution and import
`gatedtrain` 0.3.0, persisted identifiers `gatedtrain.<x>` (schema ids), `gatedtrain.ledger@0.3.0` (ledger writer id)
and `gatedtrain 0.3.0` (analysis code version), environment variables `GATEDTRAIN_*`. This recipe followed: distribution
`gatedtrain-recipe-cosyvoice-dpo-tts`, package `gatedtrain_recipe_cosyvoice_dpo_tts`, entry point group
`gatedtrain.adapters`, the variables of the launch wrapper and the run scripts `GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_{OUT,RUNTIME,IMAGE,SPEC,RECORDS}`
and `GATEDTRAIN_RECIPE_PYTHON`, the runtime projects `gatedtrain-recipe-cosyvoice-dpo-tts-runtime` and
`gatedtrain-recipe-cosyvoice-dpo-tts-eval` (both now with `license = "Apache-2.0"`; `uv.lock` regenerated with `uv lock`,
every locked package and version unchanged, only the project's own entry renamed), `runtime/prepare_gatedtrain_runtime.sh`
(was `prepare_ptc_runtime.sh`), the record schema ids the recipe writes (`gatedtrain_recipe_cosyvoice_dpo_tts.cut`,
`.run_args`, `.eval_run`, `.selection`; no code reads them back), and the rerun templates `launch/gate/` (a new rerun
registers as `mm-a11-tts-gatedtrain-rerun-v1`, workflow `mm-a11-tts-dpo-gatedtrain`, analysis code version gatedtrain
0.3.0).

Kept on purpose (historical evidence): everything under results/ (the 2026-11-18 run's frozen `gate/registration.yaml`
and `gate/spec.yaml` with registration id `mm-a11-tts-ptc-rerun-v1` and workflow `mm-a11-tts-dpo-ptc`, `ptc-run/`
with its `ptc_hub.*` schema ids and writer id, the arms with `ptc_hub.item_eval`, `records/selection.json` with
`ptc_recipe_cosyvoice_dpo_tts.selection`, and all of results/2026-11-17/), the byte-copied fixtures under
tests/fixtures/area11/, and the run's outputs, which carry the old record schema ids. `launch/image/Dockerfile` keeps
its RUN instruction byte-identical, including the package-list file name `/etc/ptc-a11-packages.txt`: the launch
records pin the image the run used by digest; only a comment was added.

Re-pinned (`pins.SCRIPTS`; old -> new SHA-256; the scorers `gen_tts.py`, `eval/score_tts.py`, `eval/wer_canon.py`, the
analysis script and the item sets are unchanged):

| File | Reason | SHA-256 before | SHA-256 now |
|---|---|---|---|
| `runtime/train_dpo.py` | `gatedtrain` in docstrings and a help text; the record schema ids it writes | `9813b91417e3336e86f1f61f64fbe796003c3dfd8e50a526901b30f4f0ae2e67` (`pins.RERUN_SCRIPTS`) | `bc2daa7c6540e806f6752569c48e491de72dc91bff54f41d0efe0ae05cc8e032` |
| `runtime/eval_arm.py` | as above | `a07b851a1edc5147cf2e322b8dcbdd90e70ddac840cb99e2feda77840b3f5452` (`pins.RERUN_SCRIPTS`) | `319b7f26a2efe1119e98cabe8cad10e982b67e29ffd9547aad06eba900802b81` |
| `runtime/fetch_models.py` | User-Agent `ptc-area11-fetch` -> `gatedtrain-recipe-cosyvoice-dpo-tts-fetch` | `8cd1d28c8c82964e0281ad26cbc2a77164acbdf64049a1f34b1d53d6a08ff7fc` (`pins.PUBLISHED_FETCH_MODELS_SHA256`) | `4626dc28e8dfdd18cf6011f9fab6d0ef82a169ee27812eaa8bc8e4fb6b54ae44` |
| `runtime/uv.lock` (not pinned by the adapter; recorded as the launch records' `venv_lock_sha256`) | project name | `d3b89cb69ae42f8b538262e91c921613b90e08a2dcf147e88909955ea02d07e4` | `76046194c7fb6dbee7612e98c3d3ee63a9ee3ba71f0eb074c40f13fa279c8544` |
| `runtime/eval/uv.lock` | project name | `432afef6b8f3a5f46cca6589050bb90c0c9f6d8496be138b5f5ff90b6f1aaf71` | `2d5a46e81b6c0d6d07735eccd5a4c3bd2aadfd0b51edfa36d8e6c400f1f89ed7` |

The run executed the earlier train_dpo.py and eval_arm.py (its launch records' `script_sha256`), from its own runtime
directory; the collect checks the scripts there against the records, so the re-pin does not touch the replay.

### The amendment

`results/2026-11-18/gate/registration-gatedtrain.yaml` (sha256
`6bf4ae907720e52231cf260ad00caa2a3a7a6761adc474731609c324166c64c2`) amends `registration.yaml` (`amends:
sha256:522c3f0c…0988`): besides a header comment it differs only in `analysis_code_version: gatedtrain 0.3.0`, `amends`
and `created_at` (2026-11-18T09:53:06Z). The evaluators' scorers did not change, so every `manifest_hash` is the same
and no `rename_proof` is needed; `post_hoc: false` (version-only, 05 section 3.7, R17-1; the report says
`registration_amendment: version_only`). `spec-gatedtrain.yaml` (sha256
`e247c7d23094374f642938b1f41352bb709b5b54275640fc07bcfdd39c6ddc24`) is `spec.yaml` with that registration URI and a
header comment; `gatedtrain validate spec-gatedtrain.yaml` exits 0 in the gate directory's layout.

### Re-judgement on a scratch replay (2026-11-18)

The run's records carry `ptc_hub.*` schema ids, which gatedtrain 0.3.0 refuses, so the gate was re-collected from
scratch copies with this repository and gatedtrain 0.3.0; nothing under the run's directories was written.

1. Copies of the launch record `train-retrain.launch.json`, the smoke record `smoke.json` and the four test arms
   (`arms-test-base/`, `arms-test-cand/`, each `wer_qwen_canon/` and `sim_prompt/` with their `outputs/`; apart from
   `outputs/` they equal the copies in results/2026-11-18/). Only the schema ids changed (`ptc_hub.launch_record`,
   `ptc_hub.smoke_record`, `ptc_hub.item_eval` -> `gatedtrain.*`).
2. The launch record copy was moved onto a scratch host-work directory: a copy of `gpu-step.sh` with only its `WORK=`
   line changed, the run's two guard-log lines copied under the new label; `host_work`, `gpu_step_sha256`,
   `host_config_sha256` and `label` recomputed with gatedtrain's own functions (label
   `mm-a11-tts-train-retrain-eeadc03c1839` -> `mm-a11-tts-train-retrain-fe705cfd4533`). The real guard log was not
   written.
3. A scratch gate directory with `registration.yaml`, `registration-gatedtrain.yaml`, `spec-gatedtrain.yaml`, the run's
   `trainer.yaml` and the `runtime` link to the run's runtime; `gatedtrain ext-collect spec-gatedtrain.yaml --adapter
   cosyvoice_dpo` on the run's own (unchanged, read-only) training output `train-retrain/`, as `launch/tts.sh collect`
   does.

Compared with the published report of run `run-20261118T033735Z-e885aa1b` (`ptc-run/report/gate_report.json`): status
PASS, both clauses equal field for field (quality: `delta_hat` 0.037037, interval [0.025026, 0.049595], p 1.0e-04,
achieved power 0.8562, 450 items, baseline 0.107829, candidate 0.070792; speaker_guard: `delta_hat` 0.001524, interval
[-0.002815, 0.008104], achieved power 0.9394, baseline 0.837823, candidate 0.839347), the model, checkpoint, item-set
and sampler hashes equal, 32 report fields in all. The differences are exactly the expected ones:
`analysis_code_version`, `registration_hash`, `registration_amends` (null -> `sha256:522c3f0c…`),
`registration_amendment` (null -> `version_only`), the report `schema` id, `lockfile_hash`, `run_id` and `created_at`.
The ledger (1,664 rows) is equal except `writer_id` (`gatedtrain.ledger@0.3.0`) and the run-id-derived `run_id` and
`sample_id`; the lineage tables are equal except `run_id`, and the lineage edges except the activity ids (run id), the
registration hash and the schema id.

## Earlier result (2026-11-17): gate PASS under the recipe's own analysis, not collected by ptc-hub

Fun-CosyVoice3-0.5B, DPO on the LLM speech tokens with a Whisper-large-v3-turbo WER reward (`dpo-train-k6`, final
weights `llm_dpo.pt`, sha256 `65cd9a45…f794`), against the base `llm.pt` (`69f43bd5…4970`), on the 450 held-out
items of `runtime/items-hard-test` (38 test_clean voices), one generation per item with seed 11.

| Clause (both blocking) | Metric | Baseline | DPO | Mean paired difference | One-sided 97.5 % bound | Verdict |
|---|---|---|---|---|---|---|
| quality: superiority, margin 0 | `wer_qwen_canon` (Qwen3-ASR-1.7B, lower is better) | 0.1078 | 0.0701 | -0.0378 | upper -0.0259 < 0 | PASS |
| speaker_guard: non-inferiority, margin 0.01 | `sim_prompt` (ECAPA cosine to the voice prompt, higher is better) | 0.8378 | 0.8390 | +0.0012 | lower -0.0027 > -0.01 | PASS |

Interval: cluster bootstrap over speakers (all items of a drawn voice resampled together; 38 clusters), 10,000
replicates, seed 20261117, percentile bound of the mean difference; no unpaired item (max_unpaired 0, no error row in
either arm). Both clauses must pass (intersection-union, each at its own alpha). Analysis: `runtime/analyze_gate.py`
(sha256 `7b815399…9c9a`) applying `results/2026-11-17/gate/registration-v2.yaml` (a redacted copy; the gate-time file is not
published); output `results/2026-11-17/evidence/gate-test-v2/gate_result.json`.

**Not collected by ptc-hub.** The gate was decided by the recipe's own script; ptc-hub 0.2.0 cannot express this
registration as it stands, and the run was launched outside `ptc-hub ext-launch` (see "Collecting with ptc-hub").

Descriptive (never deciding): Whisper-large-v3-turbo canon WER (the reward's ASR) 0.1213 to 0.0778. Per category,
`wer_qwen_canon` base to DPO: number 0.227 to 0.067, date 0.048 to 0.013, money 0.240 to 0.176, name 0.239 to
0.195, unit 0.092 to 0.078, long 0.054 to 0.047, natural_long 0.019 to 0.014, abbr 0.035 to 0.031 (homograph and
twister 0 in both). The vendor's RL weights `llm.rl.pt` (reference arm): `wer_qwen_canon` 0.0980, `sim_prompt`
0.8343; better than base on numbers and dates, worse on money (0.312). Naturalness (MOS) was not measured: no MOS
predictor is installed, and the registration leaves it out of the gate.

What a pass shows (the registration's statement): fewer ASR-judged reading errors on these categories (numbers,
money, units, dates, abbreviations, names, long sentences), not a general naturalness gain. The test texts come from
the same kind of programmatic templates as the training texts (disjoint templates and lexicon, checked), plus
held-out LibriTTS-R test_clean sentences; the voices are held-out speakers. The reward ASR (Whisper) and the deciding
ASR (Qwen3-ASR) can share biases; Qwen3-ASR was never used for training or for building pairs, but it was used on the
dev set for checkpoint selection.

### How the gate was fixed

1. **Training** (`runtime/run_train.sh`): 2,000 training items x 6 candidates with the base LLM, Whisper rewards,
   1,034 pairs (yield 0.517; chosen WER 0.045, rejected 0.274), DPO 2 epochs = 130 steps, checkpoints at steps 26,
   52, 78, 104 and the final weights (`evidence/dpo-train-k6/train_log.jsonl`, `pairs-train-k6/`,
   `cand-train-k6/`). The training and reward metadata (`dpo-train-k6/train_meta.json`,
   `reward-train-k6/reward_meta.json`) were removed from the publication because they recorded hashes of unpublished
   originals.
2. **Checkpoint selection on the dev set only** (`runtime/run_dev_select.sh`; 150 items with training templates and
   dev_clean voices): `wer_qwen_canon` base 0.0668, step 26 0.0440, step 52 0.0426, step 78 0.0409, step 104 0.0340,
   final 0.0329, the lowest, so final was selected; `sim_prompt` 0.832 for base and final (`evidence/dev-*-score/`).
   The planning values of the registration come from the dev paired differences (base vs final): `wer_qwen_canon`
   SD 0.0993 (sigma_plan 0.12, n required 283 <= 450 for a 0.02 effect at power 0.8), `sim_prompt` SD 0.0255
   (sigma_plan 0.03).
3. **Registration v1** (`gate/registration.yaml`, sha256 `ac51251b…032a`, `created_at` 2026-11-17T16:28:01Z) was
   written before any test-set generation (the first test generation started at 16:28:33Z). The analysis script was
   written and pinned (`b91429a9…`) before any test-set score existed; a dry run on the dev arms was refused on the
   item-set pin.
4. **The scorer defect.** The first test scoring (`evidence/test-*-score/summary.json`) left one error row per arm
   (base and DPO: the same `natural_long` item; rl: another): audio over 30 s made the Whisper pipeline switch to
   long-form decoding, which requires timestamps, so `score_tts.py` raised and the whole item row became an error,
   the deciding Qwen3-ASR WER included. Under max_unpaired 0 that alone fails the gate, for an evaluator defect, not
   a model property. The analysis script stopped on its first run with a KeyError in its descriptive part, after the
   pin checks and before printing any statistic.
5. **Amendment v2** (`gate/registration-v2.yaml`, `created_at` 18:23:26Z, `amends` v1): `score_tts.py`
   `552ee513…8ebf` to `4a5f29d6…9d34` passes `return_timestamps=True` to Whisper only for audio over 30 s; a 2-item
   smoke showed the short item's transcripts unchanged. Nothing else changed (`tests/test_gate_files.py` checks that
   the two files differ only in `registration_id`, `created_at`, `amends` and that one pin). The analysis script
   changed with it (`7b815399…`: descriptive statistics skip error rows). All three arms were re-scored from the same
   audio into `<tag>-score-v2`; the first scorings were kept. `post_hoc: false`: the maintainers decided that no
   result of the judged candidate had been observed when the amendment was written. Disclosure: the first scoring
   had written per-arm summaries to disk (`evidence/test-*-score/summary.json`, included here); the record of the
   amendment states that no test metric was looked at, and the analysis printed none.
6. **Gate** (`runtime/run_gate.sh`): registration v2 applied to the v2 scorings: PASS (above).

### GPU hours

14.95 GPU-h on one RTX 4090 over 29 logged steps (`runtime/README.md` "GPU cost"): smoke and pilots 0.70, pipeline
smokes 0.13, candidates 9.87, rewards 0.47, DPO 0.07, dev selection 1.13, test runs 1.67, scorer fix smoke and
re-scoring 0.91.

### Collecting with ptc-hub

The run predates this recipe's ptc adapter and was not launched through `ptc-hub ext-launch`, so it has no launch
record and no GPU-guard log entries, and `ptc-hub ext-collect` refuses it. It was therefore redone through ptc-hub
on 2026-11-18 (above).

### Reproducing

`runtime/run_train.sh`, `run_dev_select.sh dpo-train-k6`, `run_test.sh`, `run_test_rescore_v2.sh` and `run_gate.sh`
repeat the published run with the pinned scripts (`src/gatedtrain_recipe_cosyvoice_dpo_tts/pins.py`). The scripts are
byte-identical to the ones the run executed except train_dpo.py, which now writes the framework's launcher layout (the bytes the
run executed, `pins.PUBLISHED_TRAIN_DPO_SHA256`, were in a pre-release commit of this repository that is not in its
published history; `run_dev_select.sh` and `run_test.sh`
read their `llm_dpo*.pt` files) and fetch_models.py, whose User-Agent changed with the rename to gatedtrain 0.3.0
(above; `pins.PUBLISHED_FETCH_MODELS_SHA256`); only the `run_*.sh` wrappers are new (paths relative to the runtime instead of the
original host paths). bf16 training and GPU sampling are not bit-reproducible, so a new run gives new
weights; the published candidate is identified by its SHA-256.
