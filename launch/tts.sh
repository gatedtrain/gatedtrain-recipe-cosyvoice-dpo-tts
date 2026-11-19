#!/usr/bin/env bash
# The gate run of this recipe under gatedtrain: every GPU step is written as an `ext-launch` request, checked and turned
# into a launch record and a gpu-step.sh command; host steps (arms, smoke record, selection, collect) run on CPU.
# Nothing here names a machine: the paths come from the environment and gatedtrain's host layout.
#
# Environment:
#   GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_OUT      the recipe's directory under the ext output root: {output_root}/cosyvoice-dpo-tts
#   GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RUNTIME  the prepared runtime (runtime/README.md "Under gatedtrain"; prepare_gatedtrain_runtime.sh)
#   GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_IMAGE    the pinned image, <name>@sha256:<digest>
#   GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_SPEC     the spec (default <out>/gate/spec.yaml, written by gate-freeze), for --spec
#                                         and ext-collect
#   GATEDTRAIN_RECIPE_PYTHON                     an interpreter with gatedtrain and this package (default python3)
#   RUN=1                                 also run `ext-launch --recheck` and the printed command (GPU); otherwise the
#                                         command is only written to $OUT/records/<step>.cmds
#
# Steps, in order (GPU-h: the published run's measured cost on one RTX 4090, README "GPU cost"):
#   stage-inputs PAIRS_DIR CAND_DIR REF_CACHE  copy the reused pairs, candidates (no audio) and reference log-probs
#   gate-init                      copy launch/gate/registration.yaml (not frozen) to <out>/gate/
#   gate-freeze                    after `created_at` was added: spec.yaml (registration hash pinned), trainer.yaml,
#                                  the `runtime` link; check the spec and registration with gatedtrain (no GPU)
#   train-smoke                    GPU ~0.03  6 steps, cuts 3 and 6
#   eval-smoke-base                GPU ~0.05  base on the first 20 test items
#   smoke-record RAM_GB SWAP_GB [SCORE_PEAK_GB]  arms of the smoke baseline (both metrics), smoke.json (verdict
#                                  computed per clause evaluator, R18-8)
#   train-full                     GPU ~0.10  130 steps, cuts 26 52 78 104 130
#   dev-base, dev-cut STEP         GPU ~0.19 each (6 arms x 150 dev items: 1.13 GPU-h in the published run)
#   select                         selection.json by the pre-registered rule (select.py)
#   train-retrain                  GPU ~0.10  only when select names a cut before the last
#   test-arm base|cand|rl          GPU ~0.6 each (generation 0.42 + scoring 0.17-0.43)
#   arms base|cand|rl              per-metric arms, ext-arm-fill (base, cand)
#   collect                        ext-collect of the training run against the two test arms
# usage: launch/tts.sh <step> [args]
set -euo pipefail
OUT="${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_OUT:?set GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_OUT}"
RT="${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RUNTIME:?set GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RUNTIME}"
PY="${GATEDTRAIN_RECIPE_PYTHON:-python3}"
SPEC="${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_SPEC:-$OUT/gate/spec.yaml}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REC="$OUT/records"
ADAPTER=cosyvoice_dpo
BASE="$RT/models/fun-cosyvoice3-0.5b-2512"
mkdir -p "$REC"
HW="$("$PY" -c 'from gatedtrain.ext.launch import host_config; print(host_config().host_work)')"

image() { echo "${GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_IMAGE:?set GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_IMAGE}"; }

# launch <name> <profile> <train|eval> [request args...]: request, record, command; RUN=1 runs it.
launch() {
  local name="$1" profile="$2" kind="$3"
  shift 3
  local d="$OUT/$name" extra=()
  if [ -e "$d" ]; then echo "$d exists: a launch writes into a new, empty directory" >&2; exit 2; fi
  mkdir "$d"
  "$PY" -m gatedtrain_recipe_cosyvoice_dpo_tts.launch "$kind" --profile "$profile" --label "mm-a11-tts-$name" \
    --output-dir "$d" --image "$(image)" --host-work "$HW" "$@" > "$REC/$name.request.yaml"
  if [ "$profile" = full ]; then extra+=(--smoke-record "$REC/smoke.json"); fi
  # --spec: a training launch binds the spec's base and pairs to read-only mounts; a full launch's smoke record is
  # checked against every clause evaluator of the spec's registration (R18-8).
  if [ -e "$SPEC" ] && { [ "$kind" = train ] || [ "$profile" = full ]; }; then extra+=(--spec "$SPEC"); fi
  if [ "$profile" = full ] && [ ! -e "$SPEC" ]; then echo "$SPEC is missing: run gate-freeze first" >&2; exit 2; fi
  "$PY" -m gatedtrain ext-launch --adapter "$ADAPTER" --profile "$profile" --request "$REC/$name.request.yaml" \
    --gpu-log "$HW/gpu-log.md" --out-record "$REC/$name.launch.json" "${extra[@]}" > "$REC/$name.cmds"
  echo "record $REC/$name.launch.json; command $REC/$name.cmds"
  if [ "${RUN:-0}" = 1 ]; then
    "$PY" -m gatedtrain ext-launch --adapter "$ADAPTER" --recheck "$REC/$name.launch.json"
    bash "$REC/$name.cmds" > "$REC/$name.log" 2>&1
    echo "done $name (log $REC/$name.log)"
  fi
}

train_inputs() {
  echo --pairs "$OUT/inputs/pairs" --candidates "$OUT/inputs/cand" --ref-cache "$OUT/inputs/ref/ref_logps.jsonl"
}

# candidate: the cut the gate evaluates (selection.json): the full run's last cut, or the retrained run's.
candidate() {
  "$PY" - "$REC/selection.json" "$OUT" <<'PYEOF'
import json, sys
sel = json.load(open(sys.argv[1]))
run = "train-full" if sel["selected_step"] == sel["last_step"] else "train-retrain"
print(f"{sys.argv[2]}/{run}/checkpoint-{sel['selected_step']}")
PYEOF
}

step="${1:?usage: launch/tts.sh <step> [args]}"
shift
case "$step" in
  stage-inputs)
    pairs="${1:?PAIRS_DIR}" cand="${2:?CAND_DIR}" ref="${3:?REF_CACHE}"
    if [ -e "$OUT/inputs" ]; then echo "$OUT/inputs exists" >&2; exit 2; fi
    mkdir -p "$OUT/inputs/pairs" "$OUT/inputs/cand" "$OUT/inputs/ref"
    cp -p "$pairs/pairs.jsonl" "$pairs/pairs_meta.json" "$OUT/inputs/pairs/"
    cp -p "$cand/candidates.jsonl" "$cand/candidates_meta.json" "$OUT/inputs/cand/"
    cp -rp "$cand/tokens" "$OUT/inputs/cand/tokens"
    cp -p "$ref" "$OUT/inputs/ref/ref_logps.jsonl"
    (cd "$OUT/inputs" && find . -type f -print0 | sort -z | xargs -0 sha256sum > "$REC/inputs.sha256")
    echo "staged into $OUT/inputs ($(wc -l < "$REC/inputs.sha256") files, hashes in $REC/inputs.sha256)"
    ;;
  gate-init)
    mkdir -p "$OUT/gate"
    if [ -e "$OUT/gate/registration.yaml" ]; then echo "$OUT/gate/registration.yaml exists" >&2; exit 2; fi
    cp "$HERE/gate/registration.yaml" "$OUT/gate/registration.yaml"
    echo "add created_at (RFC 3339, UTC) to $OUT/gate/registration.yaml right before the first GPU step, then gate-freeze"
    ;;
  gate-freeze)
    G="$OUT/gate"
    if ! grep -q '^created_at: ' "$G/registration.yaml"; then
      echo "$G/registration.yaml has no created_at: it is not frozen" >&2; exit 2
    fi
    if [ -e "$G/spec.yaml" ]; then echo "$G/spec.yaml exists" >&2; exit 2; fi
    ln -sfn "$RT" "$G/runtime"
    printf 'backend: %s\nprofile: full\n' "$ADAPTER" > "$G/trainer.yaml"
    h="$(sha256sum "$G/registration.yaml" | cut -d' ' -f1)"
    sed "s/{registration_sha256}/$h/" "$HERE/gate/spec.template.yaml" > "$G/spec.yaml"
    "$PY" -m gatedtrain validate "$G/spec.yaml"
    ;;
  train-smoke)
    # shellcheck disable=SC2046
    launch train-smoke smoke train --gpu-est-h 0.05 $(train_inputs)
    ;;
  eval-smoke-base)
    launch eval-smoke-base smoke eval --gpu-est-h 0.1 --llm base --item-set items-hard-test --limit 20
    ;;
  smoke-record)
    ram="${1:?RAM_GB}" swap="${2:?SWAP_GB}"
    "$PY" -m gatedtrain_recipe_cosyvoice_dpo_tts.arms --eval-run "$OUT/eval-smoke-base" --items "$RT/items-hard-test" \
      --checkpoint "$BASE" --out "$OUT/arms-smoke-base"
    for m in wer_qwen_canon sim_prompt; do
      "$PY" -m gatedtrain ext-arm-fill --adapter "$ADAPTER" --checkpoint "$BASE" --items "$RT/items-hard-test" \
        "$OUT/arms-smoke-base/$m"
    done
    "$PY" -m gatedtrain_recipe_cosyvoice_dpo_tts.smoke --train-out "$OUT/train-smoke" --baseline-eval "$OUT/eval-smoke-base" \
      --baseline-arms "$OUT/arms-smoke-base" --base "$BASE" --items "$RT/items-hard-test" \
      --gpu-log "$HW/gpu-log.md" --host-ram-peak-gb "$ram" --swap-peak-gb "$swap" \
      ${3:+--score-peak-gb "$3"} --out "$REC/smoke.json"
    ;;
  train-full)
    # shellcheck disable=SC2046
    launch train-full full train $(train_inputs)
    ;;
  dev-base)
    launch dev-base full eval --llm base --item-set items-hard-dev
    ;;
  dev-cut)
    s="${1:?STEP}"
    launch "dev-cut$s" full eval --llm "$OUT/train-full/checkpoint-$s" --item-set items-hard-dev
    ;;
  select)
    args=()
    for c in "$OUT"/train-full/checkpoint-*; do
      s="${c##*-}"
      args+=(--dev "$s=$OUT/dev-cut$s")
    done
    "$PY" -m gatedtrain_recipe_cosyvoice_dpo_tts.select --run "$OUT/train-full" "${args[@]}" --base "$OUT/dev-base" \
      --out "$REC/selection.json"
    ;;
  train-retrain)
    s="$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("" if d["selected_step"] == d["last_step"] else d["selected_step"])' "$REC/selection.json")"
    if [ -z "$s" ]; then echo "selection.json selects the last cut: no retraining" >&2; exit 2; fi
    # shellcheck disable=SC2046
    launch train-retrain full train $(train_inputs) --max-steps "$s"
    ;;
  test-arm)
    which="${1:?base|cand|rl}"
    case "$which" in
      base|rl) launch "test-$which" full eval --llm "$which" --item-set items-hard-test ;;
      cand) launch test-cand full eval --llm "$(candidate)" --item-set items-hard-test ;;
      *) echo "test-arm base|cand|rl" >&2; exit 2 ;;
    esac
    ;;
  arms)
    which="${1:?base|cand|rl}"
    case "$which" in
      base) ck="$BASE"; fill=() ;;
      cand) ck="$(candidate)"; fill=(--base "$BASE") ;;
      rl) ck=""; fill=() ;;
      *) echo "arms base|cand|rl" >&2; exit 2 ;;
    esac
    if [ "$which" = rl ]; then
      # The vendor's llm.rl.pt is a descriptive reference, outside gatedtrain: its score_tts.py summary is the report.
      cp "$OUT/test-rl/score/summary.json" "$REC/test-rl.summary.json"
      echo "rl (descriptive): $REC/test-rl.summary.json"
      exit 0
    fi
    "$PY" -m gatedtrain_recipe_cosyvoice_dpo_tts.arms --eval-run "$OUT/test-$which" --items "$RT/items-hard-test" \
      --checkpoint "$ck" --out "$OUT/arms-test-$which"
    for m in wer_qwen_canon sim_prompt; do
      "$PY" -m gatedtrain ext-arm-fill --adapter "$ADAPTER" --checkpoint "$ck" --items "$RT/items-hard-test" \
        "${fill[@]}" "$OUT/arms-test-$which/$m"
    done
    ;;
  collect)
    run="$(dirname "$(candidate)")"
    # One arm pair per clause, in the order of the gate's evaluators (R18-1): quality (wer_qwen_canon), then
    # speaker_guard (sim_prompt).
    "$PY" -m gatedtrain ext-collect "$SPEC" --adapter "$ADAPTER" --stage-out "$run" \
      --launch-record "$REC/$(basename "$run").launch.json" --smoke-record "$REC/smoke.json" \
      --eval-baseline "$OUT/arms-test-base/wer_qwen_canon" --eval-candidate "$OUT/arms-test-cand/wer_qwen_canon" \
      --eval-baseline "$OUT/arms-test-base/sim_prompt" --eval-candidate "$OUT/arms-test-cand/sim_prompt" \
      --runs-root "$OUT/runs"
    ;;
  *)
    echo "unknown step $step" >&2
    exit 2
    ;;
esac
