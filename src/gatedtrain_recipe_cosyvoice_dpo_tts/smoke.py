"""The smoke record (`gatedtrain.smoke_record`, MULTIMODAL-SPEC MM-38, MM-39) of this recipe, from a smoke training run
and a smoke baseline arm. Host side, CPU only.

Measurements:
- `peak_gpu_mem_gb`: the larger of the training smoke's peak (train_log.jsonl `peak_gpu_gib`, the maximum over steps)
  and the baseline generation's (gen_meta.json `peak_gpu_gib`), GiB converted to GB; score_tts.py records no peak, so
  `--score-peak-gb` (from the operator's GPU memory sampler, nvidia-smi) is added when given;
- `seconds_per_step`: the mean `seconds` of the training smoke's steps after the first (the first carries warm-up);
- `projected_gpu_h`: (`--full-steps` x seconds_per_step + `--overhead-s`) / 3600, the full training launch's estimate;
- `baseline_scores_by_evaluator` (design revision 18, R18-8): the scores of the converted smoke baseline arm of each
  clause's evaluator (`arms.py`; the first `--limit` items of the registered item set), `baseline_scores` the first
  (deciding) evaluator's, `error_count` the error lines of all of them;
- `host_ram_peak_gb`, `swap_peak_gb`: given by the operator (the host sampler), `gpu_log_total_h`: the running total of
  `<HOST_WORK>/gpu-log.md` (gatedtrain's reader);
- `base_model_hash`: the adapter's identity of the base directory; `item_set_hash`: the manifest hash of the
  registered item set directory.
The verdict is gatedtrain's `smoke_verdict` with each evaluator's row and direction (a score on the metric's best bound,
WER 0, is exempt from the one-value rule, R18-8); the record PASSes only when every evaluator's does, and its reasons
name the evaluator.

usage: python -m gatedtrain_recipe_cosyvoice_dpo_tts.smoke --train-out DIR --baseline-eval DIR --baseline-arms DIR
           --base BASE_DIR --items ITEMS_DIR --gpu-log GPU_LOG_MD --host-ram-peak-gb X --swap-peak-gb Y --out smoke.json
           [--full-steps 130] [--overhead-s 300] [--score-peak-gb Z]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .rows import EVALUATOR_ROWS, NAME, SIM_ID, WER_ID

EVALUATORS = (WER_ID, SIM_ID)                      # the clauses' evaluators, in clause order
METRIC_OF = {WER_ID: "wer_qwen_canon", SIM_ID: "sim_prompt"}

GIB = 2 ** 30 / 1e9


def build(*, train_out: Path, baseline_eval: Path, baseline_arms: Path, base: Path, items: Path, gpu_log: Path,
          host_ram_peak_gb: float, swap_peak_gb: float, full_steps: int, overhead_s: float,
          score_peak_gb: float | None = None, evaluators: tuple[str, ...] = EVALUATORS) -> dict:
    """`baseline_arms`: the directory holding one converted arm per metric (`<baseline_arms>/<metric>/`)."""
    from gatedtrain.canonical import manifest_hash
    from gatedtrain.ext.launch import gpu_used_h
    from gatedtrain.ext.registry import adapter as get_adapter
    from gatedtrain.ext.smoke import SMOKE_SCHEMA, smoke_verdict

    log = [json.loads(line) for line in (Path(train_out) / "train_log.jsonl").read_text().splitlines() if line.strip()]
    if len(log) < 2:
        raise ValueError("the training smoke logged fewer than 2 steps; seconds_per_step needs a step after the first")
    gen_meta = json.loads((Path(baseline_eval) / "gen" / "gen_meta.json").read_text(encoding="utf-8"))
    peaks = [max(r.get("peak_gpu_gib", 0) for r in log) * GIB, gen_meta["peak_gpu_gib"] * GIB]
    if score_peak_gb is not None:
        peaks.append(score_peak_gb)
    by: dict[str, list[float]] = {}
    errors = 0
    for ev in evaluators:
        arm = Path(baseline_arms) / METRIC_OF[ev]
        meta = json.loads((arm / "eval_meta.json").read_text(encoding="utf-8"))
        if meta.get("evaluator_id") != ev:
            raise ValueError(f"{arm} is the arm of {meta.get('evaluator_id')}, not of {ev}")
        lines = [json.loads(x) for x in (arm / "items.jsonl").read_text().splitlines() if x.strip()]
        by[ev] = [x["score"] for x in lines if x["score"] is not None]
        errors += sum(1 for x in lines if x["error"] is not None)
    sps = sum(r["seconds"] for r in log[1:]) / (len(log) - 1)
    record = {
        "schema": SMOKE_SCHEMA, "adapter": NAME, "area": "tts",
        "base_model_hash": get_adapter(NAME).identity(Path(base))["model_hash"],
        "item_set_hash": manifest_hash(Path(items)),
        "peak_gpu_mem_gb": round(max(peaks), 2), "host_ram_peak_gb": host_ram_peak_gb, "swap_peak_gb": swap_peak_gb,
        "seconds_per_step": round(sps, 3), "projected_gpu_h": round((full_steps * sps + overhead_s) / 3600, 3),
        "gpu_log_total_h": gpu_used_h(Path(gpu_log)),
        "baseline_scores": by[evaluators[0]], "baseline_scores_by_evaluator": by, "error_count": errors,
    }
    reasons: list[str] = []
    for ev in evaluators:
        row = EVALUATOR_ROWS[ev]
        v = smoke_verdict({**record, "baseline_scores": by[ev]}, score_range=row["score_range"], binary=row["binary"],
                          direction=row["direction"])
        reasons += [r if len(evaluators) == 1 else f"{r} ({ev})" for r in v["reasons"]]
    record.update(verdict="FAIL" if reasons else "PASS", reasons=list(dict.fromkeys(reasons)))
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gatedtrain_recipe_cosyvoice_dpo_tts.smoke")
    for name in ("--train-out", "--baseline-eval", "--baseline-arms", "--base", "--items", "--gpu-log", "--out"):
        ap.add_argument(name, required=True)
    ap.add_argument("--host-ram-peak-gb", type=float, required=True)
    ap.add_argument("--swap-peak-gb", type=float, required=True)
    ap.add_argument("--score-peak-gb", type=float)
    ap.add_argument("--full-steps", type=int, default=130)
    ap.add_argument("--overhead-s", type=float, default=300.0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print(f"error: {out} exists; a smoke record is never overwritten", file=sys.stderr)
        return 3
    rec = build(train_out=Path(a.train_out), baseline_eval=Path(a.baseline_eval), baseline_arms=Path(a.baseline_arms),
                base=Path(a.base), items=Path(a.items), gpu_log=Path(a.gpu_log), host_ram_peak_gb=a.host_ram_peak_gb,
                swap_peak_gb=a.swap_peak_gb, full_steps=a.full_steps, overhead_s=a.overhead_s,
                score_peak_gb=a.score_peak_gb)
    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"smoke record {out}: {rec['verdict']} {rec['reasons']}")
    return 0 if rec["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
