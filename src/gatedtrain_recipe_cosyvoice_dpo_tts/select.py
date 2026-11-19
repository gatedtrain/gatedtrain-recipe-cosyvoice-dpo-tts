"""Checkpoint selection on the dev set, pre-registered (host side, CPU only).

The rule of the published run (runtime/run_dev_select.sh, RESULT.md "How the gate was fixed" step 2), written out:

1. Every cut of the training run (train_dpo.py `checkpoint-<N>/`, N = 26, 52, 78, 104, 130 for the full profile) and
   the base generate `items-hard-dev` (150 items: training templates, dev_clean voices) with seed 11 and are scored by
   score_tts.py (eval_arm.py); the base is reported, never selected.
2. The statistic is the cut's mean `wer_qwen_canon` over the dev items as score_tts.py's summary.json gives it (error
   rows skipped). A cut whose dev arm has an error row is eligible only if every cut has one; then the fewest error rows
   come first.
3. The selected cut has the lowest statistic; ties (equal floats) go to the later step.
4. gatedtrain gates the last cut of a run. If the selected cut is the run's last, the run is collected as it is. Otherwise
   a new full training launch with `--max-steps N` (same settings, seed and inputs; the LR schedule is warm-up then
   constant, so its steps are the first N of the same schedule, not bit-identical under bf16) gives the candidate: its
   last cut, evaluated on the test set without another dev selection.
Nothing of the test set is read.

usage: python -m gatedtrain_recipe_cosyvoice_dpo_tts.select --run TRAIN_OUT --dev STEP=EVAL_DIR [--dev STEP=EVAL_DIR ...]
           [--base EVAL_DIR] --out selection.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .arms import sha256

RULE = ("lowest mean dev wer_qwen_canon (score_tts.py summary.json, error rows skipped) among the run's cuts; a cut "
        "with an error row only if all have one (fewest first); ties to the later step; the base is reported only; a "
        "selected cut that is not the run's last is retrained with --max-steps N and that run's last cut is the "
        "candidate")


def select(run: Path, dev: dict[int, Path], base: Path | None = None) -> dict:
    """The selection record; raises ValueError when the dev arms do not cover the run's cuts."""
    cuts = sorted(int(p.name.split("-")[1]) for p in Path(run).glob("checkpoint-*") if (p / "cut.json").is_file())
    if sorted(dev) != cuts:
        raise ValueError(f"dev arms for steps {sorted(dev)}, but the run's cuts are {cuts}")
    stats = {}
    for step, d in dev.items():
        summary = json.loads((Path(d) / "score" / "summary.json").read_text(encoding="utf-8"))
        run_rec = json.loads((Path(d) / "eval_run.json").read_text(encoding="utf-8"))
        if run_rec.get("item_set") != "items-hard-dev" or run_rec.get("limit") is not None:
            raise ValueError(f"{d} is not a full items-hard-dev arm")
        cut_json = Path(run) / f"checkpoint-{step}" / "cut.json"
        if (run_rec.get("cut") or {}).get("cut_json_sha256") != sha256(cut_json):
            raise ValueError(f"{d} did not evaluate {cut_json}")
        stats[step] = {"wer_qwen_canon": summary["wer_qwen_canon"], "errors": summary["errors"], "n": summary["n"],
                       "sim_prompt": summary.get("sim_prompt")}
    min_err = min(s["errors"] for s in stats.values())
    eligible = [k for k, s in stats.items() if s["errors"] == min_err]
    best = min(eligible, key=lambda k: (stats[k]["wer_qwen_canon"], -k))
    rec = {"schema": "gatedtrain_recipe_cosyvoice_dpo_tts.selection", "rule": RULE, "run": str(run), "cuts": cuts,
           "dev": {str(k): v for k, v in sorted(stats.items())}, "selected_step": best, "last_step": cuts[-1],
           "action": ("collect the run as it is" if best == cuts[-1]
                      else f"train again with --max-steps {best}; its last cut is the candidate")}
    if base is not None:
        summary = json.loads((Path(base) / "score" / "summary.json").read_text(encoding="utf-8"))
        rec["base"] = {"wer_qwen_canon": summary["wer_qwen_canon"], "errors": summary["errors"],
                       "sim_prompt": summary.get("sim_prompt")}
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gatedtrain_recipe_cosyvoice_dpo_tts.select")
    ap.add_argument("--run", required=True)
    ap.add_argument("--dev", action="append", required=True, help="STEP=EVAL_DIR (an eval_arm.py output)")
    ap.add_argument("--base")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = {int(k): Path(v) for k, v in (x.split("=", 1) for x in a.dev)}
    out = Path(a.out)
    if out.exists():
        print(f"error: {out} exists", file=sys.stderr)
        return 3
    rec = select(Path(a.run), dev, Path(a.base) if a.base else None)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"selected step {rec['selected_step']} of {rec['cuts']}: {rec['action']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
