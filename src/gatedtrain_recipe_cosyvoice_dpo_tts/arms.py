"""score_tts.py output to gatedtrain per-item arms (MULTIMODAL-SPEC MM-1 to MM-3), one arm per metric.

Host side, CPU only, standard library plus this package: reads an evaluation's `gen/` (gen_tts.py: gen.jsonl,
gen_meta.json, wav/) and `score/` (score_tts.py: scores.jsonl, score_meta.json) and the item set, and writes
`<out>/<metric>/items.jsonl`, `eval_meta.json` and `outputs/<n>.json` (the item's text, reference, transcripts and the
audio's SHA-256) for each requested metric (`wer_qwen_canon`, `sim_prompt`). score_tts.py is not changed.

Checks before anything is written (exit 3 on a problem, nothing written):
- score_meta.json names the item set's items.jsonl SHA-256, the gen.jsonl it scored, and the pinned score_tts.py and
  wer_canon.py; gen_meta.json names the same item set and the pinned gen_tts.py;
- the evaluated weights: gen_meta.json llm_sha256 is the SHA-256 of `--checkpoint`'s llm.pt (the base directory, or a
  train_dpo.py cut whose cut.json lists that llm.pt);
- every scored item is an item of the set, once; the arm covers the item set's first n items in order (n = all,
  or gen_meta.json n_items for a --limit run).
A line is a score, or an error of class `gen_error` (gen_tts.py recorded an error) or `score_error` (score_tts.py
raised, or the value is not a finite number in the metric's range). Seeds: K = 1, `eval_seeds` [gen_meta.json seed].
`checkpoint`, `item_set_hash` and `sampler_hash` are left null for `gatedtrain ext-arm-fill`, which computes them from
the checkpoint and the item set (EXT-110).

usage: python -m gatedtrain_recipe_cosyvoice_dpo_tts.arms --gen GEN --score SCORE --items ITEMS --checkpoint DIR --out OUT
           [--metric wer_qwen_canon --metric sim_prompt]
       python -m gatedtrain_recipe_cosyvoice_dpo_tts.arms --eval-run EVAL_OUT ...   (GEN = EVAL_OUT/gen, SCORE = EVAL_OUT/score)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from . import pins
from .rows import EVALUATOR_ROWS, NAME, SAMPLER, SIM_ID, WER_ID

METRICS = {"wer_qwen_canon": WER_ID, "sim_prompt": SIM_ID}
# The evaluator's framework per metric (score_meta.json `versions`): the package that computes the score.
FRAMEWORK = {"wer_qwen_canon": "qwen-asr", "sim_prompt": "speechbrain"}
GEN_ERROR_PREFIX = "RuntimeError: generation error:"          # score_tts.py score_live for a gen.jsonl error


class ArmError(Exception):
    """Problems that stop the conversion; nothing has been written."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]


def checkpoint_llm(checkpoint: Path) -> tuple[Path, list[str]]:
    """The llm.pt `checkpoint` deploys: a cut's (listed in its cut.json) or the base directory's."""
    c = Path(checkpoint)
    if (c / "cut.json").is_file():
        doc = json.loads((c / "cut.json").read_text(encoding="utf-8"))
        files = doc.get("files") if isinstance(doc, dict) else None
        if not (isinstance(files, dict) and files.get("llm.pt") == sha256(c / "llm.pt")):
            return c / "llm.pt", [f"{c}/cut.json does not list its llm.pt with that file's SHA-256"]
    return c / "llm.pt", []


def build(*, gen: Path, score: Path, items: Path, checkpoint: Path,
          metrics: list[str]) -> dict[str, tuple[dict, list[dict], dict[str, dict]]]:
    """{metric: (eval_meta, item lines, {output path: output doc})}; raises ArmError."""
    gen, score, items = Path(gen), Path(score), Path(items)
    problems: list[str] = []
    bad_metrics = [m for m in metrics if m not in METRICS]
    if bad_metrics or not metrics:
        raise ArmError([f"metrics {bad_metrics or metrics} are not among {sorted(METRICS)}"])
    gm = json.loads((gen / "gen_meta.json").read_text(encoding="utf-8"))
    sm = json.loads((score / "score_meta.json").read_text(encoding="utf-8"))
    items_sha = sha256(items / "items.jsonl")
    item_rows = _jsonl(items / "items.jsonl")
    if sm.get("items_sha256") != items_sha:
        problems.append(f"score_meta.json items_sha256 {sm.get('items_sha256')} is not {items}/items.jsonl {items_sha}")
    if gm.get("items_sha256") != items_sha:
        problems.append(f"gen_meta.json items_sha256 {gm.get('items_sha256')} is not {items}/items.jsonl {items_sha}")
    if sm.get("gen_jsonl_sha256") != sha256(gen / "gen.jsonl"):
        problems.append("score_meta.json gen_jsonl_sha256 is not the SHA-256 of gen/gen.jsonl: it scored another "
                        "generation")
    for key, rel, meta in (("script_sha256", "eval/score_tts.py", sm), ("wer_canon_sha256", "eval/wer_canon.py", sm),
                           ("script_sha256", "gen_tts.py", gm)):
        if meta.get(key) != pins.SCRIPTS[rel]:
            problems.append(f"{rel} of the run is {meta.get(key)}, not the pinned {pins.SCRIPTS[rel]}")
    llm, why = checkpoint_llm(checkpoint)
    problems += why
    if not why and gm.get("llm_sha256") != sha256(llm):
        problems.append(f"gen_meta.json llm_sha256 {gm.get('llm_sha256')} is not {llm} ({sha256(llm)}): the arm "
                        "evaluated other weights than --checkpoint")
    if gm.get("text_frontend") is not False:
        problems.append("gen_meta.json text_frontend is not false (the registered sampler reads the raw text)")
    gen_rows = _jsonl(gen / "gen.jsonl")
    scores = _jsonl(score / "scores.jsonl")
    n = gm.get("n_items")
    want_items = [r["item"] for r in item_rows][: n if isinstance(n, int) else None]
    if [r.get("item") for r in scores] != want_items:
        problems.append(f"scores.jsonl covers {len(scores)} items, not the item set's first {len(want_items)} in order")
    if [r.get("item") for r in gen_rows] != want_items:
        problems.append(f"gen.jsonl covers {len(gen_rows)} items, not the item set's first {len(want_items)} in order")
    if problems:
        raise ArmError(problems)
    by_item = {r["item"]: r for r in item_rows}
    gen_by = {r["item"]: r for r in gen_rows}
    seed = gm.get("seed")
    out: dict[str, tuple[dict, list[dict], dict[str, dict]]] = {}
    for metric in metrics:
        eid = METRICS[metric]
        row = EVALUATOR_ROWS[eid]
        lo, hi = row["score_range"]
        lines: list[dict] = []
        outputs: dict[str, dict] = {}
        for k, s in enumerate(scores):
            it, g = by_item[s["item"]], gen_by[s["item"]]
            rel = f"outputs/{k:05d}.json"
            wav = gen / g["wav"] if g.get("wav") else None
            outputs[rel] = {"item": s["item"], "text": it["text"], "reference": it.get("ref_text") or it["text"],
                            "ref_source": s.get("ref_source"), "asr_qwen": s.get("asr_qwen"),
                            "asr_whisper": s.get("asr_whisper"), "audio_seconds": s.get("audio_seconds"),
                            "wav": g.get("wav"), "wav_sha256": sha256(wav) if wav is not None and wav.is_file() else None,
                            "gen_error": g.get("error"), "score_error": s.get("error"),
                            "wer_whisper_canon": s.get("wer_whisper_canon")}
            line: dict[str, Any] = {"item": s["item"], "repeat": 0, "seed": seed, "score": None, "error": None}
            v = s.get(metric)
            if s.get("error") is not None:
                cls = "gen_error" if str(s["error"]).startswith(GEN_ERROR_PREFIX) or g.get("error") else "score_error"
                line["error"] = {"class": cls, "message": str(s["error"])[:300]}
            elif not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                      and (lo is None or v >= lo) and (hi is None or v <= hi)):
                line["error"] = {"class": "score_error", "message": f"{metric} is {v!r}, not a finite number in "
                                                                    f"[{lo}, {hi}]"}
            else:
                line["score"] = float(v)
            line["output_ref"] = {"path": rel, "sha256": None}         # filled when the file is written
            line["item_meta"] = {"category": it.get("category"), "speaker": it.get("speaker")}
            lines.append(line)
            outputs[rel]["arm_metric"] = metric
        versions = sm.get("versions") or {}
        meta = {"schema": "gatedtrain.item_eval", "schema_version": "1.0", "evaluator_id": eid, "adapter": NAME,
                "metric": row["metric"], "direction": row["direction"], "score_range": list(row["score_range"]),
                "scorer": dict(row["scorer"]), "checkpoint": {"model_hash": None, "base_model_hash": None},
                "normalizer_sha256": None, "env_version_sha256": None, "item_set_hash": None, "eval_seeds": [seed],
                "sampler": SAMPLER, "sampler_hash": None,
                "framework": {"name": FRAMEWORK[metric], "version": str(versions.get(FRAMEWORK[metric]))}}
        out[metric] = (meta, lines, outputs)
    return out


def write(out_dir: Path, arms: dict[str, tuple[dict, list[dict], dict[str, dict]]]) -> list[Path]:
    """Write each arm into a new directory `<out_dir>/<metric>`; refuses an existing one."""
    out_dir = Path(out_dir)
    existing = [m for m in arms if (out_dir / m).exists()]
    if existing:
        raise ArmError([f"{out_dir / m} exists; an arm is written into a new directory" for m in existing])
    done = []
    for metric, (meta, lines, outputs) in arms.items():
        d = out_dir / metric
        (d / "outputs").mkdir(parents=True)
        for rel, doc in outputs.items():
            (d / rel).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        for line in lines:
            line["output_ref"]["sha256"] = sha256(d / line["output_ref"]["path"])
        (d / "items.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in lines), encoding="utf-8")
        (d / "eval_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        done.append(d)
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gatedtrain_recipe_cosyvoice_dpo_tts.arms")
    ap.add_argument("--eval-run", help="an eval_arm.py output directory (gen/ and score/)")
    ap.add_argument("--gen")
    ap.add_argument("--score")
    ap.add_argument("--items", required=True, help="the item set directory the arm evaluated")
    ap.add_argument("--checkpoint", required=True, help="the base directory or the evaluated cut")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", action="append", choices=sorted(METRICS))
    a = ap.parse_args(argv)
    gen = Path(a.gen) if a.gen else (Path(a.eval_run) / "gen" if a.eval_run else None)
    score = Path(a.score) if a.score else (Path(a.eval_run) / "score" if a.eval_run else None)
    if gen is None or score is None:
        ap.error("give --eval-run, or --gen and --score")
    try:
        arms = build(gen=gen, score=score, items=Path(a.items), checkpoint=Path(a.checkpoint),
                     metrics=a.metric or sorted(METRICS))
        for d in write(Path(a.out), arms):
            print(f"arm: {d}")
    except ArmError as exc:
        for p in exc.problems:
            print(f"error: {p}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
