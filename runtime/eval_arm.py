"""One evaluation arm in one process: generate an item set with gen_tts.py, then score it with eval/score_tts.py.

The launch unit of an arm under `gatedtrain ext-launch` (one interpreter per launch: this runtime's .venv). gen_tts.py runs
with this interpreter; score_tts.py runs with the evaluator environment's interpreter (eval/.venv/bin/python), both from
this runtime directory, so a read-only runtime mount carries everything. Nothing here changes what the two scripts do;
the arguments are those of the published run (`--seed 11`, `--no-real`).

--llm base | rl | CUT_DIR: the base's llm.pt, the vendor's llm.rl.pt (reference arm, descriptive only), or a cut of
train_dpo.py (`checkpoint-<N>/`, mounted read-only at /inputs/cut under ext-launch): its cut.json is checked (the
SHA-256 of llm.pt and base_ref.json, deployable ["llm.pt"]) before its llm.pt is loaded.
--item-set NAME: an item set directory of this runtime (items-hard-test, items-hard-dev).

Writes <out>/gen/ (gen_tts.py), <out>/score/ (score_tts.py) and <out>/eval_run.json: the scripts' SHA-256, the LLM
weights' SHA-256, the cut's cut.json SHA-256 and step, the item set and its items.jsonl SHA-256, seed and limit.
`gatedtrain_recipe_cosyvoice_dpo_tts.arms` (host side) turns <out> into per-metric gatedtrain arms.

The speaker model's links (eval/ecapa-cache/) must exist before a read-only launch: score_tts.py would create them in
the runtime directory. They are checked first, so a missing cache stops the arm before any GPU work.

usage: eval_arm.py --llm base|rl|CUT_DIR --item-set items-hard-test --out OUT [--seed 11] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models" / "fun-cosyvoice3-0.5b-2512"
EVAL_PYTHON = HERE / "eval" / ".venv" / "bin" / "python"
ITEM_SETS = ("items-hard-test", "items-hard-dev")
ECAPA_FILES = ("classifier.ckpt", "embedding_model.ckpt", "hyperparams.yaml", "label_encoder.ckpt",
               "mean_var_norm_emb.ckpt")
CUT_FILES = ("llm.pt", "base_ref.json")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def check_cut(cut: Path) -> dict:
    """cut.json of a train_dpo.py cut: the files it lists are exactly llm.pt and base_ref.json with their SHA-256."""
    doc = json.loads((cut / "cut.json").read_text(encoding="utf-8"))
    present = sorted(p.name for p in cut.iterdir())
    if present != sorted([*CUT_FILES, "cut.json"]) or sorted(doc.get("files", {})) != sorted(CUT_FILES):
        raise SystemExit(f"{cut} holds {present}, cut.json lists {sorted(doc.get('files', {}))}: not a complete cut")
    for n in CUT_FILES:
        if sha256(cut / n) != doc["files"][n]:
            raise SystemExit(f"{cut}/{n} is not the SHA-256 cut.json records")
    if doc.get("deployable") != ["llm.pt"]:
        raise SystemExit(f"{cut}/cut.json deployable is {doc.get('deployable')!r}")
    return {"cut_json_sha256": sha256(cut / "cut.json"), "global_step": doc.get("global_step")}


def check_ecapa_cache() -> None:
    d = HERE / "eval" / "ecapa-cache"
    missing = [n for n in ECAPA_FILES if not (d / n).exists()]
    if missing:
        raise SystemExit(f"{d} lacks {missing}: create the speaker model's links before a read-only launch "
                         "(runtime/README.md, prepare_runtime.sh)")


def plan(a: argparse.Namespace) -> tuple[list[list[str]], dict]:
    """The two commands and the run record (without the results), from the arguments; no side effects."""
    if a.item_set not in ITEM_SETS:
        raise SystemExit(f"--item-set {a.item_set!r} is not one of {list(ITEM_SETS)}")
    items = HERE / a.item_set
    out = Path(a.out)
    rec: dict = {"llm": a.llm, "cut": None}
    if a.llm in ("base", "rl"):
        llm_arg, llm_file = a.llm, MODEL_DIR / ("llm.pt" if a.llm == "base" else "llm.rl.pt")
    else:
        cut = Path(a.llm)
        rec["cut"] = check_cut(cut)
        llm_arg = llm_file = cut / "llm.pt"
    gen = [sys.executable, str(HERE / "gen_tts.py"), "--model-dir", str(MODEL_DIR), "--items", str(items), "--out",
           str(out / "gen"), "--llm", str(llm_arg), "--seed", str(a.seed)] + (
        ["--limit", str(a.limit)] if a.limit is not None else [])
    score = [str(EVAL_PYTHON), str(HERE / "eval" / "score_tts.py"), "--gen", str(out / "gen"), "--items", str(items),
             "--out", str(out / "score"), "--no-real"]
    rec.update({"schema": "gatedtrain_recipe_cosyvoice_dpo_tts.eval_run", "schema_version": "1.0",
                "script_sha256": sha256(Path(__file__).resolve()), "gen_tts_sha256": sha256(HERE / "gen_tts.py"),
                "score_tts_sha256": sha256(HERE / "eval" / "score_tts.py"),
                "wer_canon_sha256": sha256(HERE / "eval" / "wer_canon.py"), "llm_file": str(llm_file),
                "item_set": a.item_set, "items_sha256": sha256(items / "items.jsonl"), "seed": a.seed,
                "limit": a.limit, "argv": {"gen": gen[1:], "score": score[1:]}})
    return [gen, score], rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True, help="base, rl, or a train_dpo.py cut directory")
    ap.add_argument("--item-set", required=True, help=f"one of {', '.join(ITEM_SETS)}")
    ap.add_argument("--out", required=True, help="a new or empty directory")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"--out {out} is not empty")
    check_ecapa_cache()
    cmds, rec = plan(a)
    rec["llm_sha256"] = sha256(Path(rec["llm_file"]))
    for name, cmd in zip(("gen", "score"), cmds):
        t = time.time()
        print(json.dumps({"start": name, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)
        subprocess.run(cmd, check=True)
        rec[f"{name}_seconds"] = round(time.time() - t, 1)
    gen_meta = json.loads((out / "gen" / "gen_meta.json").read_text(encoding="utf-8"))
    if gen_meta.get("llm_sha256") != rec["llm_sha256"]:
        raise SystemExit(f"gen_meta.json llm_sha256 {gen_meta.get('llm_sha256')} is not {rec['llm_sha256']}")
    (out / "eval_run.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
