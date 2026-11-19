"""Apply gate/registration.yaml to the scored test-set arms (paired, cluster bootstrap by speaker).

For each clause the per-item difference d = candidate - baseline is formed on every registered item (an item missing
or errored in either arm fails the gate: max_unpaired 0). The mean difference gets a one-sided (1 - alpha) bound from
a cluster bootstrap that resamples speakers with replacement (all items of a drawn speaker, as many speakers as there
are). Superiority, lower is better: PASS if the upper bound < -margin. Non-inferiority, higher is better: PASS if the
lower bound > -margin. The gate passes only if every blocking clause passes. Checks every pinned hash first; any
mismatch stops the analysis. The rl arm, if given, is reported descriptively against the baseline.

usage: analyze_gate.py --registration gate/registration.yaml --baseline OUT/test-base-score
                       --candidate OUT/test-dpo-train-k6-final-score [--reference OUT/test-rl-score] --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent


def sha256(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def load_arm(score_dir: Path) -> tuple[dict, dict, dict]:
    rows = {}
    for line in (score_dir / "scores.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["item"]] = r
    meta = json.loads((score_dir / "score_meta.json").read_text())
    gen_meta = json.loads((Path(meta["gen"]) / "gen_meta.json").read_text()) if meta.get("gen") else {}
    return rows, meta, gen_meta


def check_pins(reg: dict, arm: str, meta: dict, gen_meta: dict, problems: list[str]) -> None:
    want = reg["arms"][arm]["llm_sha256"] if arm in reg["arms"] else None
    if want and gen_meta.get("llm_sha256") != want:
        problems.append(f"{arm}: llm_sha256 {gen_meta.get('llm_sha256')} != registered {want}")
    if gen_meta.get("seed") not in reg["eval_seeds"]:
        problems.append(f"{arm}: seed {gen_meta.get('seed')} not in {reg['eval_seeds']}")
    if gen_meta.get("items_sha256") != reg["eval_distribution"]["items_sha256"]:
        problems.append(f"{arm}: generated items {gen_meta.get('items_sha256')} != registered")
    if gen_meta.get("text_frontend") is not reg["code"]["text_frontend"]:
        problems.append(f"{arm}: text_frontend {gen_meta.get('text_frontend')}")
    if meta.get("script_sha256") != reg["code"]["score_tts_sha256"]:
        problems.append(f"{arm}: score_tts {meta.get('script_sha256')} != registered")
    if meta.get("canon_version") != reg["code"]["canon_version"]:
        problems.append(f"{arm}: canon {meta.get('canon_version')}")


def cluster_bootstrap_means(d: np.ndarray, clusters: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    keys = np.unique(clusters)
    sums = np.array([d[clusters == k].sum() for k in keys])
    counts = np.array([(clusters == k).sum() for k in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(replicates, len(keys)))
    return sums[draws].sum(axis=1) / counts[draws].sum(axis=1)


def judge(clause: dict, d: np.ndarray, boot: np.ndarray) -> dict:
    alpha, margin = clause["alpha"], clause["margin"]
    mean = float(d.mean())
    if clause["direction"] == "lower_is_better":     # improvement = negative d
        bound = float(np.quantile(boot, 1 - alpha))
        ok = bound < -margin if clause["type"] == "superiority" else bound < margin
        side = "upper"
    else:                                             # improvement = positive d
        bound = float(np.quantile(boot, alpha))
        ok = bound > margin if clause["type"] == "superiority" else bound > -margin
        side = "lower"
    return {"name": clause["name"], "type": clause["type"], "metric": clause["metric"], "direction": clause["direction"],
            "margin": margin, "alpha": alpha, "n": int(len(d)), "mean_diff": round(mean, 5),
            f"{side}_bound": round(bound, 5), "verdict": "PASS" if ok else "FAIL", "blocking": clause["blocking"]}


def analyze(reg: dict, base: dict, cand: dict, problems: list[str]) -> dict:
    items = sorted(base)
    if sorted(cand) != items:
        problems.append("baseline and candidate cover different items")
    if len(items) != reg["eval_distribution"]["n_items"]:
        problems.append(f"{len(items)} items, registered {reg['eval_distribution']['n_items']}")
    bad = [i for i in items if base[i].get("error") or (i in cand and cand[i].get("error"))]
    if len(bad) > reg["pairing"]["max_unpaired"]:
        problems.append(f"{len(bad)} items with an error in an arm (max_unpaired {reg['pairing']['max_unpaired']})")
    if problems:
        return {"verdict": "FAIL", "problems": problems}
    clusters = np.array([base[i][reg["cluster_key"]] for i in items])
    if any(cand[i][reg["cluster_key"]] != base[i][reg["cluster_key"]] for i in items):
        return {"verdict": "FAIL", "problems": ["arms differ in the cluster key of an item"]}
    out = []
    for c in reg["clauses"]:
        d = np.array([cand[i][c["metric"]] - base[i][c["metric"]] for i in items], dtype=float)
        boot = cluster_bootstrap_means(d, clusters, reg["bootstrap"]["replicates"], reg["bootstrap"]["seed"])
        out.append(judge(c, d, boot))
    verdict = "PASS" if all(r["verdict"] == "PASS" for r in out if r["blocking"]) else "FAIL"
    return {"verdict": verdict, "clauses": out, "n_items": len(items), "n_clusters": int(len(np.unique(clusters)))}


def descriptive(base: dict, arm: dict, metrics: list[str]) -> dict:
    items = sorted(i for i in set(base) & set(arm) if not base[i].get("error") and not arm[i].get("error"))
    res = {}
    for m in metrics:
        vals = [(base[i][m], arm[i][m]) for i in items if base[i].get(m) is not None and arm[i].get(m) is not None]
        res[m] = {"baseline_mean": round(float(np.mean([v[0] for v in vals])), 5),
                  "arm_mean": round(float(np.mean([v[1] for v in vals])), 5), "n": len(vals)}
    cats = sorted({i.split(":")[1] for i in items})
    res["per_category_wer_qwen_canon"] = {
        c: {"baseline": round(float(np.mean([base[i]["wer_qwen_canon"] for i in items if i.split(":")[1] == c])), 4),
            "arm": round(float(np.mean([arm[i]["wer_qwen_canon"] for i in items if i.split(":")[1] == c])), 4)}
        for c in cats}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reg_path, out = Path(a.registration), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    reg = yaml.safe_load(reg_path.read_text())
    problems: list[str] = []
    base, bmeta, bgen = load_arm(Path(a.baseline))
    cand, cmeta, cgen = load_arm(Path(a.candidate))
    check_pins(reg, "baseline", bmeta, bgen, problems)
    check_pins(reg, "candidate", cmeta, cgen, problems)
    result = analyze(reg, base, cand, problems)
    metrics = ["wer_qwen_canon", "wer_whisper_canon", "wer_qwen", "sim_prompt"]
    result["descriptive"] = {"candidate": descriptive(base, cand, metrics)}
    if a.reference:
        ref, rmeta, rgen = load_arm(Path(a.reference))
        rp: list[str] = []
        check_pins(reg, "reference_descriptive", rmeta, rgen, rp)
        result["descriptive"]["reference_rl"] = {**descriptive(base, ref, metrics), "pin_problems": rp}
    result.update({"registration": str(reg_path.resolve()), "registration_sha256": sha256(reg_path),
                   "registration_id": reg["registration_id"], "script_sha256": sha256(Path(__file__)),
                   "baseline": str(Path(a.baseline).resolve()), "candidate": str(Path(a.candidate).resolve()),
                   "baseline_scores_sha256": sha256(Path(a.baseline) / "scores.jsonl"),
                   "candidate_scores_sha256": sha256(Path(a.candidate) / "scores.jsonl")})
    (out / "gate_result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
