"""DPO preference pairs from WER-scored candidates (stdlib only; CPU).

Per item: among the candidates that pass the sanity checks, chosen = lowest reward WER (`wer_whisper_canon` from
eval/score_candidates.py, Whisper-large-v3-turbo), rejected = highest; ties go to the lower candidate index k (the
seed order). The pair is kept when WER(rejected) - WER(chosen) >= `--margin` (default 0.05). The deciding ASR
(Qwen3-ASR) is never read: rewards come only from a reward_meta.json that names Whisper as the reward ASR.

Sanity checks (a candidate failing any is neither chosen nor rejected): no sampling or scoring error; at least one
speech token and some audio; sampling ended on a stop token (not at max_len); a non-empty transcript and a finite
WER; audio duration within [`--min-ratio`, `--max-ratio`] (default 0.5, 2.5) x the expected duration
len(text) / cps, where cps is the median characters-per-second of the category's candidates (all candidates of the
set that pass the other checks; a category with fewer than 20 such candidates uses the median over all categories).
Per-category rates are needed because numerals are short in characters but long when spoken (pilot: ~7.5 chars/s
for numbers, ~17 for natural sentences).

Reads <candidates>/candidates.jsonl + candidates_meta.json, <rewards>/rewards.jsonl + reward_meta.json, and the
train item set (categories and texts; its hash must match both metas; any other set, dev included, is refused,
B18).
Writes <out>/pairs.jsonl and <out>/pairs_meta.json (inputs' hashes, arguments, pair-yield statistics).

usage: make_pairs.py --candidates CAND_DIR --rewards REWARD_DIR --items ITEMS_DIR --out OUT [--margin 0.05]
                     [--min-ratio 0.5] [--max-ratio 2.5]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

MIN_CATEGORY_CANDIDATES = 20


def sha256(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def basic_failure(c: dict) -> str | None:
    """The first sanity check (other than duration) a candidate fails, or None."""
    if c.get("error"):
        return "sampling_error"
    if c.get("reward_error"):
        return "reward_error"
    if not c.get("n_tokens") or not c.get("audio_seconds"):
        return "empty"
    if c.get("stop") is None:
        return "max_len"
    if not (c.get("asr_whisper") or "").strip():
        return "empty_transcript"
    w = c.get("wer_whisper_canon")
    if not isinstance(w, (int, float)) or not math.isfinite(w):
        return "no_wer"
    return None


def chars_per_second(cands: list[dict], items: dict) -> tuple[dict, float]:
    rates: dict[str, list[float]] = defaultdict(list)
    for c in cands:
        if basic_failure(c) is None:
            rates[items[c["item"]]["category"]].append(len(items[c["item"]]["text"]) / c["audio_seconds"])
    every = [r for rs in rates.values() for r in rs]
    if not every:
        raise SystemExit("no candidate passes the basic sanity checks")
    overall = statistics.median(every)
    return {cat: (statistics.median(rs) if len(rs) >= MIN_CATEGORY_CANDIDATES else overall)
            for cat, rs in rates.items()}, overall


def make_pairs(cands: list[dict], items: dict, margin: float, min_ratio: float, max_ratio: float):
    cps, overall = chars_per_second(cands, items)
    reasons: Counter = Counter()
    by_item: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        it = items[c["item"]]
        why = basic_failure(c)
        if why is None:
            expected = len(it["text"]) / cps.get(it["category"], overall)
            if not (min_ratio * expected <= c["audio_seconds"] <= max_ratio * expected):
                why = "duration"
        reasons[why or "ok"] += 1
        if why is None:
            by_item[c["item"]].append(c)
    pairs, drops = [], Counter()
    items_seen = sorted({c["item"] for c in cands})
    for item in items_seen:
        sane = by_item.get(item, [])
        if len(sane) < 2:
            drops["fewer_than_2_sane"] += 1
            continue
        chosen = min(sane, key=lambda c: (c["wer_whisper_canon"], c["k"]))
        rejected = min(sane, key=lambda c: (-c["wer_whisper_canon"], c["k"]))
        gap = rejected["wer_whisper_canon"] - chosen["wer_whisper_canon"]
        if gap < margin or chosen["k"] == rejected["k"]:
            drops["all_zero" if rejected["wer_whisper_canon"] == 0 else "below_margin"] += 1
            continue
        pairs.append({"item": item, "category": items[item]["category"], "chosen_k": chosen["k"],
                      "rejected_k": rejected["k"], "chosen_seed": chosen["seed"], "rejected_seed": rejected["seed"],
                      "wer_chosen": chosen["wer_whisper_canon"], "wer_rejected": rejected["wer_whisper_canon"],
                      "margin": round(gap, 6), "n_sane": len(sane), "chosen_tokens": chosen["n_tokens"],
                      "rejected_tokens": rejected["n_tokens"]})
    cats = sorted({it["category"] for it in (items[i] for i in items_seen)})
    per_cat = {c: {"items": sum(items[i]["category"] == c for i in items_seen),
                   "pairs": sum(p["category"] == c for p in pairs)} for c in cats}
    gaps = sorted(p["margin"] for p in pairs)
    q = lambda f: round(gaps[min(len(gaps) - 1, int(f * len(gaps)))], 4) if gaps else None  # noqa: E731
    stats = {"items": len(items_seen), "candidates": len(cands), "pairs": len(pairs),
             "pair_yield": round(len(pairs) / len(items_seen), 4) if items_seen else None,
             "candidate_checks": dict(sorted(reasons.items())), "item_drops": dict(sorted(drops.items())),
             "per_category": per_cat, "chars_per_second": {k: round(v, 3) for k, v in sorted(cps.items())},
             "chars_per_second_overall": round(overall, 3),
             "mean_wer_chosen": round(statistics.fmean(p["wer_chosen"] for p in pairs), 4) if pairs else None,
             "mean_wer_rejected": round(statistics.fmean(p["wer_rejected"] for p in pairs), 4) if pairs else None,
             "margin_quantiles": {"p10": q(0.1), "p50": q(0.5), "p90": q(0.9)}}
    return pairs, stats


def load(cand_dir: Path, reward_dir: Path, items_dir: Path) -> tuple[list[dict], dict, dict]:
    cmeta = json.loads((cand_dir / "candidates_meta.json").read_text())
    rmeta = json.loads((reward_dir / "reward_meta.json").read_text())
    imeta = json.loads((items_dir / "item_set_meta.json").read_text())
    body = (items_dir / "items.jsonl").read_text(encoding="utf-8")
    items_sha = "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
    if any((m.get("text_split"), m.get("voice_split")) != ("train", "train_clean_100") for m in (imeta, cmeta)):
        raise SystemExit("pairs are made from the train set only (text split train, voices train_clean_100)")
    if not (items_sha == cmeta.get("items_sha256") == rmeta.get("items_sha256")):
        raise SystemExit("item set, candidates and rewards do not belong together")
    if rmeta.get("reward_asr") != "whisper-large-v3-turbo":
        raise SystemExit(f"reward ASR is {rmeta.get('reward_asr')!r}; pairs use Whisper rewards only")
    if rmeta.get("candidates_sha256") != sha256(cand_dir / "candidates.jsonl"):
        raise SystemExit("rewards were computed for another candidates.jsonl")
    items = {json.loads(l)["item"]: json.loads(l) for l in body.splitlines() if l}
    rewards = {(r["item"], r["k"]): r for r in
               (json.loads(l) for l in (reward_dir / "rewards.jsonl").read_text().splitlines() if l)}
    cands = []
    for l in (cand_dir / "candidates.jsonl").read_text().splitlines():
        if not l:
            continue
        c = json.loads(l)
        r = rewards.get((c["item"], c["k"]))
        if r is None:
            continue                                  # a --limit reward run: unscored candidates are skipped
        if r["seed"] != c["seed"]:
            raise SystemExit(f"seed mismatch for {c['item']} k={c['k']}")
        cands.append({**c, "reward_error": r.get("error") if not c.get("error") else None,
                      "asr_whisper": r.get("asr_whisper"), "wer_whisper_canon": r.get("wer_whisper_canon")})
    return cands, items, {"candidates_meta": cmeta, "reward_meta": rmeta, "items_sha256": items_sha}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--rewards", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--min-ratio", type=float, default=0.5)
    ap.add_argument("--max-ratio", type=float, default=2.5)
    a = ap.parse_args()
    cand_dir, reward_dir, out = Path(a.candidates), Path(a.rewards), Path(a.out)
    cands, items, metas = load(cand_dir, reward_dir, Path(a.items))
    pairs, stats = make_pairs(cands, items, a.margin, a.min_ratio, a.max_ratio)
    out.mkdir(parents=True, exist_ok=False)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in pairs))
    meta = {"script_sha256": sha256(Path(__file__)), "candidates": str(cand_dir.resolve()),
            "candidates_sha256": sha256(cand_dir / "candidates.jsonl"), "rewards": str(reward_dir.resolve()),
            "rewards_sha256": sha256(reward_dir / "rewards.jsonl"), "items": str(Path(a.items).resolve()),
            "items_sha256": metas["items_sha256"], "reward_asr": metas["reward_meta"]["reward_asr"],
            "canon_version": metas["reward_meta"].get("canon_version"), "k": metas["candidates_meta"].get("k"),
            "sampling_seed": metas["candidates_meta"].get("seed"), "margin": a.margin, "min_ratio": a.min_ratio,
            "max_ratio": a.max_ratio, "pairs_sha256": sha256(out / "pairs.jsonl"), "stats": stats}
    (out / "pairs_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
