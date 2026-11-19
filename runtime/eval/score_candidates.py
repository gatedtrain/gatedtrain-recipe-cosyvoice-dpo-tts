"""Reward for DPO candidates: Whisper-large-v3-turbo WER of every sampled candidate (evaluator environment).

The reward ASR is Whisper only; Qwen3-ASR (the deciding metric of score_tts.py) is never loaded here, so nothing it
outputs can reach pair selection or training. WER is computed exactly as score_tts.py computes `wer_whisper_canon`:
Whisper's English text normalizer plus the `canon_version` canonicalization (wer_canon.py), against the item's
`ref_text` when present, else `text`. Transcription is batched (`--batch-size`, the transformers ASR pipeline).

Reads <candidates>/candidates.jsonl and candidates_meta.json, and the item set the candidates were sampled from
(must be a train text split; its hash must match the candidates' meta). Writes <out>/rewards.jsonl (item, k, seed,
asr_whisper, wer_whisper_canon, wer_whisper (normalizer only, against `text`), ref_source, error) and
<out>/reward_meta.json.

usage: score_candidates.py --candidates CAND_DIR --items ITEMS_DIR --out OUT [--batch-size 16] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AREA = HERE.parent
MODELS = AREA / "models"
sys.path.insert(0, str(HERE))

import wer_canon  # noqa: E402

REWARD_ASR = "whisper-large-v3-turbo"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def read_inputs(cand: Path, items_dir: Path) -> tuple[list[dict], dict, dict, dict]:
    cmeta = json.loads((cand / "candidates_meta.json").read_text())
    body = (items_dir / "items.jsonl").read_text(encoding="utf-8")
    imeta = json.loads((items_dir / "item_set_meta.json").read_text())
    items_sha = "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
    if (imeta.get("text_split"), imeta.get("voice_split")) != ("train", "train_clean_100"):
        raise SystemExit("rewards are computed for candidates of the train set only (not dev or test)")
    if items_sha != cmeta.get("items_sha256"):
        raise SystemExit("the candidates were sampled from another item set")
    items = {json.loads(l)["item"]: json.loads(l) for l in body.splitlines() if l}
    recs = [json.loads(l) for l in (cand / "candidates.jsonl").read_text().splitlines() if l]
    return recs, items, cmeta, {"items_sha256": items_sha}


def score_records(recs: list[dict], items: dict, cand: Path, transcribe, norm, batch_size: int, log=print):
    """Rewards in record order. `transcribe(list of wav paths) -> list of str` (batched)."""
    rows = [{"item": r["item"], "k": r["k"], "seed": r["seed"], "error": r.get("error")} for r in recs]
    todo = [i for i, r in enumerate(recs) if not r.get("error")]
    for row in rows:
        if row["error"]:
            row["error"] = f"sampling error: {row['error']}"[:300]
    for s in range(0, len(todo), batch_size):
        idx = todo[s: s + batch_size]
        try:
            texts = transcribe([str(cand / recs[i]["wav"]) for i in idx])
        except Exception:                            # e.g. long-form and short-form audio in one batch:
            texts = []                               # retry one by one; a failure is an error record
            for i in idx:
                try:
                    texts.extend(transcribe([str(cand / recs[i]["wav"])]))
                except Exception as exc:
                    rows[i]["error"] = f"{exc.__class__.__name__}: {exc}"[:300]
                    texts.append(None)
        for i, hyp in zip(idx, texts):
            if hyp is None:
                continue
            it = items[recs[i]["item"]]
            rows[i].update(asr_whisper=hyp, wer_whisper_canon=wer_canon.wer_canon(wer_canon.reference(it), hyp, norm),
                           wer_whisper=wer_canon.wer_plain(it["text"], hyp, norm),
                           ref_source="ref_text" if it.get("ref_text") else "text")
        log(json.dumps({"done": min(s + batch_size, len(todo)), "of": len(todo)}))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cand, out = Path(a.candidates), Path(a.out)
    recs, items, cmeta, ih = read_inputs(cand, Path(a.items))
    recs = recs[: a.limit]
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import pipeline
    t0 = time.time()
    norm = wer_canon.load_normalizer(MODELS)
    whisper = pipeline("automatic-speech-recognition", model=str(MODELS / REWARD_ASR), torch_dtype=torch.float16,
                       device="cuda:0")

    def transcribe(paths):
        res = whisper(paths, batch_size=len(paths), generate_kwargs={"language": "english", "task": "transcribe"})
        return [r["text"] for r in res]

    rows = score_records(recs, items, cand, transcribe, norm, a.batch_size)
    (out / "rewards.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    import importlib.metadata as md
    ok = [r for r in rows if r["error"] is None]
    meta = {"reward_asr": REWARD_ASR,
            "reward_asr_revision": json.loads((MODELS / "MODELS.json").read_text())["models"][REWARD_ASR]["revision"],
            "canon_version": wer_canon.CANON_VERSION, "wer_canon_sha256": sha256(HERE / "wer_canon.py"),
            "script_sha256": sha256(Path(__file__)), "candidates": str(cand.resolve()),
            "candidates_sha256": sha256(cand / "candidates.jsonl"), "candidates_meta_sha256":
            sha256(cand / "candidates_meta.json"), "items": str(Path(a.items).resolve()), **ih,
            "n": len(rows), "errors": len(rows) - len(ok), "limit": a.limit, "batch_size": a.batch_size,
            "mean_wer_whisper_canon": round(sum(r["wer_whisper_canon"] for r in ok) / len(ok), 4) if ok else None,
            "rewards_sha256": sha256(out / "rewards.jsonl"), "wall_seconds": round(time.time() - t0, 1),
            "python": platform.python_version(),
            "versions": {p: md.version(p) for p in ("torch", "transformers", "jiwer")}}
    (out / "reward_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
