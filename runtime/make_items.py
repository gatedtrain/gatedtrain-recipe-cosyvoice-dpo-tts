"""Build a zero-shot TTS item set from a LibriTTS-R split (deterministic).

An item is (target text, reference recording of the same speaker and its transcript): the model speaks the target
text in the reference voice. The reference is another utterance than the target, so the target recording never
enters the prompt; the speaker's real recording of the target text is kept as `target_wav` (similarity and quality
references only, never shown to the model).

Per speaker: the reference is the speaker's utterance with the most words among those of 4-10 s; targets are drawn
(seeded, without replacement) from the speaker's other utterances with 6-40 words. Paths are relative to this
directory's `data-source/`.

usage: make_items.py --split test_clean --per-speaker 13 --seed 11 --out items-test [--limit-speakers N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
SRC = HERE / "data-source"


def utterances(split: str) -> dict[str, list[dict]]:
    root = SRC / f"unpacked-{split}" / "LibriTTS_R" / split.replace("_", "-")
    out: dict[str, list[dict]] = {}
    for wav in sorted(root.glob("*/*/*.wav")):
        txt = wav.with_name(wav.name[:-4] + ".normalized.txt")
        if not txt.exists():
            continue
        text = txt.read_text(encoding="utf-8").strip()
        info = sf.info(str(wav))
        out.setdefault(wav.parts[-3], []).append({
            "id": wav.stem, "wav": str(wav.relative_to(SRC)), "text": text, "words": len(text.split()),
            "seconds": round(info.frames / info.samplerate, 3)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["test_clean", "dev_clean"])
    ap.add_argument("--per-speaker", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-speakers", type=int, default=None)
    a = ap.parse_args()
    out = HERE / a.out
    if out.exists():
        raise SystemExit(f"{out} exists: an item set is never overwritten")
    speakers = utterances(a.split)
    names = sorted(speakers, key=int)[: a.limit_speakers]
    rows = []
    for spk in names:
        utts = speakers[spk]
        refs = [u for u in utts if 4.0 <= u["seconds"] <= 10.0]
        if not refs:
            continue
        ref = max(refs, key=lambda u: (u["words"], u["id"]))
        pool = [u for u in utts if u["id"] != ref["id"] and 6 <= u["words"] <= 40 and u["text"] != ref["text"]]
        rng = random.Random(f"{a.seed}:{spk}")
        for tgt in rng.sample(pool, min(a.per_speaker, len(pool))):
            rows.append({"item": f"{a.split}:{tgt['id']}", "speaker": spk, "text": tgt["text"],
                         "prompt_text": ref["text"], "prompt_wav": ref["wav"], "prompt_id": ref["id"],
                         "target_wav": tgt["wav"], "target_seconds": tgt["seconds"]})
    out.mkdir(parents=True)
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (out / "items.jsonl").write_text(body, encoding="utf-8")
    meta = {"split": a.split, "seed": a.seed, "per_speaker": a.per_speaker, "limit_speakers": a.limit_speakers,
            "n_items": len(rows), "n_speakers": len({r["speaker"] for r in rows}),
            "items_sha256": "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "source": "LibriTTS-R (OpenSLR 141, CC BY 4.0)"}
    (out / "item_set_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
