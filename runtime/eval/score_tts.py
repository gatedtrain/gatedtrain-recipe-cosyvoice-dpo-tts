"""Score a synthesized item set: per-item WER by two ASRs and speaker similarity (evaluator environment).

Per item (records in <out>/scores.jsonl, summary in <out>/summary.json, provenance in <out>/score_meta.json):
- `wer_qwen_canon`: THE DECIDING METRIC. Word error rate of Qwen3-ASR-1.7B's transcript of the synthesized audio
  against the item's reference (`ref_text` when the item has one, else `text`; see wer_canon.reference) after
  Whisper's English text normalizer and the canonicalization `canon_version` (wer_canon.py: times to one HMM token,
  runs of single letters joined). An ASR that the training reward never sees.
- `wer_whisper_canon`: the same with Whisper-large-v3-turbo, the ASR of the DPO reward (reported only, never deciding).
- `wer_qwen`, `wer_whisper`: the pilot-era metric, kept with its old meaning for comparability: normalizer only (no
  canonicalization) against the item's `text` (never `ref_text`). Known to over-count dotted/spelled acronyms,
  h:mm times and abbreviated units.
- `sim_prompt`: cosine similarity of SpeechBrain ECAPA embeddings of the synthesized audio and the prompt recording
  (16 kHz).
- `wer_real`, `wer_real_canon`: Qwen3-ASR's WER (old metric / canon) on the speaker's real recording of the target
  text, for items that have one (`target_wav`; the attainable floor; reported). `--no-real` skips them.
- `ref_source`: `ref_text` or `text`.
An item whose audio is missing or fails to score is an error record (`error`), never a score. Summary means skip
missing and NaN values (null when none).

`--rescore SCORES_DIR` recomputes every WER field from the transcripts stored in an earlier run's scores.jsonl (no
model is loaded; CPU only): the way to bring a pilot run onto the current `canon_version`. `wer_real_canon` needs the
stored `asr_real` (runs from before canon-1 did not store it; their `wer_real` is copied as is).

`mos_scores(wavs)` is the hook for the naturalness guard; it raises NotImplementedError until a MOS predictor is
pinned.

usage: score_tts.py --gen GEN_DIR --items ITEMS_DIR --out OUT [--no-real]
       score_tts.py --rescore SCORES_DIR --items ITEMS_DIR --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AREA = HERE.parent
MODELS = AREA / "models"
sys.path.insert(0, str(HERE))

import wer_canon  # noqa: E402

WER_FIELDS = ("wer_qwen", "wer_whisper", "wer_qwen_canon", "wer_whisper_canon", "wer_real", "wer_real_canon")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def load_16k(path: Path):
    import soundfile as sf
    import torch
    import torchaudio
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = torch.from_numpy(wav.mean(axis=1))
    return torchaudio.functional.resample(x, sr, 16000) if sr != 16000 else x


def mos_scores(wavs: list[Path]) -> list[float]:
    """Naturalness guard: one predicted MOS per wav. Not implemented until a MOS predictor is pinned (recommended:
    UTMOS22 strong learner; pin its repository revision and checkpoint SHA-256 in models/MODELS.json first)."""
    raise NotImplementedError("no MOS predictor is pinned yet")


def score_texts(item: dict, asr_qwen: str, asr_whisper: str, norm) -> dict:
    """All transcript-based WER fields of one item (pure; shared by the live and the --rescore path)."""
    ref = wer_canon.reference(item)
    return {"wer_qwen": wer_canon.wer_plain(item["text"], asr_qwen, norm),
            "wer_whisper": wer_canon.wer_plain(item["text"], asr_whisper, norm),
            "wer_qwen_canon": wer_canon.wer_canon(ref, asr_qwen, norm),
            "wer_whisper_canon": wer_canon.wer_canon(ref, asr_whisper, norm),
            "ref_source": "ref_text" if item.get("ref_text") else "text"}


def score_real(item: dict, asr_real: str, norm) -> dict:
    return {"wer_real": wer_canon.wer_plain(item["text"], asr_real, norm),
            "wer_real_canon": wer_canon.wer_canon(wer_canon.reference(item), asr_real, norm), "asr_real": asr_real}


def summarize(scores: list[dict]) -> dict:
    ok = [s for s in scores if s["error"] is None]

    def mean(k):
        vals = [s[k] for s in ok if isinstance(s.get(k), (int, float)) and not math.isnan(s[k])]
        return round(sum(vals) / len(vals), 4) if vals else None
    out = {"n": len(scores), "errors": len(scores) - len(ok), "sim_prompt": mean("sim_prompt"),
           "canon_version": wer_canon.CANON_VERSION, "deciding_metric": "wer_qwen_canon"}
    out.update({k: mean(k) for k in WER_FIELDS})
    return out


def read_items(items_dir: Path) -> tuple[dict, str]:
    body = (items_dir / "items.jsonl").read_text(encoding="utf-8")
    items = {json.loads(l)["item"]: json.loads(l) for l in body.splitlines() if l}
    return items, "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def rescore(src: Path, items: dict, norm) -> list[dict]:
    rows = []
    for l in (src / "scores.jsonl").read_text(encoding="utf-8").splitlines():
        if not l:
            continue
        old = json.loads(l)
        row = {k: v for k, v in old.items() if k not in WER_FIELDS}
        if old.get("error") is None:
            row.update(score_texts(items[old["item"]], old["asr_qwen"], old["asr_whisper"], norm))
            if "asr_real" in old:
                row.update(score_real(items[old["item"]], old["asr_real"], norm))
            elif "wer_real" in old:
                row["wer_real"] = old["wer_real"]
        rows.append(row)
    return rows


def _versions() -> dict:
    import importlib.metadata as md
    out = {}
    for p in ("torch", "transformers", "jiwer", "qwen-asr", "speechbrain"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def score_live(gen: Path, items: dict, norm, out: Path, no_real: bool) -> tuple[list[dict], dict]:
    import torch
    from qwen_asr import Qwen3ASRModel
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import pipeline

    out.mkdir(parents=True, exist_ok=False)
    recs = [json.loads(l) for l in (gen / "gen.jsonl").read_text().splitlines() if l]
    qwen = Qwen3ASRModel.from_pretrained(str(MODELS / "qwen3-asr-1.7b"), dtype=torch.bfloat16, device_map="cuda:0",
                                         max_inference_batch_size=16, max_new_tokens=256)
    whisper = pipeline("automatic-speech-recognition", model=str(MODELS / "whisper-large-v3-turbo"),
                       torch_dtype=torch.float16, device="cuda:0")
    # hyperparams.yaml names the Hub repo as `pretrained_path`; point it at the pinned local copy (offline).
    ecapa = EncoderClassifier.from_hparams(source=str(MODELS / "spkrec-ecapa-voxceleb"),
                                           savedir=str(HERE / "ecapa-cache"), run_opts={"device": "cuda:0"},
                                           overrides={"pretrained_path": str(MODELS / "spkrec-ecapa-voxceleb")})
    src = AREA / "data-source"

    def emb(x):
        return ecapa.encode_batch(x.unsqueeze(0).to("cuda:0")).squeeze()

    scores = []
    for r in recs:
        it = items[r["item"]]
        row = {"item": r["item"], "speaker": it["speaker"], "error": None}
        try:
            if r.get("error"):
                raise RuntimeError(f"generation error: {r['error']}")
            wav = gen / r["wav"]
            q = qwen.transcribe(audio=str(wav), language="English")[0].text
            g, p = load_16k(wav), load_16k(src / it["prompt_wav"])
            # Over 30 s Whisper switches to long-form decoding, which needs timestamp tokens; shorter audio keeps the
            # short-form call unchanged.
            extra = {"return_timestamps": True} if g.shape[-1] > 30 * 16000 else {}
            w = whisper(str(wav), generate_kwargs={"language": "english", "task": "transcribe"}, **extra)["text"]
            sim = float(torch.nn.functional.cosine_similarity(emb(g), emb(p), dim=0))
            row.update(score_texts(it, q, w, norm))
            row.update(sim_prompt=sim, asr_qwen=q, asr_whisper=w, audio_seconds=r.get("audio_seconds"))
            if not no_real and it.get("target_wav"):
                real = qwen.transcribe(audio=str(src / it["target_wav"]), language="English")[0].text
                row.update(score_real(it, real, norm))
        except Exception as exc:
            row["error"] = f"{exc.__class__.__name__}: {exc}"[:300]
        scores.append(row)
        print(json.dumps({k: v for k, v in row.items() if not k.startswith("asr_")}), flush=True)
    return scores, {"gen": str(gen.resolve()), "gen_jsonl_sha256": sha256(gen / "gen.jsonl")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen")
    ap.add_argument("--rescore")
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-real", action="store_true")
    a = ap.parse_args()
    if (a.gen is None) == (a.rescore is None):
        raise SystemExit("give exactly one of --gen and --rescore")
    out = Path(a.out)
    items, items_sha = read_items(Path(a.items))
    norm = wer_canon.load_normalizer(MODELS)
    models = {n: json.loads((MODELS / "MODELS.json").read_text())["models"][n]["revision"]
              for n in ("qwen3-asr-1.7b", "whisper-large-v3-turbo", "spkrec-ecapa-voxceleb")}
    if a.rescore:
        out.mkdir(parents=True, exist_ok=False)
        scores = rescore(Path(a.rescore), items, norm)
        source = {"rescore": str(Path(a.rescore).resolve()),
                  "rescore_scores_sha256": sha256(Path(a.rescore) / "scores.jsonl")}
    else:
        scores, source = score_live(Path(a.gen), items, norm, out, a.no_real)
    (out / "scores.jsonl").write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in scores))
    summary = summarize(scores)
    summary.update(items=str(Path(a.items).resolve()), script_sha256=sha256(Path(__file__)), models=models, **source)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    meta = {"script_sha256": sha256(Path(__file__)), "wer_canon_sha256": sha256(HERE / "wer_canon.py"),
            "canon_version": wer_canon.CANON_VERSION, "items": str(Path(a.items).resolve()),
            "items_sha256": items_sha, "models": models, "python": platform.python_version(),
            "versions": _versions(), **source}
    (out / "score_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
