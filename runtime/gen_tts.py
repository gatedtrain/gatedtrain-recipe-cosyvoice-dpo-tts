"""Zero-shot synthesis of an item set with Fun-CosyVoice3-0.5B (src/CosyVoice @ 074ca6dc).

For each item the model speaks `text` in the voice of `prompt_wav`, whose transcript is `prompt_text` (CosyVoice3's
prompt form `You are a helpful assistant.<|endofprompt|>{transcript}`). `--llm` selects the LLM weights: `base` (the
model directory's llm.pt), `rl` (the vendor's llm.rl.pt) or a path to a trained state dict; the flow decoder and the
vocoder are always the model directory's. One seed per item (`derive_seed('tts', seed, item)`), so base and trained
runs sample with the same noise. The text frontend is off (`text_frontend=False`): the LLM reads the raw text, digits
and symbols included, so number reading is the model's own skill (CosyVoice's fallback normalizer mangles thousands
separators and decimals).

Writes <out>/wav/<item>.wav (24 kHz) and <out>/gen.jsonl (item, seconds of audio, wall seconds, error), and
<out>/gen_meta.json (model files' SHA-256, llm weights, seed, peak GPU memory).

usage: gen_tts.py --model-dir models/fun-cosyvoice3-0.5b-2512 --items items-smoke --out OUT [--llm base|rl|PATH]
                  [--seed 11] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "src" / "CosyVoice"), str(HERE / "src" / "CosyVoice" / "third_party" / "Matcha-TTS")]

import torch  # noqa: E402
import torchaudio  # noqa: E402

PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"


def derive_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()[:8], "big") >> 33


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm", default="base")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    # The text frontend is off (raw text to the LLM). Blocking wetext also stops the frontend from fetching its
    # resources from modelscope at load (unpinned, and refused with 403 on this network).
    sys.modules["wetext"] = None
    from cosyvoice.cli.cosyvoice import AutoModel
    from cosyvoice.utils.common import set_all_random_seed

    model_dir = Path(a.model_dir).resolve()
    out = Path(a.out)
    (out / "wav").mkdir(parents=True, exist_ok=False)
    items = [json.loads(l) for l in (Path(a.items) / "items.jsonl").read_text(encoding="utf-8").splitlines() if l]
    items = items[: a.limit]
    t0 = time.time()
    cv = AutoModel(model_dir=str(model_dir))
    llm_file = {"base": model_dir / "llm.pt", "rl": model_dir / "llm.rl.pt"}.get(a.llm, Path(a.llm))
    if a.llm != "base":
        state = torch.load(llm_file, map_location="cpu", weights_only=True)
        cv.model.llm.load_state_dict(state, strict=True)
        cv.model.llm.to(torch.device("cuda")).eval()
    load_s = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    src = HERE / "data-source"
    with (out / "gen.jsonl").open("w", encoding="utf-8") as log:
        for it in items:
            rec = {"item": it["item"], "error": None}
            t = time.time()
            try:
                set_all_random_seed(derive_seed("tts", a.seed, it["item"]))
                chunks = [j["tts_speech"] for j in cv.inference_zero_shot(
                    it["text"], PROMPT_PREFIX + it["prompt_text"], str(src / it["prompt_wav"]), stream=False,
                    text_frontend=False)]
                speech = torch.cat(chunks, dim=1)
                name = it["item"].replace(":", "__") + ".wav"
                torchaudio.save(str(out / "wav" / name), speech.cpu(), cv.sample_rate)
                rec.update(wav=f"wav/{name}", audio_seconds=round(speech.shape[1] / cv.sample_rate, 3))
            except Exception as exc:                      # an item that fails is an error record, not a crash
                rec["error"] = f"{exc.__class__.__name__}: {exc}"[:300]
            rec["wall_seconds"] = round(time.time() - t, 3)
            log.write(json.dumps(rec, sort_keys=True) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
    meta = {"model_dir": str(model_dir), "llm": a.llm, "llm_sha256": sha256(llm_file),
            "files_sha256": {n: sha256(model_dir / n) for n in ("flow.pt", "hift.pt", "campplus.onnx",
                                                                  "speech_tokenizer_v3.onnx", "cosyvoice3.yaml")},
            "items": str(Path(a.items).resolve()), "items_sha256": sha256(Path(a.items) / "items.jsonl"),
            "n_items": len(items), "seed": a.seed, "text_frontend": False,
            "sample_rate": cv.sample_rate, "load_seconds": round(load_s, 1),
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_gpu_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "code": "src/CosyVoice@074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc", "script_sha256": sha256(Path(__file__))}
    (out / "gen_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
