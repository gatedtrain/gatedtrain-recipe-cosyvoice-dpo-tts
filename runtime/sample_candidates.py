"""Sample K zero-shot candidates per train item with Fun-CosyVoice3-0.5B and keep what DPO needs (main environment).

Same synthesis path as gen_tts.py: `inference_zero_shot(text, PROMPT_PREFIX + prompt_text, prompt_wav, stream=False,
text_frontend=False)` with wetext blocked, the model directory's flow and vocoder, the LLM from `--llm` (base by
default). Candidate k of an item is sampled after `set_all_random_seed(derive_seed('cand', seed, item, k))`.

What the LLM sees and emits is captured without touching src/: `LLMCapture` wraps, on the model instance,
`llm.inference` (records its tensor inputs and every token it yields) and `llm.sampling_ids` (records every sampled
id and its log-prob under the full distribution, read before sampling masks it). This gives, per candidate:
- the LLM's speech tokens as sampled, including the silent/breath tokens that `CosyVoice3Model.llm_job`
  (src/CosyVoice/cosyvoice/cli/model.py:101-131) drops from the flow input after 5 in a row; the DPO sequence is the
  LLM's own output, not the flow's input;
- the stop id that ended sampling (any of `speech_token_size + 0..199`, Qwen2LM.inference_wrapper llm.py:538-549;
  normally eos = 6562), or -1 when sampling hit max_len (20 x text tokens);
- per-token log-probs at sampling time (fp32; train_dpo.py compares its teacher-forced reference log-probs to them).
Per item (identical for all k; asserted): the target text token ids, the prompt text token ids (prefix + transcript;
the LLM reads prompt_text + text, llm.py:474-476), the LLM prompt speech tokens (after the frontend's 24 kHz
truncation to min(feat/2, tokens), frontend.py:174-178) and the CAM++ embedding the frontend passes as
`llm_embedding`. CosyVoice3's LLM does not use that embedding (Qwen2LM.inference never reads `embedding`); it is
stored for completeness only.

Writes, under --out:
- wav/<item>__k<k>.wav (24 kHz, from the flow tokens as in gen_tts.py);
- tokens/<item>.npz: text_token, prompt_text_token, prompt_speech_token, llm_embedding, and per k cand<k>_token,
  cand<k>_stop, cand<k>_logp, cand<k>_seed;
- candidates.jsonl: one record per candidate (item, k, seed, wav, audio_seconds, n_tokens, n_flow_tokens, stop,
  sum_logp, wall_seconds, error);
- candidates_meta.json: config (checked on resume), item set hash, model files' SHA-256, versions, peak GPU memory.
Resumable: an item is done once its token shard exists and it has K records; a rerun with the same arguments skips
done items and drops partial records. Refuses an item set whose text split is not `train` (B18).

usage: sample_candidates.py --model-dir models/fun-cosyvoice3-0.5b-2512 --items items-hard-train --out OUT [--k 6]
                            [--seed 11] [--llm base|rl|PATH] [--limit N]
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
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gen_tts import PROMPT_PREFIX, derive_seed, sha256  # noqa: E402  (also puts src/CosyVoice on sys.path)

SILENT_TOKENS = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]   # CosyVoice3Model.silent_tokens (model.py:423)
MAX_SILENT = 5                                                         # llm_job's max_silent_token_num (model.py:102)
CONFIG_KEYS = ("k", "seed", "llm", "items_sha256", "llm_sha256")


class LLMCapture:
    """Records one `llm.inference` call: its inputs, the yielded tokens, every sampled id and its log-prob."""

    def __init__(self, llm):
        self.llm = llm
        self._inference, self._sampling_ids = llm.inference, llm.sampling_ids
        llm.inference, llm.sampling_ids = self._wrap_inference, self._wrap_sampling_ids
        self.reset()

    def reset(self) -> None:
        self.calls, self.inputs, self.yielded, self.sampled, self.logps = 0, None, [], [], []

    def _wrap_inference(self, **kw):
        self.calls += 1
        self.inputs = {k: kw[k].detach().to("cpu").clone() for k in ("text", "prompt_text", "prompt_speech_token",
                                                                    "embedding")}
        for t in self._inference(**kw):
            self.yielded.append(int(t))
            yield t

    def _wrap_sampling_ids(self, weighted_scores, decoded_tokens, sampling, ignore_eos=True):
        lp = weighted_scores.detach().float().clone()          # sampling_ids masks weighted_scores in place
        tid = self._sampling_ids(weighted_scores, decoded_tokens, sampling, ignore_eos=ignore_eos)
        self.sampled.append(int(tid))
        self.logps.append(float(lp[int(tid)]))
        return tid

    def restore(self) -> None:
        del self.llm.inference, self.llm.sampling_ids          # back to the class methods

    def result(self) -> tuple[list[int], int | None, list[float]]:
        """(speech tokens, stop id or None at max_len, log-probs of tokens + stop)."""
        if self.calls != 1:
            raise RuntimeError(f"expected one LLM call per candidate, saw {self.calls}")
        n = len(self.yielded)
        if self.sampled[:n] != self.yielded:
            raise RuntimeError("sampled ids and yielded tokens disagree")
        if len(self.sampled) == n + 1 and self.sampled[-1] in self.llm.stop_token_ids:
            return self.yielded, self.sampled[-1], self.logps
        if len(self.sampled) == n:
            return self.yielded, None, self.logps
        raise RuntimeError(f"{len(self.sampled)} sampled ids for {n} tokens")


def flow_token_count(tokens: list[int]) -> int:
    """How many tokens llm_job passes to the flow (it drops silent tokens beyond 5 in a row)."""
    n, run = 0, 0
    for t in tokens:
        if t in SILENT_TOKENS:
            run += 1
            if run > MAX_SILENT:
                continue
        else:
            run = 0
        n += 1
    return n


def item_name(item: str) -> str:
    return item.replace(":", "__")


def sample_item(cv, cap: LLMCapture, it: dict, k: int, seed: int, src: Path, out: Path, set_seed) -> tuple:
    """K candidates of one item: (records, arrays for the shard)."""
    arrays: dict[str, np.ndarray] = {}
    recs = []
    for kk in range(k):
        s = derive_seed("cand", seed, it["item"], kk)
        rec = {"item": it["item"], "k": kk, "seed": s, "error": None}
        t = time.time()
        cap.reset()
        try:
            set_seed(s)
            chunks = [j["tts_speech"] for j in cv.inference_zero_shot(
                it["text"], PROMPT_PREFIX + it["prompt_text"], str(src / it["prompt_wav"]), stream=False,
                text_frontend=False)]
            speech = torch.cat(chunks, dim=1)
            tokens, stop, logps = cap.result()
            inp = {"text_token": cap.inputs["text"], "prompt_text_token": cap.inputs["prompt_text"],
                   "prompt_speech_token": cap.inputs["prompt_speech_token"], "llm_embedding": cap.inputs["embedding"]}
            for name, v in inp.items():
                v = v.numpy().reshape(-1).astype(np.float32 if name == "llm_embedding" else np.int32)
                if name in arrays and not np.array_equal(arrays[name], v):
                    raise RuntimeError(f"{name} differs between candidates")
                arrays[name] = v
            arrays[f"cand{kk}_token"] = np.asarray(tokens, dtype=np.int32)
            arrays[f"cand{kk}_stop"] = np.asarray(-1 if stop is None else stop, dtype=np.int64)
            arrays[f"cand{kk}_logp"] = np.asarray(logps, dtype=np.float32)
            arrays[f"cand{kk}_seed"] = np.asarray(s, dtype=np.int64)
            wav = f"wav/{item_name(it['item'])}__k{kk}.wav"
            save_wav(out / wav, speech, cv.sample_rate)
            rec.update(wav=wav, audio_seconds=round(speech.shape[1] / cv.sample_rate, 3), n_tokens=len(tokens),
                       n_flow_tokens=flow_token_count(tokens), stop=stop, sum_logp=round(float(np.sum(logps)), 4))
        except Exception as exc:                      # a failed candidate is an error record, not a crash
            rec["error"] = f"{exc.__class__.__name__}: {exc}"[:300]
        rec["wall_seconds"] = round(time.time() - t, 3)
        recs.append(rec)
    return recs, arrays


def save_wav(path: Path, speech: torch.Tensor, sr: int) -> None:
    import torchaudio
    torchaudio.save(str(path), speech.cpu(), sr)


def read_items(items_dir: Path) -> tuple[list[dict], dict]:
    body = (items_dir / "items.jsonl").read_text(encoding="utf-8")
    meta = json.loads((items_dir / "item_set_meta.json").read_text())
    if "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest() != meta["items_sha256"]:
        raise SystemExit("items.jsonl does not match item_set_meta.json")
    if (meta.get("text_split"), meta.get("voice_split")) != ("train", "train_clean_100"):
        raise SystemExit(f"{items_dir} is not a training item set (text split {meta.get('text_split')!r}, voices "
                         f"{meta.get('voice_split')!r}); candidates are sampled for the train set only")
    return [json.loads(l) for l in body.splitlines() if l], meta


def done_items(out: Path, k: int) -> tuple[set[str], list[str]]:
    """Items with a shard and K records; returns them and the records to keep (partial items dropped)."""
    path = out / "candidates.jsonl"
    if not path.exists():
        return set(), []
    by: dict[str, list[str]] = {}
    for l in path.read_text(encoding="utf-8").splitlines():
        try:
            by.setdefault(json.loads(l)["item"], []).append(l)
        except (json.JSONDecodeError, KeyError):
            continue                                  # a torn last line
    done = {i for i, ls in by.items() if len(ls) == k and (out / "tokens" / (item_name(i) + ".npz")).exists()}
    return done, [l for i in sorted(by) if i in done for l in by[i]]


def check_resume(meta_path: Path, config: dict) -> None:
    """An existing output directory is resumed only with the same K, seed, LLM and item set."""
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if {c: old.get(c) for c in CONFIG_KEYS} != config:
            raise SystemExit(f"{meta_path.parent} was sampled with {({c: old.get(c) for c in CONFIG_KEYS})}; "
                             "not resuming")


def run(cv, items: list[dict], out: Path, k: int, seed: int, src: Path, set_seed, log=print) -> dict:
    """Sample all items not yet done; returns counts. `cv` needs .model.llm, .sample_rate, .inference_zero_shot."""
    (out / "wav").mkdir(parents=True, exist_ok=True)
    (out / "tokens").mkdir(exist_ok=True)
    done, keep = done_items(out, k)
    (out / "candidates.jsonl").write_text("".join(l + "\n" for l in keep), encoding="utf-8")
    cap = LLMCapture(cv.model.llm)
    n_new = 0
    try:
        with (out / "candidates.jsonl").open("a", encoding="utf-8") as f:
            for it in items:
                if it["item"] in done:
                    continue
                recs, arrays = sample_item(cv, cap, it, k, seed, src, out, set_seed)
                tmp = out / "tokens" / (item_name(it["item"]) + ".tmp.npz")
                np.savez(tmp, **arrays)
                tmp.replace(out / "tokens" / (item_name(it["item"]) + ".npz"))
                f.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))
                f.flush()
                n_new += 1
                log(json.dumps({"item": it["item"], "errors": sum(r["error"] is not None for r in recs),
                                "audio_seconds": [r.get("audio_seconds") for r in recs],
                                "wall_seconds": round(sum(r["wall_seconds"] for r in recs), 2)}))
    finally:
        cap.restore()
    return {"skipped": len(done), "sampled": n_new}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--llm", default="base")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    model_dir = Path(a.model_dir).resolve()
    out = Path(a.out)
    items, imeta = read_items(Path(a.items))
    items = items[: a.limit]
    llm_file = {"base": model_dir / "llm.pt", "rl": model_dir / "llm.rl.pt"}.get(a.llm, Path(a.llm))
    config = {"k": a.k, "seed": a.seed, "llm": a.llm, "items_sha256": imeta["items_sha256"],
              "llm_sha256": sha256(llm_file)}
    meta_path = out / "candidates_meta.json"
    check_resume(meta_path, config)
    out.mkdir(parents=True, exist_ok=True)
    meta = {**config, "items": str(Path(a.items).resolve()), "text_split": imeta["text_split"],
            "voice_split": imeta["voice_split"],
            "n_items_in_set": imeta["n_items"], "limit": a.limit,
            "model_dir": str(model_dir), "text_frontend": False, "prompt_prefix": PROMPT_PREFIX,
            "files_sha256": {n: sha256(model_dir / n) for n in ("flow.pt", "hift.pt", "campplus.onnx",
                                                                  "speech_tokenizer_v3.onnx", "cosyvoice3.yaml")},
            "code": "src/CosyVoice@074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc", "script_sha256": sha256(Path(__file__)),
            "gen_tts_sha256": sha256(HERE / "gen_tts.py"), "python": platform.python_version()}
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    # As gen_tts.py: raw text to the LLM; blocking wetext also stops the frontend fetching resources at load.
    sys.modules["wetext"] = None
    from cosyvoice.cli.cosyvoice import AutoModel
    from cosyvoice.utils.common import set_all_random_seed
    t0 = time.time()
    cv = AutoModel(model_dir=str(model_dir))
    if a.llm != "base":
        cv.model.llm.load_state_dict(torch.load(llm_file, map_location="cpu", weights_only=True), strict=True)
        cv.model.llm.to(torch.device("cuda")).eval()
    load_s = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    counts = run(cv, items, out, a.k, a.seed, HERE / "data-source", set_all_random_seed)
    import importlib.metadata as md
    recs = [json.loads(l) for l in (out / "candidates.jsonl").read_text().splitlines() if l]
    meta.update(counts, n_records=len(recs), n_errors=sum(r["error"] is not None for r in recs),
                n_max_len=sum(r.get("stop", 0) is None and r["error"] is None for r in recs),
                candidates_sha256=sha256(out / "candidates.jsonl"), load_seconds=round(load_s, 1),
                wall_seconds=round(time.time() - t0, 1),
                peak_gpu_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                versions={p: md.version(p) for p in ("torch", "torchaudio", "transformers", "numpy")})
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("skipped", "sampled", "n_records", "n_errors", "n_max_len",
                                           "wall_seconds", "peak_gpu_gib")}))


if __name__ == "__main__":
    main()
