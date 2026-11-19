"""DPO on Fun-CosyVoice3-0.5B's speech-token LLM from WER-ranked candidate pairs (flow and vocoder untouched).

Not the upstream `Qwen2LM.forward_dpo` (src/CosyVoice/cosyvoice/llm/llm.py:407-456), which (a) averages log-probs over
the IGNORE_ID positions (`chosen_lm_mask = chosen_lm_target == IGNORE_ID`), i.e. over prompt and padding instead of
the speech tokens, (b) hard-codes `speech_token_size + 3` classes (CosyVoice3LM has `speech_token_size + 200`), and
(c) conditions without the zero-shot prompt (no prompt text, no prompt speech tokens). Here every sequence is built
exactly as zero-shot inference conditions the LLM (`Qwen2LM.inference`, llm.py:459-503, with CosyVoice3LM's ids,
llm.py:664-690):

    input : [sos] + embed_tokens(prompt_text + text) + [task_id] + speech_embedding(prompt_speech) + speech_embedding(y)
    target: y_1 .. y_n, stop        (one per position from the last prompt-speech position on; nothing else)

where `sos` and `task_id` are rows `speech_token_size + 0` and `+ 2` of `speech_embedding`, `prompt_text` is the
tokenized `You are a helpful assistant.<|endofprompt|>` + prompt transcript, `y` the candidate's LLM tokens as
sampled (including silent tokens that `CosyVoice3Model.llm_job` later drops before the flow) and `stop` the stop id
the sampler drew (normally eos = `speech_token_size + 1`). The prompt speech tokens are context only. The sequence
log-prob is the SUM over those n + 1 targets of log_softmax over all `speech_token_size + 200` outputs (the model's
full distribution; sampling's top-p/top-k/RAS truncation is not part of the policy's log-prob).
tests/test_main_dpo.py checks that the teacher-forced log-probs equal the step-wise log-probs of the model's own
`inference()` on a tiny random CosyVoice3LM.

Loss: sigmoid DPO, -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))) on summed log-probs, plus
`--nll-weight` x (chosen NLL per token) (default 0). The frozen reference is the initial LLM (`--llm`, default the
model directory's llm.pt): its log-probs are computed once, by the same forward before the first update, and cached
(`ref_logps.jsonl`), so no second model is held in memory and the step-0 log-ratios are 0 up to batch-shape
numerics (loss = log 2 = 0.6931 at step 1 in fp32; within bf16 noise under `--amp bf16`).
Policy: full fine-tune of the LLM (Qwen2 body, text embeddings, speech embedding, decoder), fp32 master weights,
bf16 autocast forward (`--amp bf16`), AdamW (betas 0.9/0.95, no weight decay), linear warmup then constant LR,
gradient-norm clipping, micro-batches of `--micro-batch` pairs (both sequences of a pair in one forward) with
`--grad-accum` accumulation. Seeds: torch/numpy/random from `--seed`; pair order per epoch from
Random(f"dpo:{seed}:{epoch}").

Logged per optimizer step (train_log.jsonl): loss, reward accuracy (share of pairs whose implicit reward margin is
> 0), mean margin, chosen/rejected implicit rewards beta*(pi-ref), mean policy log-probs of chosen and rejected,
`kl_proxy` = mean per-token (ref - pi) log-ratio over chosen and rejected tokens (the candidates are samples of the
reference, so this estimates KL(ref || pi) per token), grad norm, LR, tokens, seconds, peak GPU memory.

Data (B18): reads only <pairs>/pairs.jsonl and the candidates' token shards; refuses a candidate set whose item set
is not the train set (text split train AND voices from train_clean_100; the dev set shares the train text split),
and never reads an item set or any Qwen3-ASR output.

Output layout (the gatedtrain launcher layout; the adapter `cosyvoice_dpo` of this recipe reads it). `--out` is a new or
empty directory:
- `run_args.json` (before the first update): profile, seed, the resolved settings, the planned number of optimizer
  steps, the base binding (`base_ref`: repo, revision and SHA-256 of every base file the LLM is built from, checked
  against models/MODELS.json when it lists the model directory), the data binding (pairs file, its SHA-256 and row
  count, the candidates' hashes, the reference cache and its SHA-256), the script's SHA-256;
- `env-freeze.txt` (`name==version` of every installed distribution) and `env-runtime.json` (python, the script's
  SHA-256, the freeze's SHA-256);
- `samples.jsonl`: one line per pair of each optimizer step, written after the step's backward passes and before
  `opt.step()`: step, epoch, micro-batch, position, the pair's 0-based line in pairs.jsonl, its item, chosen_k,
  rejected_k and `row` = SHA-256 of that line's text;
- `checkpoint-<N>/` at every `--save-every` multiple and at the last step: `llm.pt` (state dict with exactly llm.pt's
  keys, fp32; `gen_tts.py --llm <cut>/llm.pt` loads it strictly), `base_ref.json`, then `cut.json` last (global step,
  pairs consumed, SHA-256 of the other files, deployable files, script SHA-256, seed, profile), written in
  `.checkpoint-<N>.partial/` and renamed;
- `train_log.jsonl`, `ref_logps.jsonl`, `train_meta.json` as before (`llm_dpo_sha256` is the last cut's llm.pt).
Saving and logging read the weights and the schedule only; the training computation is the one of the published run.

GPU memory (RTX 4090, 24 GB): 506 M unique parameters (llm.pt has 642 M entries; lm_head is tied to the text
embeddings) x 16 bytes (fp32 weights, grads, AdamW m and v) = 8.1 GB; no reference model on the GPU; activations
~78 KB per token per layer under bf16 autocast without checkpointing (measured with autograd saved-tensor hooks on a
full-width 1- and 2-layer Qwen2 at 4 x 800 tokens on CPU; SDPA keeps no L x L probabilities), i.e. ~1.9 MB per token
for 24 layers: ~8 GB for the worst micro-batch of 2 pairs (4 sequences of ~1,050 tokens, the 70-word natural items),
~4 GB for typical hard items (~550 tokens); CUDA context and allocator slack ~1.5 GB: ~18 GB worst case.
`--grad-ckpt` keeps only layer inputs (~0.5 GB of activations) for ~30% more time.

usage: train_dpo.py --model-dir models/fun-cosyvoice3-0.5b-2512 --candidates CAND_DIR --pairs PAIRS_DIR --out OUT
                    [--profile smoke|full] [--beta 0.1] [--lr 1e-6] [--epochs 2] [--micro-batch 2] [--grad-accum 8]
                    [--warmup-steps 10] [--max-steps N] [--nll-weight 0] [--clip 1.0] [--seed 11] [--amp bf16|none]
                    [--grad-ckpt] [--llm PATH] [--ref-cache FILE] [--save-every N] [--device cuda|cpu]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "src" / "CosyVoice"), str(HERE / "src" / "CosyVoice" / "third_party" / "Matcha-TTS")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CODE = "src/CosyVoice@074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
CUT_MARKER = "cut.json"
CUT_WEIGHTS = "llm.pt"                    # the trained LLM state dict; replaces the model directory's llm.pt
CUT_FILES = (CUT_WEIGHTS, "base_ref.json")
DEPLOYABLE = (CUT_WEIGHTS,)
# Files of the model directory the trained LLM is built from (load_llm): the initial weights, the model config and the
# text LM directory (Qwen2 config and tokenizer). Each present one is hashed into base_ref.
BASE_FILES = ("llm.pt", "cosyvoice3.yaml")
BASE_DIRS = ("CosyVoice-BlankEN",)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Sequence construction and log-probs (the part tests/test_main_dpo.py checks against llm.inference)


def build_example(prompt_text_token, text_token, prompt_speech_token, speech_token, stop_token) -> dict:
    """One training sequence. `stop_token` None means the sample hit max_len (no stop target)."""
    speech = [int(t) for t in speech_token]
    return {"text": [int(t) for t in prompt_text_token] + [int(t) for t in text_token],
            "prompt_speech": [int(t) for t in prompt_speech_token], "speech": speech,
            "targets": speech + ([int(stop_token)] if stop_token is not None else [])}


def embed_example(llm, ex: dict, device) -> tuple[torch.Tensor, int]:
    """[L, D] input embeddings and the index of the first predicting position (the last prompt-speech position)."""
    emb = llm.speech_embedding.weight
    t = lambda ids: torch.tensor(ids, dtype=torch.long, device=device)  # noqa: E731
    parts = [emb[llm.sos].reshape(1, -1), llm.llm.model.model.embed_tokens(t(ex["text"])), emb[llm.task_id].reshape(1, -1)]
    if ex["prompt_speech"]:
        parts.append(llm.speech_embedding(t(ex["prompt_speech"])))
    if ex["speech"]:
        parts.append(llm.speech_embedding(t(ex["speech"])))
    x = torch.cat(parts, dim=0)
    first = 1 + len(ex["text"]) + 1 + len(ex["prompt_speech"]) - 1
    return x, first


def token_logps(llm, examples: list[dict], device, amp_dtype=None) -> list[torch.Tensor]:
    """Per-target log-probs of each example (teacher forcing, one padded batch, right padding)."""
    xs, firsts = zip(*(embed_example(llm, ex, device) for ex in examples))
    lens = [x.shape[0] for x in xs]
    L, D = max(lens), xs[0].shape[1]
    x = torch.zeros(len(xs), L, D, dtype=xs[0].dtype, device=device)
    mask = torch.zeros(len(xs), L, dtype=torch.long, device=device)
    for i, xi in enumerate(xs):
        x[i, : lens[i]] = xi
        mask[i, : lens[i]] = 1
    rows, cols, tgt = [], [], []
    for i, (ex, f) in enumerate(zip(examples, firsts)):
        n = len(ex["targets"])
        rows += [i] * n
        cols += list(range(f, f + n))
        tgt += ex["targets"]
    ctx = torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype) if amp_dtype else _null()
    with ctx:
        h = llm.llm.model.model(inputs_embeds=x, attention_mask=mask, use_cache=False).last_hidden_state
        logits = llm.llm_decoder(h[torch.tensor(rows, device=device), torch.tensor(cols, device=device)])
    lp = logits.float().log_softmax(dim=-1).gather(1, torch.tensor(tgt, device=device).unsqueeze(1)).squeeze(1)
    out, k = [], 0
    for ex in examples:
        out.append(lp[k: k + len(ex["targets"])])
        k += len(ex["targets"])
    return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def sequence_logps(llm, examples, device, amp_dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
    lps = token_logps(llm, examples, device, amp_dtype)
    return torch.stack([lp.sum() for lp in lps]), torch.tensor([lp.numel() for lp in lps], device=device)


def dpo_loss(pc, pr, rc, rr, beta: float, nll_weight: float = 0.0, n_chosen=None) -> tuple[torch.Tensor, dict]:
    """Sigmoid DPO on summed log-probs (policy chosen/rejected, reference chosen/rejected); mean over pairs."""
    margin = (pc - rc) - (pr - rr)
    losses = -F.logsigmoid(beta * margin)
    if nll_weight:
        losses = losses + nll_weight * (-pc / n_chosen)
    stats = {"reward_acc": (margin > 0).float().mean().item(), "margin": (beta * margin).mean().item(),
             "chosen_reward": (beta * (pc - rc)).mean().item(), "rejected_reward": (beta * (pr - rr)).mean().item()}
    return losses.mean(), stats


# ---------------------------------------------------------------------------------------------------------------
# Data


def compact_sha256(obj) -> str:
    """SHA-256 of compact sorted JSON (the encoding of base_ref `files_sha256`)."""
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False).encode("utf-8")).hexdigest()


def write_json(path: Path, doc) -> None:
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def base_binding(model_dir: Path, llm_file: Path) -> dict:
    """The base files the LLM is built from with their SHA-256, checked against models/MODELS.json (the parent of the
    model directory) when it lists this model directory: `verified` true; a mismatch refuses the run; no MODELS.json
    entry leaves `verified` false."""
    rels = [r for r in BASE_FILES if (model_dir / r).is_file()]
    for d in BASE_DIRS:
        if (model_dir / d).is_dir():
            rels += sorted(p.relative_to(model_dir).as_posix() for p in (model_dir / d).rglob("*") if p.is_file())
    files = {r: sha256(model_dir / r) for r in rels}
    repo = revision = None
    verified = False
    manifest = model_dir.parent / "MODELS.json"
    if manifest.is_file():
        entry = json.loads(manifest.read_text(encoding="utf-8")).get("models", {}).get(model_dir.name)
        if entry:
            repo, revision = entry.get("repo"), entry.get("revision")
            want = {r: "sha256:" + entry["files"][r]["sha256"] for r in rels if r in entry.get("files", {})}
            bad = sorted(r for r in rels if want.get(r) != files[r])
            if bad:
                raise SystemExit(f"base files differ from {manifest}: {bad}")
            verified = True
    return {"repo": repo, "revision": revision, "files": files, "files_sha256": compact_sha256(files),
            "verified": verified, "llm_init_is_base": llm_file.resolve() == (model_dir / "llm.pt").resolve()}


def write_env(out: Path) -> dict:
    """env-freeze.txt (`name==version` of every installed distribution) and env-runtime.json."""
    pkgs = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            pkgs[name.lower().replace("_", "-")] = dist.version
    (out / "env-freeze.txt").write_text("".join(f"{k}=={pkgs[k]}\n" for k in sorted(pkgs)), encoding="utf-8")
    lock = HERE / "uv.lock"
    runtime = {"python": platform.python_version(), "executable": sys.executable,
               "image_cuda_version": os.environ.get("CUDA_VERSION"),
               "uv_lock_sha256": sha256(lock) if lock.is_file() else None,
               "script_sha256": sha256(Path(__file__).resolve()), "code": CODE,
               "uid": os.getuid(), "gid": os.getgid(), "env_freeze_sha256": sha256(out / "env-freeze.txt")}
    write_json(out / "env-runtime.json", runtime)
    return runtime


def save_cut(out: Path, step: int, pairs_consumed: int, state_dict: dict, base_ref: dict, extra: dict) -> Path:
    """checkpoint-<step>/ via .checkpoint-<step>.partial/ and an atomic rename: llm.pt, base_ref.json, then cut.json
    last with the SHA-256 of the other files."""
    cut = out / f"checkpoint-{step}"
    tmp = out / f".checkpoint-{step}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    torch.save(state_dict, tmp / CUT_WEIGHTS)
    write_json(tmp / "base_ref.json", {k: base_ref[k] for k in ("repo", "revision", "files", "files_sha256")})
    write_json(tmp / CUT_MARKER, {
        "schema": "gatedtrain_recipe_cosyvoice_dpo_tts.cut", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "global_step": step, "pairs_consumed": pairs_consumed, "files": {n: sha256(tmp / n) for n in CUT_FILES},
        "deployable": list(DEPLOYABLE), **extra})
    for p in tmp.iterdir():
        fd = os.open(p, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    os.replace(tmp, cut)
    return cut


def item_file(item: str) -> str:
    return item.replace(":", "__") + ".npz"


def load_pairs(cand_dir: Path, pairs_dir: Path) -> tuple[list[dict], dict]:
    """Pairs with both sequences' examples. Refuses a candidate set built from a non-train item set (B18)."""
    cmeta = json.loads((cand_dir / "candidates_meta.json").read_text())
    if (cmeta.get("text_split"), cmeta.get("voice_split")) != ("train", "train_clean_100"):
        raise SystemExit(f"candidates come from a {cmeta.get('text_split')!r}/{cmeta.get('voice_split')!r} item set; "
                         "DPO trains on the train set only")
    pmeta = json.loads((pairs_dir / "pairs_meta.json").read_text())
    if pmeta.get("candidates_sha256") != sha256(cand_dir / "candidates.jsonl"):
        raise SystemExit("pairs were made from another candidates.jsonl")
    lines = [l for l in (pairs_dir / "pairs.jsonl").read_text().splitlines() if l]
    pairs = [json.loads(l) for l in lines]
    rows = ["sha256:" + hashlib.sha256(l.encode("utf-8")).hexdigest() for l in lines]
    cache: dict[str, dict] = {}
    out = []
    for line, (p, row) in enumerate(zip(pairs, rows)):
        if p["item"] not in cache:
            with np.load(cand_dir / "tokens" / item_file(p["item"])) as z:
                cache[p["item"]] = {k: z[k] for k in z.files}
        z = cache[p["item"]]

        def ex(k):
            stop = int(z[f"cand{k}_stop"])
            return build_example(z["prompt_text_token"], z["text_token"], z["prompt_speech_token"],
                                 z[f"cand{k}_token"], None if stop < 0 else stop)
        out.append({**p, "chosen_ex": ex(p["chosen_k"]), "rejected_ex": ex(p["rejected_k"]),
                    "chosen_sample_logp": float(z[f"cand{p['chosen_k']}_logp"].sum()), "_line": line, "_row": row})
    return out, {"candidates_meta": cmeta, "pairs_meta": pmeta}


def pair_key(p: dict, side: str) -> str:
    return f"{p['item']}#{p[side + '_k']}"


@torch.no_grad()
def reference_logps(llm, pairs, device, amp_dtype, batch: int) -> dict[str, float]:
    llm.eval()
    seqs = {}
    for p in pairs:
        seqs[pair_key(p, "chosen")] = p["chosen_ex"]
        seqs[pair_key(p, "rejected")] = p["rejected_ex"]
    keys = sorted(seqs)
    out = {}
    for i in range(0, len(keys), batch):
        ks = keys[i: i + batch]
        s, _ = sequence_logps(llm, [seqs[k] for k in ks], device, amp_dtype)
        out.update({k: float(v) for k, v in zip(ks, s.tolist())})
    return out


# ---------------------------------------------------------------------------------------------------------------
# Training


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def schedule(n_pairs: int, epochs: int, micro_batch: int, grad_accum: int, seed: int, max_steps: int | None):
    """Optimizer steps as lists of micro-batches of pair indices; per-epoch shuffle, last partial step kept."""
    order = []
    for e in range(epochs):
        idx = list(range(n_pairs))
        random.Random(f"dpo:{seed}:{e}").shuffle(idx)
        order += idx
    micro = [order[i: i + micro_batch] for i in range(0, len(order), micro_batch)]
    steps = [micro[i: i + grad_accum] for i in range(0, len(micro), grad_accum)]
    return steps[:max_steps] if max_steps else steps


def train(llm, pairs, ref: dict, a, device, log_path: Path, save_fn=None, samples_path: Path | None = None) -> dict:
    """The DPO loop; `a` holds beta, lr, epochs, micro_batch, grad_accum, warmup_steps, max_steps, nll_weight, clip,
    seed, amp. Returns the last step's record. `samples_path`: the pairs of each step are logged there before
    `opt.step()`; `save_fn(step, pairs_consumed)`: called at every `a.save_every` multiple and after the last step."""
    amp_dtype = torch.bfloat16 if a.amp == "bf16" else None
    steps = schedule(len(pairs), a.epochs, a.micro_batch, a.grad_accum, a.seed, a.max_steps)
    params = [p for p in llm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup_steps)))
    llm.train()
    rec = {}
    consumed = 0                                   # pairs applied so far = the position in the concatenated epoch order
    samples = samples_path.open("a", encoding="utf-8") if samples_path is not None else None
    with log_path.open("a", encoding="utf-8") as log:
        for step, micros in enumerate(steps):
            t0 = time.time()
            n_pairs = sum(len(m) for m in micros)
            agg = {"loss": 0.0, "reward_acc": 0.0, "margin": 0.0, "chosen_reward": 0.0, "rejected_reward": 0.0,
                   "logp_chosen": 0.0, "logp_rejected": 0.0}
            lr_tok, n_tok = 0.0, 0
            opt.zero_grad(set_to_none=True)
            for m in micros:
                batch = [pairs[i] for i in m]
                exs = [p["chosen_ex"] for p in batch] + [p["rejected_ex"] for p in batch]
                s, n = sequence_logps(llm, exs, device, amp_dtype)
                pc, pr = s[: len(batch)], s[len(batch):]
                rc = torch.tensor([ref[pair_key(p, "chosen")] for p in batch], device=device)
                rr = torch.tensor([ref[pair_key(p, "rejected")] for p in batch], device=device)
                loss, st = dpo_loss(pc, pr, rc, rr, a.beta, a.nll_weight, n[: len(batch)])
                (loss * len(batch) / n_pairs).backward()
                w = len(batch) / n_pairs
                agg["loss"] += loss.item() * w
                for k in ("reward_acc", "margin", "chosen_reward", "rejected_reward"):
                    agg[k] += st[k] * w
                agg["logp_chosen"] += pc.mean().item() * w
                agg["logp_rejected"] += pr.mean().item() * w
                lr_tok += float(((rc - pc.detach()).sum() + (rr - pr.detach()).sum()).item())
                n_tok += int(n.sum().item())
            gnorm = torch.nn.utils.clip_grad_norm_(params, a.clip)
            for mb, m in enumerate(micros):
                for pos, i in enumerate(m):
                    p = pairs[i]
                    if samples is not None:
                        samples.write(json.dumps({"step": step + 1, "epoch": consumed // len(pairs), "micro_batch": mb,
                                                  "position": pos, "pair_line": p.get("_line", i), "item": p["item"],
                                                  "chosen_k": p["chosen_k"], "rejected_k": p["rejected_k"],
                                                  "row": p.get("_row")}, sort_keys=True) + "\n")
                    consumed += 1
            if samples is not None:
                samples.flush()
            opt.step()
            sched.step()
            rec = {"step": step + 1, "lr": opt.param_groups[0]["lr"], **{k: round(v, 5) for k, v in agg.items()},
                   "kl_proxy": round(lr_tok / max(1, n_tok), 6), "grad_norm": round(float(gnorm), 4),
                   "pairs": n_pairs, "tokens": n_tok, "seconds": round(time.time() - t0, 3)}
            if torch.device(device).type == "cuda":
                rec["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if not all(math.isfinite(v) for v in (agg["loss"], float(gnorm))):
                raise SystemExit(f"non-finite loss or grad norm at step {step + 1}: {rec}")
            log.write(json.dumps(rec, sort_keys=True) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            if save_fn and ((a.save_every and (step + 1) % a.save_every == 0) or step + 1 == len(steps)):
                save_fn(step + 1, consumed)
    if samples is not None:
        samples.close()
    return rec


def load_llm(model_dir: Path, llm_file: Path, device):
    """CosyVoice3LM exactly as AutoModel builds it (cosyvoice3.yaml), weights from `llm_file` (strict)."""
    from hyperpyyaml import load_hyperpyyaml
    with (model_dir / "cosyvoice3.yaml").open() as f:
        configs = load_hyperpyyaml(f, overrides={"qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN")})
    llm = configs["llm"]
    del configs
    llm.load_state_dict(torch.load(llm_file, map_location="cpu", weights_only=True), strict=True)
    return llm.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True, help="a new or empty directory")
    ap.add_argument("--profile", choices=["smoke", "full"], default="full",
                    help="recorded in run_args.json and every cut.json (the gatedtrain launch profile)")
    ap.add_argument("--llm", default=None, help="initial and reference LLM state dict (default <model-dir>/llm.pt)")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--nll-weight", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--ref-cache", default=None)
    ap.add_argument("--ref-batch", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--device", default=None, help="default: cuda when available")
    a = ap.parse_args()
    model_dir = Path(a.model_dir).resolve()
    cand_dir, pairs_dir, out = Path(a.candidates), Path(a.pairs), Path(a.out)
    llm_file = Path(a.llm) if a.llm else model_dir / "llm.pt"
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"--out {out} is not empty: a run writes into a new, empty directory")
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    on_gpu = device.type == "cuda"
    set_seeds(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pairs, metas = load_pairs(cand_dir, pairs_dir)
    if not pairs:
        raise SystemExit("no pairs")
    script_sha = sha256(Path(__file__).resolve())
    base_ref = base_binding(model_dir, llm_file)
    n_steps = len(schedule(len(pairs), a.epochs, a.micro_batch, a.grad_accum, a.seed, a.max_steps))
    settings = {k: getattr(a, k) for k in ("beta", "lr", "epochs", "micro_batch", "grad_accum", "warmup_steps",
                                           "max_steps", "nll_weight", "clip", "seed", "amp", "grad_ckpt", "ref_batch",
                                           "save_every")}
    write_json(out / "run_args.json", {
        "schema": "gatedtrain_recipe_cosyvoice_dpo_tts.run_args", "schema_version": "1.0", "profile": a.profile,
        "dry_run": False, "seed": a.seed, "settings": settings, "n_steps": n_steps, "argv": sys.argv[1:],
        "model_dir": a.model_dir, "llm_init": str(llm_file), "llm_init_sha256": sha256(llm_file), "base_ref": base_ref,
        "pairs_file": str(pairs_dir / "pairs.jsonl"), "pairs_sha256": sha256(pairs_dir / "pairs.jsonl"),
        "pairs_meta_sha256": sha256(pairs_dir / "pairs_meta.json"), "n_pairs": len(pairs),
        "candidates_dir": str(cand_dir), "candidates_jsonl_sha256": sha256(cand_dir / "candidates.jsonl"),
        "candidates_meta_sha256": sha256(cand_dir / "candidates_meta.json"),
        "ref_cache": a.ref_cache, "ref_cache_sha256": sha256(Path(a.ref_cache)) if a.ref_cache else None,
        "script_sha256": script_sha, "code": CODE})
    write_env(out)
    t0 = time.time()
    llm = load_llm(model_dir, llm_file, device)
    if a.grad_ckpt:
        llm.llm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm_sha = sha256(llm_file)
    amp_dtype = torch.bfloat16 if a.amp == "bf16" else None
    ref_meta = {"llm_sha256": llm_sha, "pairs_sha256": sha256(pairs_dir / "pairs.jsonl"), "amp": a.amp}
    if a.ref_cache:
        lines = Path(a.ref_cache).read_text().splitlines()
        head = json.loads(lines[0])
        if head != {"meta": ref_meta}:
            raise SystemExit(f"--ref-cache was computed for {head}, not {ref_meta}")
        ref = {json.loads(l)["key"]: json.loads(l)["logp"] for l in lines[1:]}
    else:
        ref = reference_logps(llm, pairs, device, amp_dtype, a.ref_batch)
    (out / "ref_logps.jsonl").write_text(json.dumps({"meta": ref_meta}) + "\n" + "".join(
        json.dumps({"key": k, "logp": v}) + "\n" for k, v in sorted(ref.items())))
    # Sampling-time (fp32 inference) vs teacher-forced reference log-probs of the chosen sequences: a mismatch beyond
    # bf16 noise means the sequence construction differs from inference.
    diffs = [abs(ref[pair_key(p, "chosen")] - p["chosen_sample_logp"]) / len(p["chosen_ex"]["targets"]) for p in pairs]
    ref_check = {"mean_abs_logp_diff_per_token": round(float(np.mean(diffs)), 5),
                 "max_abs_logp_diff_per_token": round(float(np.max(diffs)), 5)}
    print(json.dumps({"ref_vs_sampling": ref_check, "pairs": len(pairs), "load_seconds": round(time.time() - t0, 1)}),
          flush=True)
    base_keys = sorted(torch.load(model_dir / "llm.pt", map_location="cpu", weights_only=True, mmap=True).keys())
    cuts = []

    def save(step: int, pairs_consumed: int) -> None:
        sd = {k: v.detach().to("cpu", torch.float32).clone() for k, v in llm.state_dict().items()}
        if sorted(sd) != base_keys:
            raise SystemExit("state dict keys differ from llm.pt")
        cut = save_cut(out, step, pairs_consumed, sd, base_ref,
                       {"script_sha256": script_sha, "seed": a.seed, "profile": a.profile})
        cuts.append({"step": step, "dir": cut.name, "llm_sha256": sha256(cut / CUT_WEIGHTS)})

    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    last = train(llm, pairs, ref, a, device, out / "train_log.jsonl", save, out / "samples.jsonl")
    import importlib.metadata as md
    meta = {"script_sha256": script_sha, "code": CODE, "args": vars(a), "llm_init": str(llm_file),
            "llm_init_sha256": llm_sha, "model_dir": str(model_dir),
            "candidates": str(cand_dir.resolve()), "candidates_jsonl_sha256": sha256(cand_dir / "candidates.jsonl"),
            "candidates_meta_sha256": sha256(cand_dir / "candidates_meta.json"),
            "items_sha256": metas["candidates_meta"].get("items_sha256"),
            "pairs": str(pairs_dir.resolve()), "pairs_sha256": sha256(pairs_dir / "pairs.jsonl"),
            "pairs_meta_sha256": sha256(pairs_dir / "pairs_meta.json"), "n_pairs": len(pairs),
            "ref_vs_sampling": ref_check, "last_step": last, "llm_dpo_sha256": cuts[-1]["llm_sha256"], "cuts": cuts,
            "profile": a.profile, "seed": a.seed, "python": platform.python_version(),
            "versions": {p: md.version(p) for p in ("torch", "transformers", "numpy")},
            "gpu": torch.cuda.get_device_name(device) if on_gpu else None,
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if on_gpu else None,
            "wall_seconds": round(time.time() - t0, 1)}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
