"""The runtime pipeline sample_candidates.py -> (rewards) -> make_pairs.py -> train_dpo.py, and the scorer's pure part
(eval/score_tts.py, eval/wer_canon.py), on CPU in the CI environment: numpy-backed stand-ins replace torch,
torchaudio and jiwer, and a scripted LLM replaces CosyVoice3, so what runs is the scripts' own code (candidate
capture and token shards, sanity checks and pair ranking, the DPO schedule, loss, sample log and cuts, the WER field
mapping), and the outputs are read with the adapter as ext-collect reads them.

Why: the runtime's own tests (runtime/tests/test_main_*.py) need torch and CosyVoice's source in the runtime venv, so
CI never ran them and a fault in these parts was found only on the GPU. The stand-ins model only what the scripts call:
`llm.inference(**kw)` yields speech tokens and calls the instance's `sampling_ids` once per token (and once for the
stop token), as Qwen2LM.inference does; `inference_zero_shot` yields `tts_speech` chunks; jiwer.wer is word-level
Levenshtein over whitespace-split words.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import io
import json
import math
import random
import shutil
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from gatedtrain_recipe_cosyvoice_dpo_tts import adapter as adapter_mod
from gatedtrain_recipe_cosyvoice_dpo_tts import launch as launch_mod
from gatedtrain_recipe_cosyvoice_dpo_tts.adapter import CosyVoiceDpoAdapter

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime"
EVIDENCE = REPO / "results" / "2026-11-17" / "evidence"


# ---------------------------------------------------------------------------
# Stand-in modules (numpy-backed torch)
# ---------------------------------------------------------------------------

class T(np.ndarray):
    """numpy array with the tensor methods the scripts call."""

    def detach(self):
        return self

    def to(self, *args, **kw):
        return self

    def cpu(self):
        return self

    def clone(self):
        return self.copy()

    def float(self):
        # torch: .float() of a float tensor is the tensor itself, not a copy
        return self if self.dtype == np.float64 else self.astype(np.float64).view(T)

    def numpy(self):
        return np.asarray(self)

    def numel(self):
        return self.size

    def backward(self):
        pass


def tensor(x, dtype=None, device=None):
    return np.asarray(x, dtype=dtype).view(T)


class NoGrad:
    def __call__(self, fn):
        return fn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class AdamW:
    def __init__(self, params, lr, betas, eps, weight_decay):
        assert (betas, eps, weight_decay) == ((0.9, 0.95), 1e-8, 0.0)
        self.param_groups = [{"lr": lr, "initial_lr": lr}]

    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        pass


class LambdaLR:
    def __init__(self, opt, f):
        self.opt, self.f, self.k = opt, f, 0
        self._apply()

    def _apply(self):
        for g in self.opt.param_groups:
            g["lr"] = g["initial_lr"] * self.f(self.k)

    def step(self):
        self.k += 1
        self._apply()


def torch_save(obj, path):
    with zipfile.ZipFile(path, "w") as z:                  # torch.save writes a zip archive
        z.writestr("archive/data.pkl", json.dumps(sorted(obj)))


def stub_modules(saved_keys: list[str]) -> dict:
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        return m

    functional = mod("torch.nn.functional", logsigmoid=lambda x: (-np.log1p(np.exp(-np.asarray(x)))).view(T))
    nn = mod("torch.nn", functional=functional,
             utils=types.SimpleNamespace(clip_grad_norm_=lambda params, clip: tensor(1.5)))
    torch = mod("torch", tensor=tensor, stack=lambda xs: np.stack(xs).view(T),
                cat=lambda xs, dim: np.concatenate(xs, axis=dim).view(T), zeros=lambda *s: np.zeros(s).view(T),
                no_grad=NoGrad, bfloat16="bfloat16", float32="float32", int32=np.int32, nn=nn,
                optim=types.SimpleNamespace(AdamW=AdamW, lr_scheduler=types.SimpleNamespace(LambdaLR=LambdaLR)),
                device=lambda d: types.SimpleNamespace(type=str(d).split(":")[0]),
                cuda=types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda s: None),
                backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()),
                                               cudnn=types.SimpleNamespace()),
                manual_seed=lambda s: None, save=torch_save,
                load=lambda path, map_location=None, weights_only=True, mmap=False: {k: None for k in saved_keys})
    return {"torch": torch, "torch.nn": nn, "torch.nn.functional": functional,
            "torchaudio": mod("torchaudio", save=lambda path, speech, sr: Path(path).write_bytes(b"RIFF stand-in"))}


def jiwer_wer(r: str, h: str) -> float:
    ref, hyp = r.split(), h.split()
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hyp) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
    return d[len(hyp)] / len(ref)


SAVED_KEYS = ["llm.model.weight", "llm_decoder.weight", "speech_embedding.weight"]
_real_version = importlib.metadata.version


def fake_version(name):
    return {"torch": "2.9.0", "transformers": "4.57.0"}.get(name) or _real_version(name)


@contextlib.contextmanager
def stubbed():
    with mock.patch.dict(sys.modules, {**stub_modules(SAVED_KEYS), "jiwer": types.SimpleNamespace(wer=jiwer_wer)}), \
            mock.patch.object(importlib.metadata, "version", fake_version), \
            contextlib.redirect_stdout(io.StringIO()):
        yield


def load(rel: str):
    path = RUNTIME / rel
    sys.path[:0] = [str(path.parent)]
    try:
        spec = importlib.util.spec_from_file_location(f"cosyvoice_runtime_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        del sys.path[0]


with mock.patch.dict(sys.modules, stub_modules(SAVED_KEYS)):
    sys.modules.pop("gen_tts", None)
    gen_tts = load("gen_tts.py")
    sys.modules["gen_tts"] = gen_tts                    # sample_candidates imports it by name
    sc = load("sample_candidates.py")
    td = load("train_dpo.py")
    sys.modules.pop("gen_tts", None)
mp = load("make_pairs.py")
eval_arm = load("eval_arm.py")
wer_canon = load("eval/wer_canon.py")
with mock.patch.dict(sys.modules, {"wer_canon": wer_canon}):    # score_tts imports it by name (bound at import)
    score_tts = load("eval/score_tts.py")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# A scripted CosyVoice3: tokens from a seeded generator, the stop token after n tokens (or max_len)
# ---------------------------------------------------------------------------

SPEECH_TOKEN_SIZE = 6561
EOS = SPEECH_TOKEN_SIZE + 1


class ScriptedLLM:
    speech_token_size = SPEECH_TOKEN_SIZE
    stop_token_ids = [SPEECH_TOKEN_SIZE + i for i in range(200)]

    def __init__(self):
        self.seed = 0

    def sampling_ids(self, weighted_scores, decoded_tokens, sampling, ignore_eos=True):
        tid = int(np.argmax(weighted_scores))
        weighted_scores[:] = -np.inf                           # the real sampler masks the scores in place
        return tid

    def inference(self, *, text, text_len, prompt_text, prompt_text_len, prompt_speech_token, prompt_speech_token_len,
                  embedding, uuid):
        rng = np.random.default_rng(self.seed)
        n = 3 + self.seed % 5
        max_len = 20 if self.seed % 7 else 2                   # some candidates hit max_len (no stop sampled)
        for step in range(max_len):
            scores = (rng.random(SPEECH_TOKEN_SIZE + 200) - 5.0).view(T)
            want = EOS if step == n else int(rng.integers(0, 50))
            scores[want] = 0.0
            tid = self.sampling_ids(scores, [], None)
            if tid in self.stop_token_ids:
                return
            yield tid


class ScriptedCV:
    sample_rate = 24000

    def __init__(self):
        self.model = types.SimpleNamespace(llm=ScriptedLLM())

    def set_seed(self, seed):
        self.model.llm.seed = seed

    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, stream=False, text_frontend=True):
        assert prompt_text.startswith(gen_tts.PROMPT_PREFIX) and not stream and not text_frontend
        words = tts_text.split()
        ids = lambda xs: tensor([[1000 + len(w) for w in xs]], dtype=np.int32)  # noqa: E731
        toks = list(self.model.llm.inference(text=ids(words), text_len=None, prompt_text=ids(prompt_text.split()),
                                             prompt_text_len=None, prompt_speech_token=tensor([[5, 6, 7]]),
                                             prompt_speech_token_len=None, embedding=np.ones((1, 4)).view(T),
                                             uuid="u"))
        yield {"tts_speech": np.zeros((1, 960 * max(1, len(toks)))).view(T)}


# ---------------------------------------------------------------------------
# Settings pinned to the published run and the launcher
# ---------------------------------------------------------------------------

class TestPins(unittest.TestCase):
    def test_schedule_is_the_adapters(self):
        for n, ep, mb, ga, seed, ms in [(7, 2, 2, 8, 11, None), (50, 2, 2, 8, 11, None), (13, 3, 3, 2, 0, 5),
                                        (1, 1, 1, 1, 4, None), (16, 2, 2, 8, 11, 2)]:
            self.assertEqual(td.schedule(n, ep, mb, ga, seed, ms), adapter_mod.schedule(n, ep, mb, ga, seed, ms))
        steps = td.schedule(10, 2, 2, 2, 11, None)
        self.assertEqual([len(s) for s in steps], [2, 2, 2, 2, 2])
        order = [i for s in steps for m in s for i in m]
        self.assertEqual((sorted(order[:10]), sorted(order[10:])), (list(range(10)), list(range(10))))
        self.assertNotEqual(order[:10], order[10:])                  # each epoch has its own shuffle
        self.assertEqual([len(m) for m in td.schedule(5, 1, 2, 2, 0, None)[-1]], [1])   # the rest is kept

    def test_published_lr_schedule(self):
        # LambdaLR min(1, (s + 1) / warmup), logged after sched.step(): step k logs lr x min(1, (k + 1) / 10).
        log = read_jsonl(EVIDENCE / "dpo-train-k6" / "train_log.jsonl")
        for rec in log[:12]:
            self.assertAlmostEqual(rec["lr"], 1e-6 * min(1.0, (rec["step"] + 1) / 10), places=12)
            self.assertLessEqual(rec["pairs"], 16)                     # micro-batch 2 x grad-accum 8

    def test_published_sampling_and_pairs_settings(self):
        cand = json.loads((EVIDENCE / "cand-train-k6" / "candidates_meta.json").read_text())
        pairs = json.loads((EVIDENCE / "pairs-train-k6" / "pairs_meta.json").read_text())
        self.assertEqual((cand["k"], cand["seed"], cand["text_frontend"], cand["prompt_prefix"]),
                         (6, launch_mod.SEED, False, gen_tts.PROMPT_PREFIX))
        self.assertEqual((pairs["margin"], pairs["min_ratio"], pairs["max_ratio"], pairs["reward_asr"]),
                         (0.05, 0.5, 2.5, "whisper-large-v3-turbo"))

    def test_derive_seed(self):
        s = gen_tts.derive_seed("cand", 11, "item:1", 0)
        self.assertEqual(s, int.from_bytes(__import__("hashlib").sha256(b"cand:11:item:1:0").digest()[:8], "big") >> 33)
        seeds = {gen_tts.derive_seed("cand", 11, f"i{j}", k) for j in range(50) for k in range(6)}
        self.assertEqual(len(seeds), 300)
        self.assertTrue(all(0 <= x < 2 ** 31 for x in seeds))          # fits numpy's and torch's seed ranges
        self.assertNotEqual(gen_tts.derive_seed("tts", 11, "i0"), gen_tts.derive_seed("cand", 11, "i0", 0))

    def test_flow_token_count(self):
        silent = sc.SILENT_TOKENS[0]
        self.assertEqual(sc.flow_token_count([9, 9]), 2)
        self.assertEqual(sc.flow_token_count([silent] * 7 + [9]), 6)   # silent tokens beyond 5 in a row dropped
        self.assertEqual(sc.flow_token_count([silent] * 5 + [9] + [silent] * 5), 11)


# ---------------------------------------------------------------------------
# make_pairs: sanity checks, ranking, margin
# ---------------------------------------------------------------------------

def cand(item, k, wer, secs=3.0, n=60, stop=EOS, err=None, asr="some words"):
    return {"item": item, "k": k, "seed": 100 + k, "error": err, "n_tokens": n, "audio_seconds": secs, "stop": stop,
            "asr_whisper": asr, "wer_whisper_canon": wer, "reward_error": None}


ITEMS = {f"i{j}": {"item": f"i{j}", "category": "number" if j < 3 else "natural_long", "text": "x" * 45}
         for j in range(7)}


class TestMakePairs(unittest.TestCase):
    def test_choice_ties_margin_and_sanity(self):
        cands = [cand("i0", 0, 0.2), cand("i0", 1, 0.0), cand("i0", 2, 0.0), cand("i0", 3, 0.5), cand("i0", 4, 0.5),
                 cand("i1", 0, 0.1), cand("i1", 1, 0.12),                      # gap 0.02 < 0.05
                 cand("i2", 0, 0.0), cand("i2", 1, 0.0),                       # all zero
                 cand("i3", 0, 0.3), cand("i3", 1, 0.0, stop=None),            # one sane only (max_len)
                 cand("i4", 0, 0.0), cand("i4", 1, 0.4, secs=30.0),            # duration outlier
                 cand("i4", 2, 0.2),
                 cand("i5", 0, 0.1, err="boom"), cand("i5", 1, 0.0, asr=" "), cand("i5", 2, 0.3, n=0),
                 cand("i6", 0, 0.05), cand("i6", 1, 0.0)]                      # gap exactly the margin
        pairs, stats = mp.make_pairs(cands, ITEMS, 0.05, 0.5, 2.5)
        got = {p["item"]: (p["chosen_k"], p["rejected_k"]) for p in pairs}
        self.assertEqual(got, {"i0": (1, 3), "i4": (0, 2), "i6": (1, 0)})     # ties go to the lower k
        self.assertEqual(stats["item_drops"], {"all_zero": 1, "below_margin": 1, "fewer_than_2_sane": 2})
        self.assertEqual(stats["candidate_checks"], {"duration": 1, "empty": 1, "empty_transcript": 1, "max_len": 1,
                                                     "ok": 14, "sampling_error": 1})
        p0 = next(p for p in pairs if p["item"] == "i0")
        self.assertEqual((p0["chosen_seed"], p0["rejected_seed"], p0["margin"], p0["n_sane"]), (101, 103, 0.5, 5))

    def test_reward_error_and_non_finite_wer_are_not_sane(self):
        bad = dict(cand("i0", 1, 0.0), reward_error="asr failed")
        self.assertEqual(mp.basic_failure(bad), "reward_error")
        self.assertEqual(mp.basic_failure(cand("i0", 1, float("nan"))), "no_wer")
        self.assertEqual(mp.basic_failure(cand("i0", 1, None)), "no_wer")
        self.assertIsNone(mp.basic_failure(cand("i0", 1, 0.0)))

    def test_rates_per_category_need_twenty_candidates(self):
        items = {"a": {"category": "number", "text": "x" * 30}, "b": {"category": "natural", "text": "x" * 30}}
        cands = [dict(cand("a", k, 0.0, secs=3.0)) for k in range(20)] + [dict(cand("b", k, 0.0, secs=1.0))
                                                                          for k in range(19)]
        cps, overall = mp.chars_per_second(cands, items)
        self.assertEqual(cps["number"], 10.0)
        self.assertEqual(cps["natural"], overall)                       # 19 < 20: the overall median
        self.assertEqual(overall, 10.0)


# ---------------------------------------------------------------------------
# The pipeline: sample -> rewards -> pairs -> DPO, outputs read by the adapter
# ---------------------------------------------------------------------------

def write_item_set(d: Path, n: int, *, text_split="train", voice_split="train_clean_100") -> list[dict]:
    rows = [{"item": f"train:{j}", "category": "number" if j % 2 else "natural_long",
             "text": " ".join(f"w{j}{i}" for i in range(4 + j % 3)), "prompt_text": "a prompt transcript",
             "prompt_wav": f"voices/{j}.wav"} for j in range(n)]
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    d.mkdir(parents=True)
    (d / "items.jsonl").write_text(body, encoding="utf-8")
    (d / "item_set_meta.json").write_text(json.dumps({
        "items_sha256": "sha256:" + __import__("hashlib").sha256(body.encode()).hexdigest(),
        "text_split": text_split, "voice_split": voice_split}))
    return rows


class TestPipeline(unittest.TestCase):
    """8 items x K = 4 candidates; rewards rank candidates by k (k = 0 best); pairs; then train_dpo.py main() with a
    stand-in LLM loader (2 epochs, micro-batch 2, grad-accum 2, cut every 2 steps)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls._tmp.name)
        cls.items_dir = root / "items-hard-train"
        cls.rows = write_item_set(cls.items_dir, 8)
        cls.cand_dir = root / "cand"
        with stubbed():
            items, imeta = sc.read_items(cls.items_dir)
            cv = ScriptedCV()
            cls.counts = sc.run(cv, items, cls.cand_dir, 4, 11, root / "src", cv.set_seed, log=lambda s: None)
        (cls.cand_dir / "candidates_meta.json").write_text(json.dumps({
            "k": 4, "seed": 11, "llm": "base", "items_sha256": imeta["items_sha256"], "llm_sha256": "sha256:x",
            "text_split": "train", "voice_split": "train_clean_100"}))
        cands = read_jsonl(cls.cand_dir / "candidates.jsonl")
        cls.rew_dir = root / "rewards"
        cls.rew_dir.mkdir()
        (cls.rew_dir / "rewards.jsonl").write_text("".join(json.dumps(
            {"item": c["item"], "k": c["k"], "seed": c["seed"], "error": None, "asr_whisper": "heard words",
             "wer_whisper_canon": 0.1 * c["k"]}) + "\n" for c in cands))
        (cls.rew_dir / "reward_meta.json").write_text(json.dumps({
            "items_sha256": imeta["items_sha256"], "reward_asr": "whisper-large-v3-turbo", "canon_version": "canon-1",
            "candidates_sha256": mp.sha256(cls.cand_dir / "candidates.jsonl")}))
        cls.pairs_dir = root / "pairs"
        with mock.patch.object(sys, "argv", ["make_pairs.py", "--candidates", str(cls.cand_dir), "--rewards",
                                             str(cls.rew_dir), "--items", str(cls.items_dir), "--out",
                                             str(cls.pairs_dir)]), contextlib.redirect_stdout(io.StringIO()):
            mp.main()
        cls.model_dir = root / "models" / "fun-cosyvoice3-0.5b-2512"
        cls.model_dir.mkdir(parents=True)
        (cls.model_dir / "llm.pt").write_bytes(b"base llm")
        (cls.model_dir / "cosyvoice3.yaml").write_text("llm: stand-in\n")
        cls.out = root / "dpo"
        cls.run_train(cls.out)
        cls.adapter = CosyVoiceDpoAdapter()

    @classmethod
    def run_train(cls, out, *extra):
        llm = types.SimpleNamespace(parameters=lambda: [types.SimpleNamespace(requires_grad=True)], train=lambda: None, eval=lambda: None,
                                    state_dict=lambda: {k: tensor([1.0]) for k in SAVED_KEYS})

        def token_logps(llm_, examples, device, amp_dtype=None):
            # a stand-in policy: chosen (low k) sequences slightly more likely than the reference cache below
            return [tensor([-0.5] * len(ex["targets"])) for ex in examples]

        def reference_logps(llm_, pairs, device, amp_dtype, batch):
            ref = {}
            for p in pairs:
                ref[td.pair_key(p, "chosen")] = -0.5 * len(p["chosen_ex"]["targets"]) - 0.2
                ref[td.pair_key(p, "rejected")] = -0.5 * len(p["rejected_ex"]["targets"]) + 0.2
            return ref
        argv = ["train_dpo.py", "--model-dir", str(cls.model_dir), "--candidates", str(cls.cand_dir), "--pairs",
                str(cls.pairs_dir), "--out", str(out), "--grad-accum", "2", "--save-every", "2", "--device", "cpu",
                *extra]
        with stubbed(), mock.patch.object(sys, "argv", argv), mock.patch.object(td, "load_llm", lambda *a: llm), \
                mock.patch.object(td, "token_logps", token_logps), \
                mock.patch.object(td, "reference_logps", reference_logps):
            td.main()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_candidates_and_token_shards(self):
        cands = read_jsonl(self.cand_dir / "candidates.jsonl")
        self.assertEqual(self.counts, {"skipped": 0, "sampled": 8})
        self.assertEqual(len(cands), 32)
        for c in cands:
            self.assertIsNone(c["error"], c)
            self.assertEqual(c["seed"], gen_tts.derive_seed("cand", 11, c["item"], c["k"]))
            with np.load(self.cand_dir / "tokens" / td.item_file(c["item"])) as z:
                toks, stop = z[f"cand{c['k']}_token"], int(z[f"cand{c['k']}_stop"])
                self.assertEqual(len(z[f"cand{c['k']}_logp"]), len(toks) + (stop >= 0))
                self.assertEqual(int(z[f"cand{c['k']}_seed"]), c["seed"])
                # the sampled ids' log-probs, read before sampling_ids masks the scores (0.0 in the scripted LLM)
                self.assertEqual(z[f"cand{c['k']}_logp"].tolist(), [0.0] * len(z[f"cand{c['k']}_logp"]))
            self.assertEqual(c["n_tokens"], len(toks))
            self.assertEqual(c["stop"], None if stop < 0 else stop)
            self.assertEqual(c["stop"], None if c["seed"] % 7 == 0 else EOS)
            self.assertEqual(c["audio_seconds"], round(960 * max(1, len(toks)) / 24000, 3))
        self.assertTrue(any(c["stop"] is None for c in cands))

    def test_resume_resamples_a_partial_item(self):
        part = self.root / "cand-partial"
        shutil.copytree(self.cand_dir, part)
        lines = (part / "candidates.jsonl").read_text().splitlines()
        (part / "candidates.jsonl").write_text("".join(l + "\n" for l in lines[:-1]) + lines[-1][:20])   # torn
        with stubbed():
            cv = ScriptedCV()
            items, _ = sc.read_items(self.items_dir)
            got = sc.run(cv, items, part, 4, 11, self.root / "src", cv.set_seed, log=lambda s: None)
        self.assertEqual(got, {"skipped": 7, "sampled": 1})

        def rows(text_lines):
            # every field but wall_seconds (a timing, different on each sampling) must match the first run's records
            out = [json.loads(l) for l in text_lines]
            for r in out:
                del r["wall_seconds"]
            return sorted(json.dumps(r, sort_keys=True) for r in out)

        self.assertEqual(rows((part / "candidates.jsonl").read_text().splitlines()), rows(lines))

    def test_resume_skips_done_items_and_refuses_other_settings(self):
        with stubbed():
            cv = ScriptedCV()
            items, _ = sc.read_items(self.items_dir)
            self.assertEqual(sc.run(cv, items, self.cand_dir, 4, 11, self.root / "src", cv.set_seed,
                                    log=lambda s: None), {"skipped": 8, "sampled": 0})
        with self.assertRaisesRegex(SystemExit, "not resuming"):
            sc.check_resume(self.cand_dir / "candidates_meta.json",
                            {"k": 6, "seed": 11, "llm": "base", "items_sha256": "x", "llm_sha256": "sha256:x"})

    def test_pairs_rank_by_reward_and_feed_training_examples(self):
        pairs = read_jsonl(self.pairs_dir / "pairs.jsonl")
        self.assertTrue(pairs)
        for p in pairs:
            self.assertLess(p["chosen_k"], p["rejected_k"])
            self.assertAlmostEqual(p["margin"], 0.1 * (p["rejected_k"] - p["chosen_k"]), places=6)
        with stubbed():
            loaded, _ = td.load_pairs(self.cand_dir, self.pairs_dir)
        for p in loaded:
            with np.load(self.cand_dir / "tokens" / td.item_file(p["item"])) as z:
                for side in ("chosen", "rejected"):
                    k, ex = p[f"{side}_k"], p[f"{side}_ex"]
                    stop = int(z[f"cand{k}_stop"])
                    self.assertEqual(ex["speech"], z[f"cand{k}_token"].tolist())
                    self.assertEqual(ex["targets"], ex["speech"] + ([stop] if stop >= 0 else []))
                    self.assertEqual(ex["text"], z["prompt_text_token"].tolist() + z["text_token"].tolist())
                    self.assertEqual(ex["prompt_speech"], [5, 6, 7])

    def test_training_outputs_pass_the_adapter(self):
        cuts, incomplete = self.adapter.find_cuts(self.out)
        args = json.loads((self.out / "run_args.json").read_text())
        n_steps = args["n_steps"]
        self.assertEqual((args["n_pairs"], n_steps), (8, 4))           # 2 epochs x 8 pairs / (2 x 2) per step
        want_cuts = sorted({s for s in range(2, n_steps + 1, 2)} | {n_steps})
        self.assertEqual(([s for s, _ in cuts], incomplete), (want_cuts, []))
        self.assertEqual(self.adapter.consumption_problems(self.out, self.pairs_dir, cuts), [])
        self.assertEqual(n_steps, len(td.schedule(args["n_pairs"], 2, 2, 2, 11, None)))
        for _, cut in cuts:
            self.assertEqual(eval_arm.check_cut(cut)["global_step"], json.loads((cut / "cut.json").read_text())
                             ["global_step"])

    def test_max_steps_run_ends_with_a_cut(self):
        out = self.root / "dpo-3"
        self.run_train(out, "--max-steps", "3")
        cuts, _ = self.adapter.find_cuts(out)
        self.assertEqual([s for s, _ in cuts], [2, 3])
        self.assertEqual(json.loads((out / "run_args.json").read_text())["n_steps"], 3)
        self.assertEqual(self.adapter.consumption_problems(out, self.pairs_dir, cuts), [])
        self.assertEqual(json.loads((cuts[-1][1] / "cut.json").read_text())["pairs_consumed"], 12)

    def test_train_log(self):
        log = read_jsonl(self.out / "train_log.jsonl")
        self.assertEqual([r["step"] for r in log], list(range(1, len(log) + 1)))
        for r in log:
            self.assertAlmostEqual(r["lr"], 1e-6 * min(1.0, (r["step"] + 1) / 10), places=12)
            # policy - reference: chosen +0.2, rejected -0.2, beta 0.1: margin 0.04, loss -log sigmoid(0.04)
            self.assertAlmostEqual(r["loss"], math.log1p(math.exp(-0.04)), places=5)
            self.assertEqual(r["reward_acc"], 1.0)
            self.assertAlmostEqual(r["chosen_reward"], 0.02, places=5)
            self.assertAlmostEqual(r["rejected_reward"], -0.02, places=5)
        ref = read_jsonl(self.out / "ref_logps.jsonl")
        self.assertEqual(ref[0]["meta"]["amp"], "bf16")

    def test_run_args_settings_are_the_launchers(self):
        settings = json.loads((self.out / "run_args.json").read_text())["settings"]
        flags = dict(zip(launch_mod.TRAIN_SETTINGS[::2], launch_mod.TRAIN_SETTINGS[1::2]))
        for flag, value in flags.items():
            key = flag[2:].replace("-", "_")
            want = settings[key] if key == "grad_accum" else type(settings[key])(value)
            self.assertEqual(settings[key], want, key)
        self.assertEqual({k: settings[k] for k in ("beta", "lr", "epochs", "micro_batch", "warmup_steps", "clip",
                                                   "seed", "amp")},
                         {"beta": 0.1, "lr": 1e-6, "epochs": 2, "micro_batch": 2, "warmup_steps": 10, "clip": 1.0,
                          "seed": 11, "amp": "bf16"})                    # the defaults are the launched settings

    def test_dpo_loss(self):
        with stubbed():
            same = tensor([-1.0, -2.0])
            loss, st = td.dpo_loss(same, same, same, same, 0.1)
            self.assertAlmostEqual(float(loss), math.log(2), places=6)     # equal log-ratios: log 2
            loss, st = td.dpo_loss(tensor([0.0]), tensor([-1.0]), tensor([-1.0]), tensor([0.0]), 0.5)
            self.assertAlmostEqual(float(loss), math.log1p(math.exp(-1.0)), places=6)
            self.assertEqual((st["reward_acc"], st["margin"]), (1.0, 1.0))
            loss2, _ = td.dpo_loss(tensor([-4.0]), tensor([-1.0]), tensor([-5.0]), tensor([0.0]), 0.5, 1.0,
                                   tensor([4]))
            self.assertAlmostEqual(float(loss2), math.log1p(math.exp(-1.0)) + 1.0, places=6)

    def test_load_pairs_max_len_candidate_has_no_stop_target(self):
        cand, pairs = self.root / "cand-ml", self.root / "pairs-ml"
        (cand / "tokens").mkdir(parents=True)
        (cand / "candidates.jsonl").write_text("{}\n")
        (cand / "candidates_meta.json").write_text(json.dumps({"text_split": "train", "voice_split": "train_clean_100"}))
        np.savez(cand / "tokens" / td.item_file("train:0"), text_token=np.array([1, 2]), prompt_text_token=np.array([3]),
                 prompt_speech_token=np.array([4]), cand0_token=np.array([7, 8]), cand0_stop=np.array(-1),
                 cand0_logp=np.zeros(2), cand1_token=np.array([9]), cand1_stop=np.array(EOS), cand1_logp=np.zeros(2))
        pairs.mkdir()
        (pairs / "pairs.jsonl").write_text(json.dumps({"item": "train:0", "chosen_k": 1, "rejected_k": 0}) + "\n")
        (pairs / "pairs_meta.json").write_text(json.dumps({"candidates_sha256": td.sha256(cand / "candidates.jsonl")}))
        with stubbed():
            (p,), _ = td.load_pairs(cand, pairs)
        self.assertEqual((p["rejected_ex"]["targets"], p["chosen_ex"]["targets"]), ([7, 8], [9, EOS]))
        self.assertEqual((p["chosen_ex"]["text"], p["chosen_ex"]["prompt_speech"]), ([3, 1, 2], [4]))
        line = (pairs / "pairs.jsonl").read_text().splitlines()[0]
        self.assertEqual(p["_row"], adapter_mod.line_row(line))

    def test_load_pairs_refuses_other_sets(self):
        bad = self.root / "cand-dev"
        bad.mkdir()
        (bad / "candidates_meta.json").write_text(json.dumps({"text_split": "train", "voice_split": "dev_clean"}))
        with stubbed(), self.assertRaisesRegex(SystemExit, "train set only"):
            td.load_pairs(bad, self.pairs_dir)


class TestMakePairsLoad(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = self.root = Path(tmp.name)
        write_item_set(root / "items", 2)
        sha = json.loads((root / "items" / "item_set_meta.json").read_text())["items_sha256"]
        (root / "cand").mkdir()
        (root / "cand" / "candidates.jsonl").write_text(json.dumps(cand("train:0", 0, None)) + "\n" +
                                                        json.dumps(cand("train:0", 1, None)) + "\n")
        self.cmeta = {"items_sha256": sha, "text_split": "train", "voice_split": "train_clean_100"}
        self.rmeta = {"items_sha256": sha, "reward_asr": "whisper-large-v3-turbo",
                      "candidates_sha256": mp.sha256(root / "cand" / "candidates.jsonl")}
        (root / "rew").mkdir()
        (root / "rew" / "rewards.jsonl").write_text(json.dumps({"item": "train:0", "k": 0, "seed": 100,
                                                                "asr_whisper": "a", "wer_whisper_canon": 0.0}) + "\n")

    def load(self, cmeta=None, rmeta=None):
        (self.root / "cand" / "candidates_meta.json").write_text(json.dumps(cmeta or self.cmeta))
        (self.root / "rew" / "reward_meta.json").write_text(json.dumps(rmeta or self.rmeta))
        return mp.load(self.root / "cand", self.root / "rew", self.root / "items")

    def test_load(self):
        cands, items, _ = self.load()
        self.assertEqual([(c["item"], c["k"], c["wer_whisper_canon"]) for c in cands], [("train:0", 0, 0.0)])
        for cm, rm, msg in [({**self.cmeta, "voice_split": "dev_clean"}, None, "train set only"),
                            (None, {**self.rmeta, "items_sha256": "sha256:0"}, "do not belong together"),
                            (None, {**self.rmeta, "reward_asr": "qwen3-asr-1.7b"}, "Whisper rewards only"),
                            (None, {**self.rmeta, "candidates_sha256": "sha256:0"}, "another candidates.jsonl")]:
            with self.assertRaisesRegex(SystemExit, msg):
                self.load(cm, rm)
        (self.root / "rew" / "rewards.jsonl").write_text(json.dumps({"item": "train:0", "k": 0, "seed": 1}) + "\n")
        (self.root / "rew" / "rewards.jsonl").write_text(json.dumps({"item": "train:0", "k": 0, "seed": 100,
                                                                     "error": "asr failed"}) + "\n")
        cands, _, _ = self.load()
        self.assertEqual((cands[0]["reward_error"], mp.basic_failure(cands[0])), ("asr failed", "reward_error"))
        (self.root / "rew" / "rewards.jsonl").write_text(json.dumps({"item": "train:0", "k": 0, "seed": 1}) + "\n")
        with self.assertRaisesRegex(SystemExit, "seed mismatch"):
            self.load()


# ---------------------------------------------------------------------------
# score_tts / wer_canon: which transcript and which reference go into which field
# ---------------------------------------------------------------------------

class TestScoreTexts(unittest.TestCase):
    norm = staticmethod(lambda s: " ".join(s.lower().replace(".", " ").replace(",", "").split()))

    def test_field_mapping(self):
        item = {"item": "i", "text": "Take 5 mg daily", "ref_text": "take five milligrams daily"}
        with stubbed():
            got = score_tts.score_texts(item, "take five milligrams daily", "take 5 mg daily", self.norm)
        self.assertEqual(got["ref_source"], "ref_text")
        self.assertEqual(got["wer_qwen_canon"], 0.0)                    # deciding: Qwen vs ref_text
        self.assertEqual(got["wer_whisper_canon"], 0.5)                 # reported: Whisper vs ref_text
        self.assertEqual(got["wer_qwen"], 0.5)                          # pilot metric: always against text
        self.assertEqual(got["wer_whisper"], 0.0)
        with stubbed():
            plain = score_tts.score_texts({"text": "go now"}, "go now", "go", self.norm)
        self.assertEqual((plain["ref_source"], plain["wer_whisper_canon"]), ("text", 0.5))

    def test_canonicalization(self):
        n = self.norm
        self.assertEqual(wer_canon.canonical("at 9:00 a.m.", n), "at 9 am")
        self.assertEqual(wer_canon.canonical("at 6:10", n), "at 610")
        self.assertEqual(wer_canon.canonical("6.10 a.m.", n), "610 am")
        self.assertEqual(wer_canon.canonical("54.74", n), "54 74")      # a decimal stays two numbers here
        self.assertEqual(wer_canon.canonical("the L E D light", n), "the led light")
        self.assertEqual(wer_canon.post("6 10 a m"), "610 am")
        self.assertEqual(wer_canon.post("i a"), "ia")                   # documented limit
        with stubbed():
            self.assertTrue(math.isnan(wer_canon.wer_canon("...", "x", n)))

    def test_summary_and_rescore(self):
        rows = [{"item": "a", "error": None, "wer_qwen_canon": 0.2, "sim_prompt": 0.5},
                {"item": "b", "error": None, "wer_qwen_canon": float("nan"), "sim_prompt": 0.7},
                {"item": "c", "error": "generation error", "wer_qwen_canon": 9.0}]
        s = score_tts.summarize(rows)
        self.assertEqual((s["n"], s["errors"], s["wer_qwen_canon"], s["sim_prompt"], s["deciding_metric"]),
                         (3, 1, 0.2, 0.6, "wer_qwen_canon"))
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)
            old = [{"item": "a", "error": None, "asr_qwen": "go now", "asr_whisper": "go", "wer_qwen_canon": 0.9,
                    "wer_real": 0.3, "sim_prompt": 0.5},
                   {"item": "b", "error": "boom", "wer_qwen_canon": None}]
            (src / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in old))
            items = {"a": {"item": "a", "text": "go now"}, "b": {"item": "b", "text": "x"}}
            with stubbed():
                new = score_tts.rescore(src, items, self.norm)
        self.assertEqual((new[0]["wer_qwen_canon"], new[0]["wer_whisper_canon"], new[0]["wer_real"],
                          new[0]["sim_prompt"]), (0.0, 0.5, 0.3, 0.5))
        self.assertEqual(new[1], {"item": "b", "error": "boom"})


class TestEvalArmPlan(unittest.TestCase):
    def test_plan_uses_the_published_arguments(self):
        a = types.SimpleNamespace(llm="base", item_set="items-hard-test", out="/o", seed=11, limit=None)
        (gen, score), rec = eval_arm.plan(a)
        self.assertEqual(gen[gen.index("--seed") + 1], "11")
        self.assertIn("--no-real", score)
        self.assertEqual(score[score.index("--gen") + 1], "/o/gen")
        self.assertEqual(rec["items_sha256"], eval_arm.sha256(RUNTIME / "items-hard-test" / "items.jsonl"))
        with self.assertRaisesRegex(SystemExit, "is not one of"):
            eval_arm.plan(types.SimpleNamespace(**{**vars(a), "item_set": "items-hard-train"}))


class TestModuleIsolation(unittest.TestCase):
    def test_no_script_or_stand_in_is_left_in_sys_modules(self):
        # the scripts import gen_tts and wer_canon by name; this file registers them (and the stand-ins) only while
        # it loads or runs the scripts, so no other test module, and no later import, finds this file's copies
        for name in ("gen_tts", "wer_canon"):
            self.assertFalse(name in sys.modules, f"{name} left in sys.modules")
        for name in ("torch", "torchaudio", "jiwer"):
            if name in sys.modules:
                self.assertIsNotNone(getattr(sys.modules[name], "__file__", None), f"stand-in {name} left behind")


if __name__ == "__main__":
    unittest.main()
