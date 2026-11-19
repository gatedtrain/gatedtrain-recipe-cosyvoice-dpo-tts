"""make_pairs.py, and the CPU pipeline sample -> (fake) reward -> pairs -> DPO steps (main venv)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import _support
import torch

import make_pairs as mp
import sample_candidates as sc
import train_dpo as td


def cand(item, k, wer, secs=3.0, n=60, stop=6562, err=None, asr="some words"):
    return {"item": item, "k": k, "seed": 100 + k, "error": err, "n_tokens": n, "audio_seconds": secs, "stop": stop,
            "asr_whisper": asr, "wer_whisper_canon": wer, "reward_error": None}


ITEMS = {f"i{j}": {"item": f"i{j}", "category": "number" if j < 3 else "natural_long", "text": "x" * 45}
         for j in range(6)}


class Selection(unittest.TestCase):
    def test_choice_ties_and_margin(self):
        cands = [cand("i0", 0, 0.2), cand("i0", 1, 0.0), cand("i0", 2, 0.0), cand("i0", 3, 0.5), cand("i0", 4, 0.5),
                 cand("i1", 0, 0.1), cand("i1", 1, 0.12),                      # gap 0.02 < 0.05
                 cand("i2", 0, 0.0), cand("i2", 1, 0.0),                       # all zero
                 cand("i3", 0, 0.3), cand("i3", 1, 0.0, stop=None),            # one sane only
                 cand("i4", 0, 0.0), cand("i4", 1, 0.4, secs=30.0),            # duration outlier
                 cand("i4", 2, 0.2), cand("i5", 0, 0.1, err="boom"), cand("i5", 1, 0.0, asr=" "),
                 cand("i5", 2, 0.3, n=0)]
        pairs, st = mp.make_pairs(cands, ITEMS, 0.05, 0.5, 2.5)
        self.assertEqual([(p["item"], p["chosen_k"], p["rejected_k"]) for p in pairs], [("i0", 1, 3), ("i4", 0, 2)])
        self.assertEqual(pairs[0]["margin"], 0.5)
        self.assertEqual(st["item_drops"], {"all_zero": 1, "below_margin": 1, "fewer_than_2_sane": 2})
        self.assertEqual(st["candidate_checks"]["duration"], 1)
        self.assertEqual(st["candidate_checks"]["max_len"], 1)
        self.assertEqual(st["candidate_checks"]["sampling_error"], 1)
        self.assertEqual(st["candidate_checks"]["empty_transcript"], 1)
        self.assertEqual(st["candidate_checks"]["empty"], 1)
        self.assertEqual(st["pairs"], 2)
        self.assertAlmostEqual(st["pair_yield"], 2 / 6, places=4)

    def test_deterministic_under_input_order(self):
        cands = [cand("i0", k, w) for k, w in enumerate([0.3, 0.1, 0.1, 0.3])]
        a, _ = mp.make_pairs(cands, ITEMS, 0.05, 0.5, 2.5)
        b, _ = mp.make_pairs(cands[::-1], ITEMS, 0.05, 0.5, 2.5)
        self.assertEqual(a, b)
        self.assertEqual((a[0]["chosen_k"], a[0]["rejected_k"]), (1, 0))

    def test_category_rates(self):
        cands = [cand(f"i{j}", k, 0.1, secs=(3.0 if j < 3 else 6.0)) for j in range(6) for k in range(8)]
        cps, overall = mp.chars_per_second(cands, ITEMS)
        self.assertAlmostEqual(cps["number"], 15.0)           # 24 candidates: own median
        self.assertAlmostEqual(cps["natural_long"], 7.5)
        cps2, overall2 = mp.chars_per_second(cands[:30], ITEMS)   # natural_long has 6 < 20: overall median
        self.assertEqual(cps2["natural_long"], overall2)

    def test_never_reads_qwen(self):
        for f in ("make_pairs.py", "train_dpo.py", "sample_candidates.py", "eval/score_candidates.py"):
            src = (_support.AREA / f).read_text()
            for bad in ("qwen_asr", "Qwen3ASRModel", "asr_qwen", "wer_qwen"):
                self.assertNotIn(bad, src, f"{f} mentions {bad}")
            self.assertIsNone(re.search(r"items-hard-test|text_split\W+test", src), f)


def write_rewards(root: Path, cand_dir: Path, items_sha: str, wers: dict, reward_asr="whisper-large-v3-turbo"):
    rdir = root / "rewards"
    rdir.mkdir()
    recs = [json.loads(l) for l in (cand_dir / "candidates.jsonl").read_text().splitlines()]
    (rdir / "rewards.jsonl").write_text("".join(json.dumps(
        {"item": r["item"], "k": r["k"], "seed": r["seed"], "error": None, "asr_whisper": "words",
         "wer_whisper_canon": wers[(r["item"], r["k"])]}) + "\n" for r in recs))
    (rdir / "reward_meta.json").write_text(json.dumps({"reward_asr": reward_asr, "items_sha256": items_sha,
                                                       "candidates_sha256": mp.sha256(cand_dir / "candidates.jsonl"),
                                                       "canon_version": "canon-1"}))
    return rdir


class Pipeline(unittest.TestCase):
    def test_sample_reward_pairs_train(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            items = [{"item": f"hard-train:number:{i}", "category": "number", "text": f"we logged {i} parcels today",
                      "prompt_text": "p q r", "prompt_wav": "w.wav"} for i in range(3)]
            idir = root / "items"
            idir.mkdir()
            body = "".join(json.dumps(i, sort_keys=True) + "\n" for i in items)
            (idir / "items.jsonl").write_text(body)
            items_sha = "sha256:" + hashlib.sha256(body.encode()).hexdigest()
            (idir / "item_set_meta.json").write_text(json.dumps({"items_sha256": items_sha, "text_split": "train",
                                                                 "voice_split": "train_clean_100",
                                                                 "n_items": 3}))
            llm = _support.tiny_llm(seed=12)
            with torch.no_grad():                                   # let some samples stop before max_len
                llm.llm_decoder.weight[llm.eos_token] = llm.llm_decoder.weight[3]
            cand_dir = root / "cand"
            sc.run(_support.FakeCV(llm), items, cand_dir, 3, 11, root, torch.manual_seed, log=lambda s: None)
            (cand_dir / "candidates_meta.json").write_text(json.dumps({"text_split": "train", "items_sha256": items_sha,
                                                                       "voice_split": "train_clean_100",
                                                                       "k": 3, "seed": 11}))
            recs = [json.loads(l) for l in (cand_dir / "candidates.jsonl").read_text().splitlines()]
            # Fake rewards: WER grows with k; audio here is 960 samples per token, so durations vary with length.
            wers = {(r["item"], r["k"]): 0.1 * r["k"] for r in recs}
            rdir = write_rewards(root, cand_dir, items_sha, wers)
            r = subprocess.run([sys.executable, str(_support.AREA / "make_pairs.py"), "--candidates", str(cand_dir),
                                "--rewards", str(rdir), "--items", str(idir), "--out", str(root / "pairs"),
                                "--min-ratio", "0.01", "--max-ratio", "100"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            meta = json.loads((root / "pairs" / "pairs_meta.json").read_text())
            n_stop = sum(r["stop"] is not None for r in recs)
            self.assertGreater(n_stop, 1, "fixture should have stopped samples")
            self.assertGreater(meta["stats"]["pairs"], 0)
            pairs, _ = td.load_pairs(cand_dir, root / "pairs")
            for p in pairs:
                self.assertLess(p["chosen_k"], p["rejected_k"])
                self.assertEqual(p["chosen_ex"]["targets"][-1] in llm.stop_token_ids, True)
            ref = td.reference_logps(llm, pairs, "cpu", None, 4)
            a = argparse.Namespace(beta=0.1, lr=1e-4, epochs=1, micro_batch=1, grad_accum=2, warmup_steps=1,
                                   max_steps=2, nll_weight=0.0, clip=1.0, seed=11, amp="none", save_every=0)
            last = td.train(llm, pairs, ref, a, "cpu", root / "log.jsonl")
            self.assertTrue(math.isfinite(last["loss"]))
            # Guards: a reward set from another ASR, or from a test item set, is refused.
            rdir2 = root / "r2"
            rdir.rename(rdir2)
            write_rewards(root, cand_dir, items_sha, wers, reward_asr="qwen3-asr-1.7b")
            with self.assertRaises(SystemExit):
                mp.load(cand_dir, root / "rewards", idir)
            (idir / "item_set_meta.json").write_text(json.dumps({"items_sha256": items_sha, "text_split": "test"}))
            with self.assertRaises(SystemExit):
                mp.load(cand_dir, rdir2, idir)
            (idir / "item_set_meta.json").write_text(json.dumps({"items_sha256": items_sha, "text_split": "train",
                                                                 "voice_split": "dev_clean"}))
            with self.assertRaises(SystemExit):                     # the dev set is never a pair source
                mp.load(cand_dir, rdir2, idir)


if __name__ == "__main__":
    unittest.main()
