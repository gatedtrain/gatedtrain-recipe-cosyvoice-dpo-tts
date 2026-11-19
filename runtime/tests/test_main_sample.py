"""sample_candidates.py with a fake CosyVoice front end around a tiny real CosyVoice3LM (main venv, CPU)."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import _support
import numpy as np
import soundfile as sf
import torch

import sample_candidates as sc
from gen_tts import PROMPT_PREFIX, derive_seed

ITEMS = [{"item": f"hard-train:unit:{i:03d}", "text": f"the route covers {i} km roughly", "prompt_text": "a b c d",
          "prompt_wav": f"unpacked-train_clean_100/p{i % 2}.wav"} for i in range(4)]


def read_dir(out: Path) -> tuple[list[dict], dict]:
    recs = [json.loads(l) for l in (out / "candidates.jsonl").read_text().splitlines()]
    shards = {}
    for p in sorted((out / "tokens").glob("*.npz")):
        with np.load(p) as z:
            shards[p.name] = {k: z[k].copy() for k in z.files}
    return recs, shards


def strip(recs):
    return [{k: v for k, v in r.items() if k != "wall_seconds"} for r in recs]


class Sampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.llm = _support.tiny_llm(seed=7)
        cls.cv = _support.FakeCV(cls.llm)
        cls.out = cls.root / "full"
        cls.counts = sc.run(cls.cv, ITEMS, cls.out, 3, 11, cls.root, torch.manual_seed, log=lambda s: None)
        cls.recs, cls.shards = read_dir(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_records_and_seeds(self):
        self.assertEqual(self.counts, {"skipped": 0, "sampled": 4})
        self.assertEqual(len(self.recs), 12)
        for r in self.recs:
            self.assertIsNone(r["error"])
            self.assertEqual(r["seed"], derive_seed("cand", 11, r["item"], r["k"]))
            self.assertTrue((self.out / r["wav"]).exists())
            self.assertEqual(sf.info(str(self.out / r["wav"])).samplerate, 24000)
        self.assertEqual(len({r["seed"] for r in self.recs}), 12)

    def test_synthesis_call_matches_gen_tts(self):
        for c in self.cv.calls:
            self.assertFalse(c["text_frontend"])
            self.assertFalse(c["stream"])
            self.assertTrue(c["prompt_text"].startswith(PROMPT_PREFIX))
            self.assertTrue(c["prompt_wav"].startswith(str(self.root)))

    def test_shard_contents(self):
        z = self.shards[sc.item_name(ITEMS[0]["item"]) + ".npz"]
        self.assertEqual(z["text_token"].tolist(), _support.toy_tokens(ITEMS[0]["text"]))
        head, _, tail = (PROMPT_PREFIX + ITEMS[0]["prompt_text"]).partition("<|endofprompt|>")
        self.assertEqual(z["prompt_text_token"].tolist(),
                         _support.toy_tokens(head) + [_support.ENDOFPROMPT] + _support.toy_tokens(tail))
        self.assertEqual(len(z["prompt_speech_token"]), 7)
        self.assertEqual(z["llm_embedding"].shape, (192,))
        recs = [r for r in self.recs if r["item"] == ITEMS[0]["item"]]
        for r in recs:
            k = r["k"]
            n_lp = len(z[f"cand{k}_token"]) + (1 if int(z[f"cand{k}_stop"]) >= 0 else 0)
            self.assertEqual(len(z[f"cand{k}_logp"]), n_lp)
            self.assertEqual(int(z[f"cand{k}_seed"]), r["seed"])
            self.assertEqual(r["n_tokens"], len(z[f"cand{k}_token"]))
            self.assertAlmostEqual(r["sum_logp"], float(z[f"cand{k}_logp"].sum()), places=2)

    def test_seeded_and_reproducible(self):
        out = self.root / "again"
        sc.run(_support.FakeCV(self.llm), ITEMS[:2], out, 3, 11, self.root, torch.manual_seed, log=lambda s: None)
        recs, shards = read_dir(out)
        self.assertEqual(strip(recs), strip(self.recs[:6]))
        for name, z in shards.items():
            for key in z:
                np.testing.assert_array_equal(z[key], self.shards[name][key])
        cands = [tuple(self.shards[sc.item_name(ITEMS[0]["item"]) + ".npz"][f"cand{k}_token"]) for k in range(3)]
        self.assertEqual(len(set(cands)), 3)                     # different seeds, different samples

    def test_resume(self):
        out = self.root / "resume"
        sc.run(_support.FakeCV(self.llm), ITEMS[:2], out, 3, 11, self.root, torch.manual_seed, log=lambda s: None)
        # A crash mid-item: a partial record of item 2 and a torn line.
        with (out / "candidates.jsonl").open("a") as f:
            f.write(json.dumps({"item": ITEMS[2]["item"], "k": 0}) + "\n" + '{"item": "torn')
        cv = _support.FakeCV(self.llm)
        counts = sc.run(cv, ITEMS, out, 3, 11, self.root, torch.manual_seed, log=lambda s: None)
        self.assertEqual(counts, {"skipped": 2, "sampled": 2})
        self.assertEqual(len(cv.calls), 6)
        recs, shards = read_dir(out)
        self.assertEqual(strip(recs), strip(self.recs))
        self.assertEqual(sorted(shards), sorted(self.shards))

    def test_check_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "candidates_meta.json"
            cfg = {"k": 6, "seed": 11, "llm": "base", "items_sha256": "sha256:a", "llm_sha256": "sha256:b"}
            sc.check_resume(p, cfg)                              # no meta yet: fine
            p.write_text(json.dumps({**cfg, "other": 1}))
            sc.check_resume(p, cfg)
            with self.assertRaises(SystemExit):
                sc.check_resume(p, {**cfg, "k": 4})

    def test_capture_restores_methods(self):
        llm = _support.tiny_llm(seed=8)
        cap = sc.LLMCapture(llm)
        self.assertIn("inference", vars(llm))
        cap.restore()
        self.assertNotIn("inference", vars(llm))
        self.assertNotIn("sampling_ids", vars(llm))

    def test_flow_token_count(self):
        s = sc.SILENT_TOKENS[0]
        self.assertEqual(sc.flow_token_count([5, s, s, s, s, s, s, s, 6]), 1 + 5 + 1)
        self.assertEqual(sc.flow_token_count([s] * 5 + [9] + [s] * 5), 11)

    def test_read_items_guards(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            body = "".join(json.dumps(i) + "\n" for i in ITEMS)
            (d / "items.jsonl").write_text(body)
            h = "sha256:" + hashlib.sha256(body.encode()).hexdigest()
            (d / "item_set_meta.json").write_text(json.dumps({"items_sha256": h, "text_split": "test", "n_items": 4}))
            with self.assertRaises(SystemExit):
                sc.read_items(d)
            (d / "item_set_meta.json").write_text(json.dumps({"items_sha256": "sha256:0", "text_split": "train"}))
            with self.assertRaises(SystemExit):
                sc.read_items(d)
            (d / "item_set_meta.json").write_text(json.dumps({"items_sha256": h, "text_split": "train",
                                                              "voice_split": "dev_clean", "n_items": 4}))
            with self.assertRaises(SystemExit):                  # dev set: train templates, but not for training
                sc.read_items(d)
            (d / "item_set_meta.json").write_text(json.dumps({"items_sha256": h, "text_split": "train",
                                                              "voice_split": "train_clean_100", "n_items": 4}))
            self.assertEqual(len(sc.read_items(d)[0]), 4)


if __name__ == "__main__":
    unittest.main()
