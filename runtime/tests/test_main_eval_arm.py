"""eval_arm.py (main venv, CPU): the two commands of an arm, the cut check and the speaker-cache check; nothing is run.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _support  # noqa: F401  (paths; no GPU)

import eval_arm as ea


def write_cut(d: Path, step: int = 26) -> Path:
    d.mkdir(parents=True)
    (d / "llm.pt").write_bytes(b"PK\x03\x04 weights")
    (d / "base_ref.json").write_text("{}\n")
    (d / "cut.json").write_text(json.dumps({"global_step": step, "deployable": ["llm.pt"],
                                            "files": {n: ea.sha256(d / n) for n in ("llm.pt", "base_ref.json")}}))
    return d


class Plan(unittest.TestCase):
    def ns(self, **kw):
        base = dict(llm="base", item_set="items-hard-test", out="/out", seed=11, limit=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_base_commands(self):
        (gen, score), rec = ea.plan(self.ns())
        self.assertEqual(gen[0], sys.executable)
        self.assertEqual(gen[1:], [str(ea.HERE / "gen_tts.py"), "--model-dir", str(ea.MODEL_DIR), "--items",
                                   str(ea.HERE / "items-hard-test"), "--out", "/out/gen", "--llm", "base", "--seed", "11"])
        self.assertEqual(score, [str(ea.EVAL_PYTHON), str(ea.HERE / "eval" / "score_tts.py"), "--gen", "/out/gen",
                                 "--items", str(ea.HERE / "items-hard-test"), "--out", "/out/score", "--no-real"])
        self.assertEqual(rec["llm_file"], str(ea.MODEL_DIR / "llm.pt"))
        self.assertEqual(rec["items_sha256"], ea.sha256(ea.HERE / "items-hard-test" / "items.jsonl"))
        self.assertIsNone(rec["cut"])

    def test_cut_and_limit(self):
        with tempfile.TemporaryDirectory() as d:
            cut = write_cut(Path(d) / "checkpoint-26")
            (gen, _), rec = ea.plan(self.ns(llm=str(cut), limit=20, item_set="items-hard-dev"))
            self.assertEqual(gen[gen.index("--llm") + 1], str(cut / "llm.pt"))
            self.assertEqual(gen[-2:], ["--limit", "20"])
            self.assertEqual(rec["cut"], {"cut_json_sha256": ea.sha256(cut / "cut.json"), "global_step": 26})

    def test_refusals(self):
        with self.assertRaises(SystemExit):
            ea.plan(self.ns(item_set="items-smoke"))
        with tempfile.TemporaryDirectory() as d:
            cut = write_cut(Path(d) / "checkpoint-26")
            (cut / "llm.pt").write_bytes(b"PK\x03\x04 changed")
            with self.assertRaises(SystemExit):
                ea.plan(self.ns(llm=str(cut)))
            cut2 = write_cut(Path(d) / "checkpoint-52")
            (cut2 / "extra").write_text("x")
            with self.assertRaises(SystemExit):
                ea.plan(self.ns(llm=str(cut2)))

    def test_missing_speaker_cache_stops_first(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(ea, "HERE", Path(d)), self.assertRaises(SystemExit) as cm:
                ea.check_ecapa_cache()
            self.assertIn("ecapa-cache", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
