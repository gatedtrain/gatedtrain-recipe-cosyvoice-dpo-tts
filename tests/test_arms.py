"""score_tts.py output to gatedtrain arms (`gatedtrain_recipe_cosyvoice_dpo_tts.arms`) and the smoke record (`.smoke`).

The conversion is checked on byte-copied rows of the published run's base arm (tests/fixtures/area11: the first five
items of items-hard-test, gen.jsonl and the v2 scores.jsonl) and on the synthetic evaluations of tests/_tts.py; every
arm written must pass gatedtrain's own arm reader (`gatedtrain.ext.arm`) with the registry row's range and error classes.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from _support import FIXTURES, RUNTIME, read_jsonl
from _tts import SAMPLER, TtsScenario, build_eval, write_json, write_jsonl
from gatedtrain.ext.arm import read_lines, read_meta
from gatedtrain.testing.support import EXIT_OK, describe, file_sha256, manifest_hash
from gatedtrain_recipe_cosyvoice_dpo_tts import arms, pins, smoke
from gatedtrain_recipe_cosyvoice_dpo_tts.rows import EVALUATOR_ROWS, SIM_ID, WER_ID


def check_arm(test: unittest.TestCase, arm: Path, eid: str) -> list[dict]:
    """gatedtrain's reader accepts the arm; before ext-arm-fill only the (null) model hash is missing."""
    row = EVALUATOR_ROWS[eid]
    meta, why = read_meta(arm)
    test.assertEqual(why, ["eval_meta.json checkpoint is not {model_hash, base_model_hash} (MM-2)"])
    test.assertIsNone(meta["checkpoint"]["model_hash"])
    recs, bad = read_lines(arm, seeds=[11], score_range=row["score_range"], error_classes=row["error_classes"])
    test.assertEqual(bad, [])
    test.assertEqual(meta["evaluator_id"], eid)
    test.assertEqual(meta["scorer"], row["scorer"])
    test.assertEqual(meta["sampler"], SAMPLER)
    return recs


class TestPublishedRows(unittest.TestCase):
    """The real score rows of the published base arm convert to valid arms."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-tts-arms-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        items = self.tmp / "items"
        items.mkdir()
        lines = (RUNTIME / "items-hard-test" / "items.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)[:5]
        (items / "items.jsonl").write_text("".join(lines), encoding="utf-8")
        self.items = items
        self.base = self.tmp / "base"
        self.base.mkdir()
        (self.base / "llm.pt").write_bytes(b"PK\x03\x04 stand-in base llm")
        gen, score = self.tmp / "gen", self.tmp / "score"
        gen.mkdir()
        score.mkdir()
        (gen / "gen.jsonl").write_bytes((FIXTURES / "test-base-gen.gen.head.jsonl").read_bytes())
        (score / "scores.jsonl").write_bytes((FIXTURES / "test-base-score-v2.scores.head.jsonl").read_bytes())
        write_json(gen / "gen_meta.json", {"llm_sha256": file_sha256(self.base / "llm.pt"), "n_items": 5,
                                           "items_sha256": file_sha256(items / "items.jsonl"), "seed": 11,
                                           "text_frontend": False, "script_sha256": pins.SCRIPTS["gen_tts.py"]})
        write_json(score / "score_meta.json", {"items_sha256": file_sha256(items / "items.jsonl"),
                                               "gen_jsonl_sha256": file_sha256(gen / "gen.jsonl"),
                                               "script_sha256": pins.SCRIPTS["eval/score_tts.py"],
                                               "wer_canon_sha256": pins.SCRIPTS["eval/wer_canon.py"],
                                               "versions": {"qwen-asr": "0.0.6", "speechbrain": "1.1.2"}})
        self.gen, self.score = gen, score

    def convert(self, **kw):
        args = dict(gen=self.gen, score=self.score, items=self.items, checkpoint=self.base,
                    metrics=["wer_qwen_canon", "sim_prompt"])
        args.update(kw)
        return arms.build(**args)

    def test_converts(self):
        out = self.tmp / "arms"
        arms.write(out, self.convert())
        scores = read_jsonl(self.score / "scores.jsonl")
        for metric, eid in (("wer_qwen_canon", WER_ID), ("sim_prompt", SIM_ID)):
            recs = check_arm(self, out / metric, eid)
            self.assertEqual([r["item"] for r in recs], [s["item"] for s in scores])
            self.assertEqual([r["score"] for r in recs], [s[metric] for s in scores])
            meta = json.loads((out / metric / "eval_meta.json").read_text())
            self.assertEqual(meta["checkpoint"], {"model_hash": None, "base_model_hash": None})
            self.assertEqual(meta["framework"]["version"], {"wer_qwen_canon": "0.0.6", "sim_prompt": "1.1.2"}[metric])
            doc = json.loads((out / metric / recs[0]["output_ref"]["path"]).read_text())
            self.assertEqual((doc["item"], doc["asr_qwen"]), (scores[0]["item"], scores[0]["asr_qwen"]))

    def test_other_weights_refuse(self):
        (self.base / "llm.pt").write_bytes(b"PK\x03\x04 another llm")
        with self.assertRaises(arms.ArmError) as cm:
            self.convert()
        self.assertTrue(any("other weights" in p for p in cm.exception.problems))

    def test_other_item_set_refuses(self):
        (self.items / "items.jsonl").write_text((self.items / "items.jsonl").read_text() + "\n")
        with self.assertRaises(arms.ArmError) as cm:
            self.convert()
        self.assertTrue(any("items_sha256" in p for p in cm.exception.problems))

    def test_unpinned_scorer_refuses(self):
        meta = json.loads((self.score / "score_meta.json").read_text())
        meta["script_sha256"] = pins.SCORE_TTS_V1_SHA256
        write_json(self.score / "score_meta.json", meta)
        with self.assertRaises(arms.ArmError) as cm:
            self.convert()
        self.assertTrue(any("eval/score_tts.py" in p for p in cm.exception.problems))

    def test_existing_arm_refuses(self):
        out = self.tmp / "arms"
        arms.write(out, self.convert())
        with self.assertRaises(arms.ArmError):
            arms.write(out, self.convert())

    def test_cli(self):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            rc = arms.main(["--gen", str(self.gen), "--score", str(self.score), "--items", str(self.items),
                            "--checkpoint", str(self.base), "--out", str(self.tmp / "cli"), "--metric", "sim_prompt"])
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(p.name for p in (self.tmp / "cli").iterdir()), ["sim_prompt"])


class TestErrorsAndFill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-tts-arms2-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = TtsScenario(cls.tmp / "sc", n_items=12)

    def test_error_classes(self):
        sc = self.sc
        ev = build_eval(self.tmp / "err", sc.items_dir, sc.item_rows, llm_sha256=file_sha256(sc.base / "llm.pt"),
                        wer=lambda i: float("nan") if i == 5 else 0.1, sim=lambda i: 0.8, gen_errors=(2,))
        scores = read_jsonl(ev / "score" / "scores.jsonl")
        scores[7] = {**scores[7], "error": "ValueError: decoding failed"}
        write_jsonl(ev / "score" / "scores.jsonl", scores)
        out = self.tmp / "err-arms"
        arms.write(out, arms.build(gen=ev / "gen", score=ev / "score", items=sc.items_dir, checkpoint=sc.base,
                                   metrics=["wer_qwen_canon"]))
        lines = read_jsonl(out / "wer_qwen_canon" / "items.jsonl")
        self.assertEqual(lines[2]["error"]["class"], "gen_error")
        self.assertEqual(lines[5]["error"]["class"], "score_error")              # NaN WER (empty reference)
        self.assertEqual(lines[7]["error"]["class"], "score_error")
        self.assertTrue(all(math.isfinite(x["score"]) for x in lines if x["score"] is not None))
        check_arm(self, out / "wer_qwen_canon", WER_ID)

    def test_fill_and_collect_hashes(self):
        """ext-arm-fill fills the arms' hashes from the base and the cut (MM-18: --base for a cut)."""
        sc = self.sc
        proc = sc.fill(sc.arms["cand"] / "sim_prompt", sc.final_cut, base=sc.base)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        meta = json.loads((sc.arms["cand"] / "sim_prompt" / "eval_meta.json").read_text())
        from gatedtrain_recipe_cosyvoice_dpo_tts.adapter import CosyVoiceDpoAdapter
        self.assertEqual(meta["checkpoint"]["model_hash"],
                         CosyVoiceDpoAdapter().identity(sc.final_cut, sc.base)["model_hash"])
        self.assertEqual(meta["checkpoint"]["base_model_hash"], manifest_hash(sc.base))
        self.assertEqual(meta["item_set_hash"], manifest_hash(sc.items_dir))
        proc = sc.fill(sc.arms["cand"] / "wer_qwen_canon", sc.final_cut)
        self.assertNotEqual(proc.returncode, EXIT_OK)                            # a cut needs --base


class TestSmokeRecord(unittest.TestCase):
    """smoke.py with design revision 18 (R18-8): every clause evaluator's baseline is judged with its own row and
    direction; a score on the metric's best bound (WER 0) is exempt from the one-value rule."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-tts-smoke-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = TtsScenario(cls.tmp / "sc", n_items=30)

    def record(self, wer, sim, name):
        sc = self.sc
        ev = build_eval(self.tmp / name, sc.items_dir, sc.item_rows, llm_sha256=file_sha256(sc.base / "llm.pt"),
                        wer=wer, sim=sim, limit=20)
        arms.write(self.tmp / f"{name}-arms", arms.build(gen=ev / "gen", score=ev / "score", items=sc.items_dir,
                                                        checkpoint=sc.base, metrics=["wer_qwen_canon", "sim_prompt"]))
        return smoke.build(train_out=sc.stage_out, baseline_eval=ev, baseline_arms=self.tmp / f"{name}-arms",
                           base=sc.base, items=sc.items_dir, gpu_log=sc.host_work / "gpu-log.md",
                           host_ram_peak_gb=20.0, swap_peak_gb=0.0, full_steps=130, overhead_s=300)

    def problems(self, rec):
        from gatedtrain.ext.smoke import smoke_problems
        return smoke_problems(rec, adapter="cosyvoice_dpo", score_range=(0, None), binary=False,
                              direction="lower_is_better", item_set_hash=manifest_hash(self.sc.items_dir),
                              evaluators=[(WER_ID, EVALUATOR_ROWS[WER_ID]), (SIM_ID, EVALUATOR_ROWS[SIM_ID])])

    def test_pass(self):
        rec = self.record(lambda i: 0.02 * (i + 1), lambda i: 0.70 + 0.01 * i, "spread")
        self.assertEqual((rec["verdict"], rec["reasons"]), ("PASS", []))
        self.assertEqual(len(rec["baseline_scores"]), 20)
        self.assertEqual(rec["baseline_scores"], rec["baseline_scores_by_evaluator"][WER_ID])
        self.assertEqual(sorted(rec["baseline_scores_by_evaluator"]), sorted([WER_ID, SIM_ID]))
        self.assertAlmostEqual(rec["projected_gpu_h"], round((130 * 1.5 + 300) / 3600, 3))
        self.assertEqual(rec["peak_gpu_mem_gb"], round(15.0 * 2 ** 30 / 1e9, 2))
        self.assertEqual(self.problems(rec), [])

    def test_fourteen_zero_wers_pass(self):
        """The published base's shape: 14 of the first 20 test items at WER 0 (the best bound, exempt), six distinct
        positive WERs (>= 1/4 of the sample and >= 2), and spread speaker similarities."""
        rec = self.record(lambda i: 0.0 if i < 14 else 0.05 * (i - 13), lambda i: 0.75 + 0.007 * i, "zeros")
        self.assertEqual(sum(1 for x in rec["baseline_scores"] if x == 0), 14)
        self.assertEqual((rec["verdict"], rec["reasons"]), ("PASS", []))
        self.assertEqual(self.problems(rec), [])

    def test_too_few_off_bound_scores_fail(self):
        rec = self.record(lambda i: 0.0 if i < 17 else 0.1 * i, lambda i: 0.75 + 0.007 * i, "allzero")
        self.assertEqual(rec["verdict"], "FAIL")
        self.assertIn(f"degenerate_bound ({WER_ID})", rec["reasons"])

    def test_a_degenerate_guard_fails(self):
        rec = self.record(lambda i: 0.02 * (i + 1), lambda i: 0.8, "flatsim")
        self.assertEqual(rec["verdict"], "FAIL")
        self.assertTrue(any(r.endswith(f"({SIM_ID})") for r in rec["reasons"]), rec["reasons"])


if __name__ == "__main__":
    unittest.main()
