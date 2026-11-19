"""The pre-registered dev selection (`gatedtrain_recipe_cosyvoice_dpo_tts.select`)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from _tts import TtsScenario, write_json
from gatedtrain_recipe_cosyvoice_dpo_tts import select
from gatedtrain_recipe_cosyvoice_dpo_tts.arms import sha256


class TestSelect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-tts-select-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = TtsScenario(cls.tmp / "sc", n_items=12)            # cuts 2, 4, 6

    def dev(self, name: str, step: int | None, wer: float, errors: int = 0, item_set: str = "items-hard-dev") -> Path:
        d = self.tmp / name
        cut = None if step is None else {"cut_json_sha256": sha256(self.sc.stage_out / f"checkpoint-{step}" / "cut.json"),
                                         "global_step": step}
        write_json(d / "eval_run.json", {"item_set": item_set, "limit": None, "cut": cut})
        write_json(d / "score" / "summary.json", {"wer_qwen_canon": wer, "errors": errors, "n": 150, "sim_prompt": 0.83})
        return d

    def run_select(self, wers: dict[int, float], errors: dict[int, int] | None = None, tag: str = "a") -> dict:
        dev = {s: self.dev(f"{tag}-dev{s}", s, w, (errors or {}).get(s, 0)) for s, w in wers.items()}
        return select.select(self.sc.stage_out, dev, self.dev(f"{tag}-base", None, 0.0668))

    def test_lowest_mean_and_last(self):
        rec = self.run_select({2: 0.044, 4: 0.041, 6: 0.033}, tag="last")
        self.assertEqual((rec["selected_step"], rec["last_step"]), (6, 6))
        self.assertEqual(rec["action"], "collect the run as it is")
        self.assertEqual(rec["base"]["wer_qwen_canon"], 0.0668)

    def test_earlier_cut_means_retraining(self):
        rec = self.run_select({2: 0.044, 4: 0.030, 6: 0.033}, tag="early")
        self.assertEqual(rec["selected_step"], 4)
        self.assertIn("--max-steps 4", rec["action"])

    def test_ties_go_to_the_later_step_and_errors_exclude(self):
        self.assertEqual(self.run_select({2: 0.03, 4: 0.03, 6: 0.05}, tag="tie")["selected_step"], 4)
        self.assertEqual(self.run_select({2: 0.04, 4: 0.01, 6: 0.05}, {4: 1}, tag="err")["selected_step"], 2)

    def test_arms_must_cover_the_cuts_on_dev(self):
        with self.assertRaises(ValueError):
            select.select(self.sc.stage_out, {2: self.dev("x2", 2, 0.1), 6: self.dev("x6", 6, 0.1)})
        with self.assertRaises(ValueError):
            select.select(self.sc.stage_out, {2: self.dev("y2", 2, 0.1), 4: self.dev("y4", 4, 0.1),
                                              6: self.dev("y6", 6, 0.1, item_set="items-hard-test")})
        with self.assertRaises(ValueError):                         # an arm of another cut
            select.select(self.sc.stage_out, {2: self.dev("z2", 4, 0.1), 4: self.dev("z4", 4, 0.1),
                                              6: self.dev("z6", 6, 0.1)})

    def test_cli_writes_once(self):
        dev = [f"{s}={self.dev(f'c-dev{s}', s, 0.1 - s / 100)}" for s in (2, 4, 6)]
        out = self.tmp / "selection.json"
        args = ["--run", str(self.sc.stage_out), *sum((["--dev", x] for x in dev), []), "--out", str(out)]
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(select.main(args), 0)
            self.assertEqual(json.loads(out.read_text())["selected_step"], 6)
            self.assertEqual(select.main(args), 3)


if __name__ == "__main__":
    unittest.main()
