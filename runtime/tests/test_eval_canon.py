"""canon-1 scoring (eval/wer_canon.py, eval/score_tts.py) on real pilot transcripts (eval venv).

The strings are copied verbatim from out/area11-tts/pilot-hard-raw-score-{base,rl}/scores.jsonl and
items-hard-pilot/items.jsonl (64 hard test-split items, dev_clean voices).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import _support  # noqa: F401  (sys.path)
import score_tts
import wer_canon

NORM = wer_canon.load_normalizer(_support.AREA / "models")

# (item, text, ref_text or None, qwen transcript, whisper transcript, stored wer_qwen, stored wer_whisper)
PILOT = {
    "abbr_base": ("hard-test:abbr:63871b8b5bdc", "According to LED, the CPR upgrade ships before 2010.", None,
                  "According to L.E.D., the C.P.R. upgrade ships before 2010.",
                  " According to L.E.D., the CPR upgrade ships before 2010.", 0.6666666666666666, 0.3333333333333333),
    "time_base": ("hard-test:date:d10f2f414260", "Doors opened at 6:10 a.m. on the 12th of August, 1908.", None,
                  "Doors opened at 6:10 a.m. on the 12th of August, 1908.",
                  " Doors opened at 6.10 a.m. on the 12th of August, 1908.", 0.0, 0.15384615384615385),
    "time_omission": ("hard-test:date:e3fb922e34a8", "Doors opened at 9:44 a.m. on the 14th of April, 1995.", None,
                      "Doors opened at 9:44 a.m. on the 14th of April 1995.",
                      " Doors opened at 9.44 a.m. on 14 April 1995.", 0.0, 0.38461538461538464),
    "unit_m3_base": ("hard-test:unit:1cbcf3dc6a9c",
                     "Engineers measured 818 m3 on the first test and 195 cubic meters on the second.",
                     "Engineers measured 818 cubic meters on the first test and 195 cubic meters on the second.",
                     "Engineers measured 818 cubic meters on the first test and 195 cubic meters on the second.",
                     " Engineers measured 818 cubic meters on the first test and 195 cubic meters on the second.",
                     0.125, 0.125),
    "unit_m3_rl": ("hard-test:unit:1cbcf3dc6a9c",
                   "Engineers measured 818 m3 on the first test and 195 cubic meters on the second.",
                   "Engineers measured 818 cubic meters on the first test and 195 cubic meters on the second.",
                   "Engineers measured 80 cubic meter on the first test and 195 cubic meters on the second.",
                   " Engineers measured 80 teak-e-meter on the first test and 195 cubic meters on the second.",
                   0.1875, 0.25),
    "unit_nmi_base": ("hard-test:unit:56284bcf51a9",
                      "Engineers measured 314 nmi on the first test and 372 nautical miles on the second.",
                      "Engineers measured 314 nautical miles on the first test and 372 nautical miles on the second.",
                      "Engineers measured 314 nmi on the first test and 372 nautical miles on the second.",
                      " Engineers measured 314 NME on the first test and 372 nautical miles on the second.",
                      0.0, 0.06666666666666667),
    "decimal": ("hard-test:number:40f33411fa10",
                "Exactly 770,332 visitors came, and attendance rose by 54.74 percent.", None,
                "Exactly 770,332 visitors came, and attendance rose by 54,274 percent.",
                " Exactly 770,332 visitors came, and attendance rose by 54,274%.", 0.1111111111111111,
                0.1111111111111111),
    "money": ("hard-test:money:347e5deb912a", "She paid $733.01 upfront and $594 each month after that.", None,
              "She paid $733 in front and $5.94 each month after that.",
              " She paid $733 in front and $5, $94 each month after that.", 0.4, 0.5),
    "pawn": ("hard-test:money:f5eb7dfdff76", "She paid $2,089.66 upfront and $602 each month after that.", None,
             "She paid two thousand eighty-nine dollars and sixty-six cents up front, and six hundred two pawn.",
             " She paid $2,089.66 up front and $602 pawn", 0.7, 0.6),
}


def item(key):
    it, text, ref, *_ = PILOT[key]
    return {"item": it, "text": text, **({"ref_text": ref} if ref else {})}


def fields(key):
    _, _, _, q, w, *_ = PILOT[key]
    return score_tts.score_texts(item(key), q, w, NORM)


class ArtifactsGone(unittest.TestCase):
    def test_dotted_acronyms(self):
        f = fields("abbr_base")
        self.assertAlmostEqual(f["wer_qwen"], 2 / 3)            # old metric: `l e d` / `c p r` vs `led` / `cpr`
        self.assertEqual(f["wer_qwen_canon"], 0.0)
        self.assertEqual(f["wer_whisper_canon"], 0.0)

    def test_whisper_dotted_time(self):
        f = fields("time_base")
        self.assertGreater(f["wer_whisper"], 0.15)              # `6.10 a m` vs `6 10 a m`
        self.assertEqual(f["wer_whisper_canon"], 0.0)
        self.assertEqual(f["wer_qwen_canon"], 0.0)

    def test_time_artifact_gone_but_omission_counts(self):
        f = fields("time_omission")
        self.assertEqual(f["wer_qwen_canon"], 0.0)
        # Whisper's `9.44` is no longer an error; `on 14 April` for `on the 14th of April` still is.
        self.assertGreater(f["wer_whisper_canon"], 0.0)
        self.assertLess(f["wer_whisper_canon"], f["wer_whisper"])
        self.assertEqual(wer_canon.canonical(" Doors opened at 9.44 a.m. on 14 April 1995.", NORM),
                         "doors opened at 944 am on 14 april 1995")

    def test_spoken_times_match_written_times(self):
        for written, spoken in [("6:10 a.m.", "six ten a m"), ("10:05 p.m.", "ten oh five p.m."),
                                ("1:11 a.m.", "one eleven a.m."), ("9:00 a.m.", "nine a.m."),
                                ("12:53 a.m.", "twelve fifty-three a.m."), ("6:10 a.m.", "6.10 AM"),
                                ("8:57 p.m.", "8 57 p.m.")]:
            with self.subTest(written=written):
                self.assertEqual(wer_canon.wer_canon(f"Doors opened at {written}", f"doors opened at {spoken}",
                                                     NORM), 0.0)

    def test_simple_rule_would_not_be_enough(self):
        # Why canon-1 joins times before the normalizer: it drops the minutes' leading zero and writes `1:` as one.
        self.assertEqual(NORM("10:05 p.m."), "10 5 p m")
        self.assertEqual(NORM("1:11 a.m."), "one 11 a m")
        self.assertEqual(NORM("ten oh five p.m."), "1005 p m")

    def test_unit_ref_text(self):
        f = fields("unit_m3_base")
        self.assertEqual(f["ref_source"], "ref_text")
        self.assertAlmostEqual(f["wer_qwen"], 0.125)            # old: `m 3` vs `cubic meters`
        self.assertEqual(f["wer_qwen_canon"], 0.0)
        self.assertEqual(f["wer_whisper_canon"], 0.0)


class GenuineErrorsStay(unittest.TestCase):
    def test_decimal_vs_thousands(self):
        f = fields("decimal")
        self.assertAlmostEqual(f["wer_qwen_canon"], 1 / 9)
        self.assertAlmostEqual(f["wer_whisper_canon"], 1 / 9)

    def test_money_misread(self):
        f = fields("money")
        self.assertGreater(f["wer_qwen_canon"], 0.0)
        ref = wer_canon.canonical(PILOT["money"][1], NORM).split()
        hyp = wer_canon.canonical(PILOT["money"][3], NORM).split()
        self.assertIn("$594", ref)
        self.assertNotIn("$594", hyp)
        self.assertIn("$5.94", hyp)

    def test_pawn(self):
        f = fields("pawn")
        self.assertGreater(f["wer_qwen_canon"], 0.3)
        self.assertGreater(f["wer_whisper_canon"], 0.3)

    def test_wrong_number_with_right_unit(self):
        f = fields("unit_m3_rl")
        self.assertGreater(f["wer_qwen_canon"], 0.0)            # `80 cubic meter` for `818 cubic meters`
        self.assertGreater(f["wer_whisper_canon"], 0.0)

    def test_abbreviation_left_unread_counts_against_ref_text(self):
        # A transcript that keeps the abbreviation (`314 nmi`) no longer matches the spoken-form reference: strict
        # by design (the listener should hear "nautical miles"); see the report's open question.
        f = fields("unit_nmi_base")
        self.assertEqual(f["wer_qwen"], 0.0)
        self.assertAlmostEqual(f["wer_qwen_canon"], 2 / 16)       # `nmi` for `nautical miles`


class OldFieldsAndSymmetry(unittest.TestCase):
    def test_old_fields_keep_their_meaning(self):
        for key, (_, _, _, q, w, wq, ww) in PILOT.items():
            with self.subTest(key=key):
                f = fields(key)
                self.assertAlmostEqual(f["wer_qwen"], wq, places=12)
                self.assertAlmostEqual(f["wer_whisper"], ww, places=12)

    def test_identity_and_symmetry(self):
        for key, (_, text, ref, q, w, *_rest) in PILOT.items():
            for s in (text, ref, q, w):
                if s:
                    self.assertEqual(wer_canon.wer_canon(s, s, NORM), 0.0)
            self.assertEqual(wer_canon.canonical(q, NORM), wer_canon.post(NORM(wer_canon.pre(q))))

    def test_letter_runs(self):
        self.assertEqual(wer_canon.post("according to l e d the c p r upgrade"), "according to led the cpr upgrade")
        self.assertEqual(wer_canon.post("i am a man"), "i am a man")          # single letters alone stay
        self.assertEqual(wer_canon.post("a b"), "ab")
        self.assertEqual(wer_canon.post("6 10 a m on"), "610 am on")
        self.assertEqual(wer_canon.post("we paid 5 94 dollars"), "we paid 5 94 dollars")   # no am/pm: untouched

    def test_empty_reference_is_nan(self):
        self.assertTrue(math.isnan(wer_canon.wer_canon("", "hello", NORM)))
        self.assertEqual(wer_canon.wer_canon("hello there", "", NORM), 1.0)

    def test_summary_and_mos_hook(self):
        rows = [{"item": "a", "error": None, **fields("abbr_base"), "sim_prompt": 0.8},
                {"item": "b", "error": "boom"}]
        s = score_tts.summarize(rows)
        self.assertEqual(s["canon_version"], wer_canon.CANON_VERSION)
        self.assertEqual(s["deciding_metric"], "wer_qwen_canon")
        self.assertEqual((s["n"], s["errors"], s["wer_qwen_canon"], s["wer_real"]), (2, 1, 0.0, None))
        with self.assertRaises(NotImplementedError):
            score_tts.mos_scores([Path("x.wav")])


class RescoreCli(unittest.TestCase):
    def test_rescore_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "items").mkdir()
            (d / "old").mkdir()
            keys = ["abbr_base", "unit_m3_base", "decimal"]
            (d / "items" / "items.jsonl").write_text("".join(json.dumps({**item(k), "speaker": "1"}) + "\n"
                                                             for k in keys))
            (d / "old" / "scores.jsonl").write_text("".join(json.dumps(
                {"item": PILOT[k][0], "speaker": "1", "error": None, "asr_qwen": PILOT[k][3],
                 "asr_whisper": PILOT[k][4], "wer_qwen": PILOT[k][5], "wer_whisper": PILOT[k][6],
                 "sim_prompt": 0.8, "wer_real": float("nan")}) + "\n" for k in keys))
            r = subprocess.run([sys.executable, str(_support.EVAL / "score_tts.py"), "--rescore", str(d / "old"),
                                "--items", str(d / "items"), "--out", str(d / "new")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            summary = json.loads((d / "new" / "summary.json").read_text())
            self.assertAlmostEqual(summary["wer_qwen_canon"], round((0 + 0 + 1 / 9) / 3, 4))
            self.assertEqual(summary["canon_version"], "canon-1")
            meta = json.loads((d / "new" / "score_meta.json").read_text())
            self.assertTrue(meta["items_sha256"].startswith("sha256:"))
            self.assertTrue(meta["script_sha256"].startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
