"""make_hard_items.py: train/test presets, disjointness, ref_text, prompt rule (main venv; stdlib + soundfile).

Runs on a fake LibriTTS-R tree (silent wavs) in a temp dir; one test also compares the hard-text generator with the
pilot item set on disk (read only).
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import _support
import make_hard_items as mhi

SCRIPT = _support.AREA / "make_hard_items.py"
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec "
         "romeo sierra tango uniform victor whiskey xray yankee zulu").split()


def fake_split(src: Path, split: str, speakers: list[str], seed: int) -> None:
    """Per speaker: 6 utterances, durations 3-12 s, 5-75 words (some natural-long candidates)."""
    rng = random.Random(f"{split}:{seed}")
    root = mhi.split_root(src, split)
    for spk in speakers:
        d = root / spk / "100"
        d.mkdir(parents=True)
        for i, (secs, words) in enumerate([(3.0, 8), (5.0, 12), (7.5, 20), (9.5, 18), (11.0, 40), (12.0, 75)]):
            words += rng.randint(-3, 3)
            uid = f"{spk}_100_{i:06d}_000000"
            sf.write(str(d / f"{uid}.wav"), np.zeros(int(secs * 1000), dtype=np.float32), 1000)
            text = " ".join(rng.choice(WORDS) for _ in range(words)) + "."
            (d / f"{uid}.normalized.txt").write_text(text.capitalize(), encoding="utf-8")


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = Path(cls.tmp.name) / "data-source"
        fake_split(cls.src, "train_clean_100", ["19", "26", "27", "32", "39"], 1)
        fake_split(cls.src, "test_clean", ["121", "237", "260"], 2)
        fake_split(cls.src, "dev_clean", ["84", "174"], 3)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def build(self, name, *args, ok=True):
        out = Path(self.tmp.name) / name
        r = subprocess.run([sys.executable, str(SCRIPT), "--data-source", str(self.src), "--seed", "11",
                            "--out", str(out), *args], capture_output=True, text=True)
        if ok:
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            return [json.loads(l) for l in (out / "items.jsonl").read_text().splitlines()], \
                json.loads((out / "item_set_meta.json").read_text())
        self.assertNotEqual(r.returncode, 0)
        return r.stderr


class Presets(Fixture):
    def test_train_and_test_builds(self):
        train, tmeta = self.build("train", "--preset", "train", "--n-hard", "60", "--n-natural", "5")
        test, smeta = self.build("test", "--preset", "test", "--n-hard", "30", "--n-natural", "3",
                                 "--check-against", str(Path(self.tmp.name) / "train"))
        self.assertEqual((len(train), len(test)), (65, 33))
        self.assertEqual((tmeta["text_split"], tmeta["voice_split"], tmeta["natural_split"]),
                         ("train", "train_clean_100", "train_clean_100"))
        self.assertEqual((smeta["text_split"], smeta["voice_split"], smeta["natural_split"]),
                         ("test", "test_clean", "test_clean"))
        self.assertEqual([c["items_sha256"] for c in smeta["checked_disjoint_from"]], [tmeta["items_sha256"]])
        self.assertTrue(tmeta["script_sha256"].startswith("sha256:"))
        self.assertEqual({r["speaker"] for r in train}, {"19", "26", "27", "32", "39"})
        self.assertEqual({r["speaker"] for r in test}, {"121", "237", "260"})
        self.assertFalse({r["text"] for r in train} & {r["text"] for r in test})
        self.assertTrue(all(r["prompt_wav"].startswith("unpacked-train_clean_100/") for r in train))
        self.assertTrue(all(r["prompt_wav"].startswith("unpacked-test_clean/") for r in test))
        for rows in (train, test):
            nat = [r for r in rows if r["category"] == "natural_long"]
            self.assertTrue(all(30 <= len(r["text"].split()) <= 70 for r in nat))
            self.assertFalse({r["source_id"] for r in nat} & {r["prompt_id"] for r in rows})
            self.assertLessEqual(sum(r["category"] == "homograph" for r in rows), 4)
            self.assertLessEqual(sum(r["category"] == "twister" for r in rows), 4)
        # Overlap is caught: the test set checked against itself.
        err = self.build("test-again", "--preset", "test", "--n-hard", "30", "--n-natural", "3",
                         "--check-against", str(Path(self.tmp.name) / "test"), ok=False)
        self.assertIn("share", err)

    def test_dev_preset(self):
        tdir, sdir = Path(self.tmp.name) / "dtrain", Path(self.tmp.name) / "dtest"
        self.build("dtrain", "--preset", "train", "--n-hard", "60", "--n-natural", "5")
        self.build("dtest", "--preset", "test", "--n-hard", "30", "--n-natural", "3", "--check-against", str(tdir))
        dev, meta = self.build("dev", "--preset", "dev", "--n-hard", "40", "--n-natural", "2",
                               "--check-against", str(tdir), str(sdir))
        again, meta2 = self.build("dev2", "--preset", "dev", "--n-hard", "40", "--n-natural", "2")
        self.assertEqual(dev, again)
        self.assertEqual(meta["items_sha256"], meta2["items_sha256"])
        self.assertEqual((meta["text_split"], meta["voice_split"], meta["natural_split"], meta["set"]),
                         ("train", "dev_clean", "dev_clean", "dev"))
        self.assertEqual(len(meta["checked_disjoint_from"]), 2)
        self.assertEqual(len(dev), 42)
        self.assertTrue(all(r["item"].startswith("hard-dev:") for r in dev))
        self.assertEqual({r["speaker"] for r in dev}, {"84", "174"})
        self.assertTrue(all(r["prompt_wav"].startswith("unpacked-dev_clean/") for r in dev))
        # Train templates and lexicon, but never a text of the train preset's hard set (same seed, full size).
        full_train = mhi.hard_texts("train", mhi.PRESETS["train"]["n_hard"], 11)
        train_texts = {t[f] for t in full_train for f in ("text", "ref_text") if f in t}
        hard = [r for r in dev if r["category"] != "natural_long"]
        self.assertFalse({r["text"] for r in hard} & train_texts)
        self.assertFalse({r.get("ref_text") for r in hard} & train_texts)
        self.assertFalse({r["category"] for r in hard} & {"homograph", "twister"})   # all four used by train
        test_lex = mhi._lexicon_entries("test")
        for r in hard:
            self.assertFalse([w for w in test_lex if w in r["text"]], r["text"])
        self.assertTrue(any("ref_text" in r for r in hard))
        # Overlap with an existing set is caught (dev checked against itself).
        self.assertIn("share", self.build("dev3", "--preset", "dev", "--n-hard", "40", "--n-natural", "2",
                                          "--check-against", str(Path(self.tmp.name) / "dev"), ok=False))

    def test_dev_split_rules(self):
        self.assertIn("fixes", self.build("bd1", "--preset", "dev", "--voice-split", "train_clean_100", ok=False))
        self.assertIn("train_clean_100", self.build("bd2", "--text-split", "train", "--voice-split", "dev_clean",
                                                    "--natural-split", "dev_clean", "--n-hard", "5",
                                                    "--n-natural", "0", ok=False))

    def test_deterministic(self):
        a, ma = self.build("det-a", "--preset", "test", "--n-hard", "25", "--n-natural", "2")
        b, mb = self.build("det-b", "--preset", "test", "--n-hard", "25", "--n-natural", "2")
        self.assertEqual(a, b)
        self.assertEqual(ma["items_sha256"], mb["items_sha256"])

    def test_split_rules(self):
        self.assertIn("train_clean_100", self.build("bad1", "--text-split", "train", "--voice-split", "test_clean",
                                                    "--natural-split", "train_clean_100", "--n-hard", "5",
                                                    "--n-natural", "0", ok=False))
        self.assertIn("never", self.build("bad2", "--text-split", "test", "--voice-split", "test_clean",
                                          "--natural-split", "train_clean_100", "--n-hard", "5", "--n-natural", "0",
                                          ok=False))
        self.assertIn("fixes", self.build("bad3", "--preset", "train", "--voice-split", "test_clean", ok=False))

    def test_speaker_overlap_between_splits_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "ds"
            fake_split(src, "train_clean_100", ["19", "121"], 4)
            fake_split(src, "test_clean", ["121"], 5)
            r = subprocess.run([sys.executable, str(SCRIPT), "--data-source", str(src), "--seed", "1", "--out",
                                str(Path(d) / "o"), "--preset", "test", "--n-hard", "5", "--n-natural", "0"],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("share speakers", r.stderr)

    def test_prompt_rule(self):
        rows, _ = self.build("prompt", "--preset", "test", "--n-hard", "9", "--n-natural", "0")
        voices = mhi.speakers("test_clean", self.src)
        for r in rows:
            cands = [u for u in voices[r["speaker"]] if 4.0 <= u["seconds"] <= 10.0]
            best = max(cands, key=lambda u: (u["words"], u["id"]))
            self.assertEqual((r["prompt_id"], r["prompt_text"]), (best["id"], best["text"]))


class RefText(unittest.TestCase):
    def test_unit_ref_text(self):
        for split in ("train", "test"):
            texts = mhi.hard_texts(split, 200, 11)
            units = [t for t in texts if t["category"] == "unit"]
            self.assertTrue(units)
            for t in texts:
                if t["category"] != "unit":
                    self.assertNotIn("ref_text", t)
            for t in units:
                full, abbr = next(u for u in mhi.LEX[split]["units"] if re.search(rf"\d {re.escape(u[1])}\b", t["text"]))
                pat = rf"(\d) {re.escape(abbr)}\b"
                self.assertEqual(t["ref_text"], re.sub(pat, rf"\1 {full}", t["text"], count=1))
                self.assertIsNone(re.search(pat, t["ref_text"]))

    def test_ordinals(self):
        self.assertEqual([mhi.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 28)],
                         ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "28th"])
        n = 0
        for t in mhi.hard_texts("test", 300, 11):
            for m in re.finditer(r"\b(\d+)(st|nd|rd|th)\b", t["text"]):
                self.assertEqual(m.group(0), mhi.ordinal(int(m.group(1))), t["text"])
                n += 1
        self.assertGreater(n, 20)

    def test_pilot_hard_texts_reproduced(self):
        pilot = [json.loads(l) for l in (_support.AREA / "items-hard-pilot" / "items.jsonl").read_text().splitlines()]
        hard = [r for r in pilot if r["category"] != "natural_long"]
        made = mhi.hard_texts("test", len(hard), 7)
        fixed = {"Doors opened at 1:11 a.m. on the 2th of April, 1933.": "Doors opened at 1:11 a.m. on the 2nd of April, 1933.",
                 "Doors opened at 8:57 a.m. on the 21th of June, 2026.": "Doors opened at 8:57 a.m. on the 21st of June, 2026."}
        self.assertEqual([fixed.get(r["text"], r["text"]) for r in hard], [t["text"] for t in made])
        self.assertEqual([r["category"] for r in hard], [t["category"] for t in made])

    def test_lexicons_disjoint(self):
        self.assertFalse(mhi._lexicon_entries("train") & mhi._lexicon_entries("test"))

    def test_check_disjoint(self):
        a = [{"speaker": "1", "prompt_id": "p1", "text": "x", "ref_text": "y", "source_id": None}]
        mhi.check_disjoint(a, [{"speaker": "2", "prompt_id": "p2", "text": "z", "source_id": None}])
        with self.assertRaises(SystemExit):
            mhi.check_disjoint(a, [{"speaker": "2", "prompt_id": "p2", "text": "y", "source_id": None}])


if __name__ == "__main__":
    unittest.main()
