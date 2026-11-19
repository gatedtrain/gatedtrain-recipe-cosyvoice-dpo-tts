"""eval/score_candidates.py with a fake batched transcriber (eval venv; no model is loaded)."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import _support
import score_candidates
import score_tts
import wer_canon

NORM = wer_canon.load_normalizer(_support.AREA / "models")
ITEMS = {
    "a": {"item": "a", "text": "Engineers measured 818 m3 on the first test and 195 cubic meters on the second.",
          "ref_text": "Engineers measured 818 cubic meters on the first test and 195 cubic meters on the second."},
    "b": {"item": "b", "text": "According to LED, the CPR upgrade ships before 2010."},
}
HYP = {"a": " Engineers measured 818 cubic meters on the first test and 195 cubic meters on the second.",
       "b": " According to L.E.D., the CPR upgrade ships before 2010."}


class Rewards(unittest.TestCase):
    def test_batches_errors_and_same_metric_as_score_tts(self):
        recs = [{"item": i, "k": k, "seed": 10 * k, "wav": f"wav/{i}{k}.wav", "error": None}
                for i in ("a", "b") for k in range(3)]
        recs[2]["error"] = "RuntimeError: x"
        batches = []

        def transcribe(paths):
            batches.append(len(paths))
            return [HYP[Path(p).name[0]] for p in paths]
        rows = score_candidates.score_records(recs, ITEMS, Path("/c"), transcribe, NORM, 2, log=lambda s: None)
        self.assertEqual(batches, [2, 2, 1])
        self.assertTrue(rows[2]["error"].startswith("sampling error"))
        for r in rows:
            if r["error"] is None:
                self.assertEqual(r["wer_whisper_canon"], 0.0)
                ref = score_tts.score_texts(ITEMS[r["item"]], "unused", r["asr_whisper"], NORM)
                self.assertEqual(r["wer_whisper_canon"], ref["wer_whisper_canon"])
                self.assertEqual(r["wer_whisper"], ref["wer_whisper"])
        self.assertGreater(rows[0]["wer_whisper"], 0.0)                     # old metric still counts `m 3`

    def test_failed_batch_is_retried_one_by_one(self):
        recs = [{"item": "a", "k": k, "seed": k, "wav": f"wav/a{k}.wav", "error": None} for k in range(4)]
        calls = []

        def transcribe(paths):
            calls.append(len(paths))
            if len(paths) == 2 or paths == ["/c/wav/a1.wav"]:   # mixed batch fails; a1 fails alone too
                raise RuntimeError("cannot batch")
            return [HYP["a"]] * len(paths)
        rows = score_candidates.score_records(recs, ITEMS, Path("/c"), transcribe, NORM, 2, log=lambda s: None)
        self.assertEqual(calls, [2, 1, 1, 2, 1, 1])
        self.assertEqual([r["error"] is None for r in rows], [True, False, True, True])
        self.assertTrue(rows[1]["error"].startswith("RuntimeError: cannot batch"))
        self.assertEqual(rows[0]["wer_whisper_canon"], 0.0)

    def test_input_guards(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "cand").mkdir()
            (d / "items").mkdir()
            body = "".join(json.dumps(v) + "\n" for v in ITEMS.values())
            (d / "items" / "items.jsonl").write_text(body)
            h = "sha256:" + hashlib.sha256(body.encode()).hexdigest()
            (d / "cand" / "candidates.jsonl").write_text("")
            (d / "cand" / "candidates_meta.json").write_text(json.dumps({"items_sha256": h}))
            (d / "items" / "item_set_meta.json").write_text(json.dumps({"text_split": "test", "items_sha256": h}))
            with self.assertRaises(SystemExit):
                score_candidates.read_inputs(d / "cand", d / "items")
            (d / "items" / "item_set_meta.json").write_text(json.dumps({"text_split": "train", "items_sha256": h,
                                                                        "voice_split": "dev_clean"}))
            with self.assertRaises(SystemExit):                  # dev set
                score_candidates.read_inputs(d / "cand", d / "items")
            (d / "items" / "item_set_meta.json").write_text(json.dumps({"text_split": "train", "items_sha256": h,
                                                                        "voice_split": "train_clean_100"}))
            (d / "cand" / "candidates_meta.json").write_text(json.dumps({"items_sha256": "sha256:other"}))
            with self.assertRaises(SystemExit):
                score_candidates.read_inputs(d / "cand", d / "items")
            (d / "cand" / "candidates_meta.json").write_text(json.dumps({"items_sha256": h}))
            recs, items, _, _ = score_candidates.read_inputs(d / "cand", d / "items")
            self.assertEqual((recs, sorted(items)), ([], ["a", "b"]))

    def test_reward_asr_is_whisper_only(self):
        src = (_support.EVAL / "score_candidates.py").read_text()
        self.assertEqual(score_candidates.REWARD_ASR, "whisper-large-v3-turbo")
        for bad in ("qwen_asr", "Qwen3ASRModel", "qwen3-asr"):
            self.assertNotIn(bad, src)


if __name__ == "__main__":
    unittest.main()
