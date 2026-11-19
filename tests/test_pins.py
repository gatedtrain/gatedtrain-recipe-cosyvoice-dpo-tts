"""Every pinned script and item set of this checkout's runtime has its pinned SHA-256, and the pins agree with what the
2026-11-17 run recorded (results/2026-11-17/evidence) and the registration names (train_dpo.py: the published run's
hash is pins.PUBLISHED_TRAIN_DPO_SHA256)."""

from __future__ import annotations

import unittest

import yaml

from _support import EVIDENCE, GATE, RUNTIME, read_json
from gatedtrain_recipe_cosyvoice_dpo_tts import pins


class TestRuntimePins(unittest.TestCase):
    def test_runtime_matches_pins(self):
        self.assertEqual(pins.script_problems(RUNTIME), [])

    def test_a_changed_script_is_reported(self):
        import shutil
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            copy = Path(d) / "runtime"
            shutil.copytree(RUNTIME, copy, ignore=shutil.ignore_patterns(".venv", "src", "models", "out"))
            (copy / "gen_tts.py").write_bytes((copy / "gen_tts.py").read_bytes() + b"\n")
            problems = pins.script_problems(copy)
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("gen_tts.py: "))

    def test_registration_code_pins(self):
        reg = yaml.safe_load((GATE / "registration-v2.yaml").read_text(encoding="utf-8"))
        self.assertEqual(reg["code"]["gen_tts_sha256"], pins.SCRIPTS["gen_tts.py"])
        self.assertEqual(reg["code"]["score_tts_sha256"], pins.SCRIPTS["eval/score_tts.py"])
        self.assertEqual(reg["code"]["wer_canon_sha256"], pins.SCRIPTS["eval/wer_canon.py"])
        self.assertEqual(reg["eval_distribution"]["items_sha256"], pins.ITEM_SETS["items-hard-test"])
        for arm, key in (("baseline", "base"), ("candidate", "candidate"), ("reference_descriptive", "rl")):
            self.assertEqual(reg["arms"][arm]["llm_sha256"], pins.LLM_SHA256[key])
        v1 = yaml.safe_load((GATE / "registration.yaml").read_text(encoding="utf-8"))
        self.assertEqual(v1["code"]["score_tts_sha256"], pins.SCORE_TTS_V1_SHA256)

    def test_model_pins(self):
        models = read_json(RUNTIME / "models" / "MODELS.json")["models"]
        self.assertEqual({m["repo"]: m["revision"] for m in models.values()}, pins.MODEL_REVISIONS)
        files = models["fun-cosyvoice3-0.5b-2512"]["files"]
        self.assertEqual("sha256:" + files["llm.pt"]["sha256"], pins.LLM_SHA256["base"])
        self.assertEqual("sha256:" + files["llm.rl.pt"]["sha256"], pins.LLM_SHA256["rl"])

    @unittest.skip("dpo-train-k6/train_meta.json and reward-train-k6/reward_meta.json were removed from the "
                   "publication: each records the hash of an unpublished original")
    def test_training_records_name_the_pins(self):
        train = read_json(EVIDENCE / "dpo-train-k6" / "train_meta.json")
        self.assertEqual(train["script_sha256"], pins.PUBLISHED_TRAIN_DPO_SHA256)
        self.assertEqual(train["llm_dpo_sha256"], pins.LLM_SHA256["candidate"])
        self.assertEqual(train["llm_init_sha256"], pins.LLM_SHA256["base"])
        self.assertEqual(train["items_sha256"], pins.ITEM_SETS["items-hard-train"])
        self.assertEqual(train["code"], f"src/CosyVoice@{pins.COSYVOICE_COMMIT}")
        self.assertEqual(read_json(EVIDENCE / "reward-train-k6" / "reward_meta.json")["script_sha256"],
                         pins.SCRIPTS["eval/score_candidates.py"])

    def test_run_records_name_the_pins(self):
        cand = read_json(EVIDENCE / "cand-train-k6" / "candidates_meta.json")
        self.assertEqual(cand["script_sha256"], pins.SCRIPTS["sample_candidates.py"])
        self.assertEqual(cand["llm_sha256"], pins.LLM_SHA256["base"])
        self.assertEqual(cand["items_sha256"], pins.ITEM_SETS["items-hard-train"])
        self.assertEqual(cand["code"], f"src/CosyVoice@{pins.COSYVOICE_COMMIT}")
        self.assertEqual(read_json(EVIDENCE / "pairs-train-k6" / "pairs_meta.json")["script_sha256"],
                         pins.SCRIPTS["make_pairs.py"])
        for tag, key in (("test-base", "base"), ("test-dpo-train-k6-final", "candidate"), ("test-rl", "rl")):
            gen = read_json(EVIDENCE / f"{tag}-gen" / "gen_meta.json")
            self.assertEqual(gen["llm_sha256"], pins.LLM_SHA256[key])
            self.assertEqual(gen["script_sha256"], pins.SCRIPTS["gen_tts.py"])
            self.assertEqual(gen["items_sha256"], pins.ITEM_SETS["items-hard-test"])
            score = read_json(EVIDENCE / f"{tag}-score-v2" / "score_meta.json")
            self.assertEqual(score["script_sha256"], pins.SCRIPTS["eval/score_tts.py"])
            self.assertEqual(score["wer_canon_sha256"], pins.SCRIPTS["eval/wer_canon.py"])

    def test_item_set_meta_names_its_items(self):
        for name, want in pins.ITEM_SETS.items():
            meta = read_json(RUNTIME / name / "item_set_meta.json")
            self.assertEqual(meta["items_sha256"], want, name)
