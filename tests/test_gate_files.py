"""The published gate files tell one consistent story: registration v2 amends v1 in the scorer pin only, the gate
result applied v2 with the pinned analysis script, and its verdict follows from its own clause numbers."""

from __future__ import annotations

import unittest

import yaml

from _support import EVIDENCE, FIXTURES, GATE, RUNTIME, read_json, read_jsonl
from gatedtrain_recipe_cosyvoice_dpo_tts import pins

# SHA-256 of the gate-time registration files (the copies here have redacted comments, README.md "Results").
REGISTRATION_V1_SHA256 = "sha256:ac51251bf89647f46e2b546d7d7144292667862741e5e55e5e8de51be448032a"
REGISTRATION_V2_SHA256 = "sha256:e61ce5745a93476a947a5160fb20e1527e7e33708507ec5c3c5bd314744b313a"


def load(name: str) -> dict:
    return yaml.safe_load((GATE / name).read_text(encoding="utf-8"))


class TestAmendment(unittest.TestCase):
    def test_v2_changes_only_the_scorer_pin(self):
        v1, v2 = load("registration.yaml"), load("registration-v2.yaml")
        self.assertEqual(v2["amends"], REGISTRATION_V1_SHA256)
        self.assertNotIn("amends", v1)
        self.assertLess(v1["created_at"], v2["created_at"])
        changed = {k for k in set(v1) | set(v2) if v1.get(k) != v2.get(k)}
        self.assertEqual(changed, {"amends", "created_at", "registration_id", "code"})
        code_changed = {k for k in v1["code"] if v1["code"][k] != v2["code"][k]}
        self.assertEqual(code_changed, {"score_tts_sha256"})
        self.assertIs(v2["post_hoc"], False)

    def test_registered_values(self):
        reg = load("registration-v2.yaml")
        quality, guard = reg["clauses"]
        self.assertEqual((quality["type"], quality["metric"], quality["direction"], quality["margin"],
                          quality["alpha"], quality["blocking"]),
                         ("superiority", "wer_qwen_canon", "lower_is_better", 0.0, 0.025, True))
        self.assertEqual((guard["type"], guard["metric"], guard["direction"], guard["margin"], guard["alpha"],
                          guard["blocking"]),
                         ("non_inferiority", "sim_prompt", "higher_is_better", 0.01, 0.025, True))
        self.assertEqual((reg["interval"], reg["cluster_key"], reg["bootstrap"], reg["pairing"]["max_unpaired"],
                          reg["eval_seeds"], reg["eval_distribution"]["n_items"]),
                         ("cluster_bootstrap", "speaker", {"replicates": 10000, "seed": 20261117}, 0, [11], 450))
        self.assertEqual(reg["multiplicity"], {"plan": "all_clauses_must_pass"})


class TestGateResult(unittest.TestCase):
    def setUp(self):
        self.result = read_json(EVIDENCE / "gate-test-v2" / "gate_result.json")
        self.reg = load("registration-v2.yaml")

    def test_applied_registration_and_script(self):
        self.assertEqual(self.result["registration_sha256"], REGISTRATION_V2_SHA256)
        self.assertEqual(self.result["registration_id"], self.reg["registration_id"])
        self.assertEqual(self.result["script_sha256"], pins.SCRIPTS["analyze_gate.py"])
        self.assertEqual((self.result["n_items"], self.result["n_clusters"]), (450, 38))
        self.assertEqual(self.result["descriptive"]["reference_rl"]["pin_problems"], [])

    def test_verdict_follows_from_the_clauses(self):
        by_name = {c["name"]: c for c in self.result["clauses"]}
        q, g = by_name["quality"], by_name["speaker_guard"]
        self.assertLess(q["upper_bound"], -q["margin"])           # superiority, lower is better
        self.assertGreater(g["lower_bound"], -g["margin"])        # non-inferiority, higher is better
        self.assertEqual({c["verdict"] for c in self.result["clauses"]}, {"PASS"})
        self.assertEqual(self.result["verdict"], "PASS")
        self.assertAlmostEqual(q["mean_diff"], -0.03778)
        self.assertAlmostEqual(g["mean_diff"], 0.00121)

    def test_summaries_agree_with_the_gate(self):
        base = read_json(EVIDENCE / "test-base-score-v2" / "summary.json")
        cand = read_json(EVIDENCE / "test-dpo-train-k6-final-score-v2" / "summary.json")
        self.assertEqual((base["errors"], cand["errors"]), (0, 0))
        q = {c["name"]: c for c in self.result["clauses"]}["quality"]
        self.assertAlmostEqual(cand["wer_qwen_canon"] - base["wer_qwen_canon"], q["mean_diff"], delta=1e-3)
        for tag in ("test-base", "test-dpo-train-k6-final", "test-rl"):      # the defect the amendment fixed
            self.assertEqual(read_json(EVIDENCE / f"{tag}-score" / "summary.json")["errors"], 1, tag)

    def test_dev_selection_picked_the_lowest_dev_wer(self):
        tags = ["step26", "step52", "step78", "step104", "final"]
        dev = {t: read_json(EVIDENCE / f"dev-dpo-train-k6-{t}-score" / "summary.json")["wer_qwen_canon"]
               for t in tags}
        self.assertEqual(min(dev, key=dev.get), "final")
        self.assertLess(dev["final"], read_json(EVIDENCE / "dev-base-score" / "summary.json")["wer_qwen_canon"])


class TestScoreRecords(unittest.TestCase):
    """Byte copies of the first rows of two real score_tts.py arms (tests/fixtures/area11)."""

    def test_rows_carry_the_gate_fields(self):
        items = {r["item"]: r for r in read_jsonl(RUNTIME / "items-hard-test" / "items.jsonl")}
        base = read_jsonl(FIXTURES / "test-base-score-v2.scores.head.jsonl")
        cand = read_jsonl(FIXTURES / "test-dpo-train-k6-final-score-v2.scores.head.jsonl")
        self.assertEqual([r["item"] for r in base], [r["item"] for r in cand])
        for row in base + cand:
            self.assertIsNone(row["error"])
            self.assertIn(row["item"], items)
            self.assertEqual(row["speaker"], items[row["item"]]["speaker"])     # the cluster key
            self.assertGreaterEqual(row["wer_qwen_canon"], 0)
            self.assertTrue(-1 <= row["sim_prompt"] <= 1)

    def test_generation_rows_name_their_audio(self):
        for tag in ("test-base", "test-dpo-train-k6-final"):
            for row in read_jsonl(FIXTURES / f"{tag}-gen.gen.head.jsonl"):
                self.assertIsNone(row["error"])
                self.assertEqual(row["wav"], "wav/" + row["item"].replace(":", "__") + ".wav")
