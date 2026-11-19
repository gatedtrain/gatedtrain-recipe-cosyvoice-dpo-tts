"""analyze_gate.py on synthetic arms (main venv; no model, no test-set data)."""
from __future__ import annotations

import unittest

import numpy as np

import _support  # noqa: F401  (paths)
import analyze_gate

REG = {
    "clauses": [
        {"name": "quality", "type": "superiority", "metric": "wer_qwen_canon", "direction": "lower_is_better",
         "margin": 0.0, "alpha": 0.025, "blocking": True},
        {"name": "speaker_guard", "type": "non_inferiority", "metric": "sim_prompt", "direction": "higher_is_better",
         "margin": 0.01, "alpha": 0.025, "blocking": True}],
    "cluster_key": "speaker", "bootstrap": {"replicates": 2000, "seed": 1},
    "pairing": {"max_unpaired": 0}, "eval_distribution": {"n_items": 120},
}


def arms(wer_shift: float, sim_shift: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    base, cand = {}, {}
    for i in range(120):
        spk = f"s{i % 12}"
        w, s = float(rng.uniform(0, 0.3)), float(rng.uniform(0.75, 0.9))
        base[f"t:cat:{i}"] = {"speaker": spk, "wer_qwen_canon": w, "sim_prompt": s, "error": None}
        cand[f"t:cat:{i}"] = {"speaker": spk, "wer_qwen_canon": max(0.0, w + wer_shift + float(rng.normal(0, 0.02))),
                              "sim_prompt": s + sim_shift + float(rng.normal(0, 0.005)), "error": None}
    return base, cand


class Gate(unittest.TestCase):
    def test_clear_gain_passes(self):
        base, cand = arms(-0.05, 0.0)
        r = analyze_gate.analyze(REG, base, cand, [])
        self.assertEqual(r["verdict"], "PASS", r)
        self.assertEqual(r["n_clusters"], 12)

    def test_no_gain_fails_quality(self):
        base, cand = arms(0.0, 0.0)
        r = analyze_gate.analyze(REG, base, cand, [])
        self.assertEqual([c["verdict"] for c in r["clauses"]], ["FAIL", "PASS"])
        self.assertEqual(r["verdict"], "FAIL")

    def test_speaker_drop_fails_guard(self):
        base, cand = arms(-0.05, -0.03)
        r = analyze_gate.analyze(REG, base, cand, [])
        self.assertEqual([c["verdict"] for c in r["clauses"]], ["PASS", "FAIL"])
        self.assertEqual(r["verdict"], "FAIL")

    def test_error_item_fails(self):
        base, cand = arms(-0.05, 0.0)
        cand["t:cat:3"]["error"] = "RuntimeError"
        del cand["t:cat:3"]["wer_qwen_canon"]                # an error row carries no scores
        self.assertEqual(analyze_gate.analyze(REG, base, cand, [])["verdict"], "FAIL")
        desc = analyze_gate.descriptive(base, cand, ["wer_qwen_canon"])     # reported without the error row
        self.assertEqual(desc["wer_qwen_canon"]["n"], 119)

    def test_missing_item_and_pin_problem_fail(self):
        base, cand = arms(-0.05, 0.0)
        del cand["t:cat:5"]
        self.assertEqual(analyze_gate.analyze(REG, base, cand, [])["verdict"], "FAIL")
        base, cand = arms(-0.05, 0.0)
        self.assertEqual(analyze_gate.analyze(REG, base, cand, ["llm mismatch"])["verdict"], "FAIL")

    def test_cluster_bootstrap_resamples_whole_speakers(self):
        d = np.array([1.0, 1.0, 0.0, 0.0])
        boot = analyze_gate.cluster_bootstrap_means(d, np.array(["a", "a", "b", "b"]), 500, 0)
        self.assertTrue(set(np.round(boot, 6)) <= {0.0, 0.5, 1.0})


if __name__ == "__main__":
    unittest.main()
