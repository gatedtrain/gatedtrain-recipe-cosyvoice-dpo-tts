"""The rerun's gate files (launch/gate/registration.yaml, spec.template.yaml) under gatedtrain design revision 18: the
registration passes gatedtrain's checks and carries the published registration-v2 values; the rendered spec loads and
validates in the layout launch/tts.sh gate-freeze writes."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _support import GATE, REPO, RUNTIME
from gatedtrain import gates
from gatedtrain.ext import registry
from gatedtrain.registration import check_against_gate, check_contents, clause_list, pilot_for
from gatedtrain.testing.support import EXIT_OK, describe, file_sha256, manifest_hash, run_cli
from gatedtrain_recipe_cosyvoice_dpo_tts.rows import SAMPLER, SIM_ID, WER_ID

LAUNCH_GATE = REPO / "launch" / "gate"


def registration() -> dict:
    return yaml.safe_load((LAUNCH_GATE / "registration.yaml").read_text(encoding="utf-8"))


class TestRegistration(unittest.TestCase):
    def test_check_contents(self):
        self.assertEqual(check_contents(registration()), [])

    def test_not_frozen(self):
        self.assertNotIn("created_at", registration())

    def test_values_of_registration_v2(self):
        reg, v2 = registration(), yaml.safe_load((GATE / "registration-v2.yaml").read_text(encoding="utf-8"))
        quality, guard = clause_list(reg)
        for mine, old in ((quality, v2["clauses"][0]), (guard, v2["clauses"][1])):
            self.assertEqual(mine["name"], old["name"])
            for k in ("type", "metric", "direction", "margin", "alpha", "power", "blocking"):
                self.assertEqual(mine[k], old[k], (mine["name"], k))
            self.assertEqual((mine["cluster_key"], mine["bootstrap_variant"], mine["unit"], mine["eval_seeds"]),
                             (v2["cluster_key"], "percentile", v2["unit"], v2["eval_seeds"]))
            self.assertEqual(reg["sigma_pilot"][mine["name"]]["sigma"], old["sigma_plan"])
        self.assertEqual(quality["planning_effect"], v2["clauses"][0]["planning_effect"])
        self.assertEqual(reg["bootstrap"], v2["bootstrap"])
        self.assertEqual(reg["pairing"], v2["pairing"])
        self.assertEqual(reg["multiplicity"], {"plan": "all_clauses_must_pass"})
        self.assertEqual([c["evaluator"] for c in clause_list(reg)], [WER_ID, SIM_ID])

    def test_hashes_and_sampler(self):
        reg = registration()
        self.assertEqual(reg["eval_distribution"]["item_set_hash"], manifest_hash(RUNTIME / "items-hard-test"))
        self.assertEqual({e["id"]: e["manifest_hash"] for e in reg["evaluators"]},
                         {ev: registry.evaluator_identity_hash(ev) for ev in (WER_ID, SIM_ID)})
        self.assertEqual(reg["sampler"], SAMPLER)

    def test_power(self):
        """gatedtrain's planning: 283 items for quality, 71 for the speaker guard (registration-v2's comment said 63)."""
        reg = registration()
        plans = [gates.plan(c, pilot_for(reg, c), n_available=450) for c in clause_list(reg)]
        self.assertEqual([p["n_required"] for p in plans], [283, 71])
        self.assertEqual([p["underpowered"] for p in plans], [False, False])


class TestSpec(unittest.TestCase):
    """The layout of launch/tts.sh gate-init and gate-freeze: <out>/gate/{registration.yaml (with created_at),
    spec.yaml, trainer.yaml, runtime -> the runtime}, <out>/inputs/pairs."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-tts-gate-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        rt = self.tmp / "rt"
        shutil.copytree(RUNTIME / "items-hard-test", rt / "items-hard-test")
        base = rt / "models" / "fun-cosyvoice3-0.5b-2512"
        base.mkdir(parents=True)
        (base / "llm.pt").write_bytes(b"PK\x03\x04 stand-in")
        pairs = self.tmp / "out" / "inputs" / "pairs"
        pairs.mkdir(parents=True)
        (pairs / "pairs.jsonl").write_text('{"item": "hard-train:x:1", "chosen_k": 1, "rejected_k": 0}\n')
        gate = self.tmp / "out" / "gate"
        gate.mkdir()
        text = (LAUNCH_GATE / "registration.yaml").read_text(encoding="utf-8")
        (gate / "registration.yaml").write_text(text + 'created_at: "2026-11-18T09:00:00Z"\n', encoding="utf-8")
        (gate / "runtime").symlink_to(rt)
        (gate / "trainer.yaml").write_text("backend: cosyvoice_dpo\nprofile: full\n")
        h = file_sha256(gate / "registration.yaml").split(":", 1)[1]
        spec = (LAUNCH_GATE / "spec.template.yaml").read_text(encoding="utf-8").replace("{registration_sha256}", h)
        (gate / "spec.yaml").write_text(spec, encoding="utf-8")
        self.gate, self.rt, self.pairs = gate, rt, pairs

    def test_spec_loads_and_binds(self):
        from gatedtrain.spec import load_spec
        wf = load_spec(self.gate / "spec.yaml")
        self.assertEqual(wf.evaluators, [WER_ID, SIM_ID])
        self.assertEqual(wf.path_of("model"), (self.rt / "models" / "fun-cosyvoice3-0.5b-2512").resolve())
        self.assertEqual(wf.path_of("data"), self.pairs.resolve())
        self.assertEqual(wf.path_of("items"), (self.rt / "items-hard-test").resolve())
        gate = yaml.safe_load((self.gate / "spec.yaml").read_text())["stages"][1]
        self.assertEqual(check_against_gate(wf.registration, gate, "stages[1]"), [])

    def test_validate(self):
        proc = run_cli(["validate", str(self.gate / "spec.yaml")], cwd=self.gate)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_an_edited_registration_refuses(self):
        p = self.gate / "registration.yaml"
        p.write_text(p.read_text().replace("margin: 0.01", "margin: 0.02"))
        proc = run_cli(["validate", str(self.gate / "spec.yaml")], cwd=self.gate)
        self.assertNotEqual(proc.returncode, EXIT_OK)


if __name__ == "__main__":
    unittest.main()
