"""Adapter `cosyvoice_dpo` (area "TTS"): the cut layout, identity, consumption, checks and a whole ext-collect.

Sources: MULTIMODAL-SPEC.md MM-1 to MM-39 (MM-11, MM-17, MM-18, MM-20, MM-21, MM-23, MM-27, MM-29, MM-33); the runtime
(`runtime/train_dpo.py` `save_cut`, `schedule`, `train`, `main`; `runtime/eval_arm.py`). Helpers: `tests/_tts.py`.
The gate here has one clause (wer_qwen_canon superiority): the gatedtrain these tests run against registers one clause per
gate; the recipe's two-clause registration is written for design revision 18.
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from _tts import ADAPTER, BASE_NAME, RT_C, TtsScenario, load_schedule, read_jsonl, sha, write_jsonl
from gatedtrain.testing.support import EXIT_OK, EXIT_REFUSED, describe, file_sha256, manifest_hash, parse_errors, parse_run_dir
from gatedtrain_recipe_cosyvoice_dpo_tts import adapter as A
from gatedtrain_recipe_cosyvoice_dpo_tts import launch as L


def ad() -> A.CosyVoiceDpoAdapter:
    return A.CosyVoiceDpoAdapter()


class Case(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-tts-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.n = 0

    def fresh(self, **kw) -> TtsScenario:
        type(self).n += 1
        return TtsScenario(self.tmp / f"sc-{type(self).n}", **kw)

    def runs(self) -> Path:
        type(self).n += 1
        return self.tmp / f"runs-{type(self).n}"

    def refused(self, sc: TtsScenario, rule: str, needle: str = "") -> list[dict]:
        proc = sc.collect(self.runs())
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and needle in e["message"]]
        self.assertTrue(hits, f"no {rule} error containing {needle!r}:\n{proc.stderr[-3000:]}")
        return errs


class TestSchedule(unittest.TestCase):
    def test_adapter_schedule_is_train_dpo_schedule(self):
        """The adapter re-derives the trainer's pair order from the standard library; the trainer's own function
        (compiled from runtime/train_dpo.py) gives the same steps for any setting."""
        theirs = load_schedule()
        rnd = random.Random(7)
        for _ in range(40):
            args = (rnd.randint(1, 40), rnd.randint(1, 3), rnd.randint(1, 4), rnd.randint(1, 9), rnd.randint(0, 99),
                    rnd.choice([None, rnd.randint(1, 30)]))
            self.assertEqual(A.schedule(*args), theirs(*args), args)

    def test_published_run_shape(self):
        """1,034 pairs, 2 epochs, micro-batch 2 x accumulation 8: 130 steps, the last of 2 micro-batches (4 pairs)."""
        steps = A.schedule(1034, 2, 2, 8, 11, None)
        self.assertEqual(len(steps), 130)
        self.assertEqual([len(m) for m in steps[-1]], [2, 2])
        self.assertEqual(sum(len(m) for st in steps for m in st), 2068)


class TestCutsAndIdentity(Case):
    def test_cut_completeness(self):
        sc = self.fresh(n_items=12)
        a = ad()
        complete, incomplete = a.find_cuts(sc.stage_out)
        self.assertEqual([s for s, _ in complete], [2, 4, 6])
        self.assertEqual(incomplete, [])
        cut = sc.final_cut
        self.assertIsNone(a.model_dir_problem(cut))
        self.assertIsNone(a.model_dir_problem(sc.base))
        (sc.stage_out / ".checkpoint-8.partial").mkdir()
        self.assertEqual([s for s, _ in a.checkpoint_dirs(sc.stage_out)], [2, 4, 6])
        (cut / "extra.bin").write_bytes(b"x")
        self.assertIn("does not list", a.cut_problem(cut, 6))
        (cut / "extra.bin").unlink()
        (cut / "llm.pt").write_bytes(b"not a zip")
        self.assertIn("SHA-256", a.cut_problem(cut, 6))
        sc.reseal(cut)
        self.assertIn("zip archive", a.cut_problem(cut, 6))
        self.assertIn("global_step", a.cut_problem(sc.cuts[0], 3))
        (sc.cuts[0] / "cut.json").unlink()
        self.assertIn("cut.json", a.cut_problem(sc.cuts[0], 2))
        self.assertIn("missing", a.base_dir_problem(sc.stage_out))

    def test_identity_is_the_deployed_directory(self):
        """MM-17: a cut's model hash is the manifest hash of the base directory with its llm.pt replaced by the cut's
        (here built as a real copy); the base's is its own manifest hash."""
        sc = self.fresh(n_items=12)
        a = ad()
        deployed = self.tmp / "deployed"
        shutil.copytree(sc.base, deployed)
        shutil.copyfile(sc.final_cut / "llm.pt", deployed / "llm.pt")
        ident = a.identity(sc.final_cut, sc.base)
        self.assertEqual(ident["model_hash"], manifest_hash(deployed))
        self.assertEqual(ident["cut_hash"], manifest_hash(sc.final_cut))
        self.assertEqual(ident["components"], ["llm.pt"])
        self.assertEqual(ident["base_ref"]["manifest_hash"], manifest_hash(sc.base))
        self.assertEqual(a.identity(sc.base)["model_hash"], manifest_hash(sc.base))
        self.assertTrue(a.needs_base(sc.final_cut))
        self.assertFalse(a.needs_base(sc.base))
        with self.assertRaises(ValueError):
            a.identity(sc.final_cut)
        self.assertNotEqual(a.identity(sc.cuts[0], sc.base)["model_hash"], ident["model_hash"])


class TestArgv(unittest.TestCase):
    PY = f"{RT_C}/.venv/bin/python"

    def train(self, *extra, profile="full"):
        return [self.PY, f"{RT_C}/train_dpo.py", "--model-dir", f"{RT_C}/models/{BASE_NAME}", "--candidates",
                "/inputs/cand", "--pairs", "/inputs/pairs", "--out", "/out", "--profile", profile, *extra]

    def evaluate(self, *extra):
        return [self.PY, f"{RT_C}/eval_arm.py", "--llm", "/inputs/cut", "--item-set", "items-hard-test", "--out",
                "/out", *extra]

    def test_train_argv(self):
        a = ad()
        self.assertEqual(a.argv_problems(self.train(), profile="full"), [])
        self.assertEqual(a.argv_problems(self.train(profile="smoke"), profile="smoke"), [])
        self.assertTrue(any("launch profile" in p for p in a.argv_problems(self.train(profile="smoke"), profile="full")))
        self.assertTrue(any("--llm" in p for p in a.argv_problems(self.train("--llm", "/x/llm.pt"), profile="full")))
        bad_base = self.train()
        bad_base[3] = "/inputs/other/model"
        self.assertTrue(any("--model-dir" in p for p in a.argv_problems(bad_base, profile="full")))
        moved = self.train()
        moved[1] = "/elsewhere/train_dpo.py"
        self.assertTrue(any("runtime's scripts" in p for p in a.argv_problems(moved, profile="full")))

    def test_eval_argv(self):
        a = ad()
        self.assertEqual(a.argv_problems(self.evaluate(), profile="full"), [])
        self.assertEqual(a.argv_problems(self.evaluate("--limit", "20"), profile="smoke"), [])
        self.assertTrue(any("--limit" in p for p in a.argv_problems(self.evaluate("--limit", "20"), profile="full")))
        argv = self.evaluate()
        argv[5] = "items-smoke"
        self.assertTrue(any("--item-set" in p for p in a.argv_problems(argv, profile="full")))
        argv = self.evaluate()
        argv[3] = "/inputs/other"
        self.assertTrue(any("--llm" in p for p in a.argv_problems(argv, profile="full")))

    def test_published_settings_pass(self):
        argv = [self.PY, f"{RT_C}/train_dpo.py", "--model-dir", f"{RT_C}/{L.BASE_REL}", "--candidates", "/inputs/cand",
                "--pairs", "/inputs/pairs", "--ref-cache", "/inputs/ref/ref_logps.jsonl", "--out", "/out", "--profile",
                "full", *L.TRAIN_SETTINGS, "--save-every", "26"]
        self.assertEqual(ad().argv_problems(argv, profile="full"), [])


class TestConsumption(Case):
    def test_ledger_rows(self):
        sc = self.fresh(n_items=12)
        a = ad()
        cuts = a.find_cuts(sc.stage_out)[0]
        self.assertEqual(a.consumption_problems(sc.stage_out, sc.data, cuts), [])
        cons = a.consumption(sc.stage_out, sc.data, cuts, run_id="r1")
        lines = [x for x in (sc.data / "pairs.jsonl").read_text().splitlines() if x]
        self.assertEqual(len(cons["rows"]), 2 * len(lines))               # two epochs, every pair twice
        self.assertEqual(sorted(r["data_ref"]["row"] for r in cons["rows"]), sorted([sha(x.encode()) for x in lines] * 2))
        self.assertEqual({r["data_ref"]["shard"] for r in cons["rows"]}, {"pairs.jsonl"})
        self.assertEqual(len({r["sample_id"] for r in cons["rows"]}), len(cons["rows"]))
        self.assertEqual((cons["precision"], cons["derivation"]), ("row", "observed"))

    def test_log_must_be_the_schedule(self):
        sc = self.fresh(n_items=12)
        a = ad()
        cuts = a.find_cuts(sc.stage_out)[0]
        samples = read_jsonl(sc.stage_out / "samples.jsonl")
        samples[0], samples[1] = samples[1], samples[0]
        write_jsonl(sc.stage_out / "samples.jsonl", samples)
        self.assertTrue(any("differs from train_dpo.py's schedule" in p
                            for p in a.consumption_problems(sc.stage_out, sc.data, cuts)))
        write_jsonl(sc.stage_out / "samples.jsonl", samples[2:])
        self.assertTrue(any("lines, not the" in p for p in a.consumption_problems(sc.stage_out, sc.data, cuts)))

    def test_training_payloads_are_the_applied_items(self):
        import os
        sc = self.fresh(n_items=12)
        a = ad()
        cuts = a.find_cuts(sc.stage_out)[0]
        old = os.environ.get("GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RUNTIME")
        os.environ["GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RUNTIME"] = str(sc.runtime)
        try:
            texts, why = a.training_payloads(sc.stage_out, sc.base, cuts, data=sc.data)
        finally:
            if old is None:
                del os.environ["GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RUNTIME"]
            else:
                os.environ["GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RUNTIME"] = old
        self.assertIsNone(why)
        self.assertEqual(texts, {r["text"] for r in sc.train_rows[:12]})


class TestCollect(Case):
    def test_collect_passes(self):
        sc = self.fresh()
        sc.fill_all()
        proc = sc.collect(self.runs())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        run = parse_run_dir(proc.stdout)
        report = json.loads((run / "report" / "gate_report.json").read_text())
        self.assertEqual(report["status"], "PASS")
        a = ad()
        self.assertEqual(report["candidate_model_hash"], a.identity(sc.final_cut, sc.base)["model_hash"])
        self.assertEqual(report["baseline_model_hash"], manifest_hash(sc.base))
        self.assertEqual(report["independence_tier"], ["T3", "T3"])
        self.assertEqual(report["multiplicity"], "all_clauses_must_pass")
        clauses = {c["name"]: c for c in report["clauses"]}
        self.assertEqual({n: (c["status"], c["evaluator"], c["hypothesis_type"]) for n, c in clauses.items()},
                         {"quality": ("PASS", "tts.wer_qwen_canon@0.1", "superiority"),
                          "speaker_guard": ("PASS", "tts.sim_prompt@0.1", "non_inferiority")})
        for c in clauses.values():
            self.assertEqual((c["interval"]["bootstrap"]["cluster_key"], c["interval"]["bootstrap"]["variant"],
                              c["interval"]["bootstrap"]["n_clusters"]), ("speaker", "percentile", 12))
        self.assertEqual(sorted(p.name for p in (run / "eval" / "candidate").iterdir()), ["quality", "speaker_guard"])
        self.assertIn("wer_whisper_canon", report["construct_alignment"]["reward"])
        ledger = pq.read_table(run / "ledger" / "consumption.parquet").to_pylist()
        self.assertEqual(len(ledger), 24)
        edges = json.loads((run / "lineage" / "edges.json").read_text())
        compose = [x for x in edges["activities"] if x["kind"] == "lora_compose"]
        self.assertEqual(len(compose), 3)                                  # one per cut (base + llm.pt)
        lock = json.loads((run / "lockfile.json").read_text())
        self.assertEqual(lock["backend"]["trainer"]["framework"]["name"], "torch")
        self.assertTrue(any(n.startswith("quality: Category ") for n in report["notes"]))
        self.assertTrue(any(n.startswith("speaker_guard: Category ") for n in report["notes"]))

    def test_a_failing_guard_fails_the_gate(self):
        """all_clauses_must_pass: the quality clause passes, the speaker guard does not (0.05 lower similarity)."""
        from _tts import base_sim, build_eval, cand_wer
        from gatedtrain_recipe_cosyvoice_dpo_tts import arms
        sc = self.fresh()
        ev = build_eval(sc.root / "eval" / "cand-low", sc.items_dir, sc.item_rows,
                        llm_sha256=file_sha256(sc.final_cut / "llm.pt"), wer=cand_wer,
                        sim=lambda i: round(base_sim(i) - 0.05 + 0.003 * (i % 3), 4))
        arms.write(sc.root / "arms" / "cand-low", arms.build(gen=ev / "gen", score=ev / "score", items=sc.items_dir,
                                                             checkpoint=sc.final_cut, metrics=["sim_prompt"]))
        sc.candidate_sim = sc.root / "arms" / "cand-low" / "sim_prompt"
        sc.fill_all()
        self.assertEqual(sc.fill(sc.candidate_sim, sc.final_cut, base=sc.base).returncode, EXIT_OK)
        proc = sc.collect(self.runs())
        self.assertEqual(proc.returncode, 1, describe(proc))                  # EXIT_GATE_NOT_PASS
        report = json.loads((parse_run_dir(proc.stdout) / "report" / "gate_report.json").read_text())
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual([(c["name"], c["status"]) for c in report["clauses"]],
                         [("quality", "PASS"), ("speaker_guard", "FAIL")])

    def test_edited_log_refuses(self):
        sc = self.fresh()
        sc.fill_all()
        samples = read_jsonl(sc.stage_out / "samples.jsonl")
        samples[3]["pair_line"] = (samples[3]["pair_line"] + 1) % 12
        write_jsonl(sc.stage_out / "samples.jsonl", samples)
        self.refused(sc, "EXT-CONSUMPTION", "schedule")

    def test_seed_mismatch_refuses(self):
        sc = self.fresh()
        sc.fill_all()
        sc.rewrite_launch(training_seed=12)
        self.refused(sc, "EXT-LAUNCH", "training_seed")

    def test_other_base_refuses(self):
        sc = self.fresh()
        sc.fill_all()
        for cut in sc.cuts:
            sc.edit_json(cut / "base_ref.json", lambda d: d["files"].update({"llm.pt": "sha256:" + "0" * 64}))
            sc.reseal(cut)
        self.refused(sc, "EXT-BASE", "base_ref.json files")

    def test_unverified_base_refuses(self):
        sc = self.fresh()
        sc.fill_all()
        sc.edit_json(sc.stage_out / "run_args.json", lambda d: d["base_ref"].update(verified=False))
        self.refused(sc, "EXT-BASE", "verified")

    def test_other_reference_logps_refuse(self):
        sc = self.fresh()
        sc.fill_all()
        lines = (sc.stage_out / "ref_logps.jsonl").read_text().splitlines()
        head = json.loads(lines[0])
        head["meta"]["llm_sha256"] = "sha256:" + "1" * 64
        (sc.stage_out / "ref_logps.jsonl").write_text("\n".join([json.dumps(head)] + lines[1:]) + "\n")
        self.refused(sc, "EXT-DATA", "ref_logps.jsonl")

    def test_other_candidates_refuse(self):
        sc = self.fresh()
        sc.fill_all()
        sc.edit_json(sc.stage_out / "run_args.json", lambda d: d.update(candidates_jsonl_sha256="sha256:" + "2" * 64))
        self.refused(sc, "EXT-DATA", "candidates")

    def test_unfinished_run_refuses(self):
        sc = self.fresh()
        sc.fill_all()
        shutil.rmtree(sc.final_cut)
        self.refused(sc, "EXT-STAGE", "n_steps")

    def test_missing_scheduled_cut_refuses(self):
        sc = self.fresh()
        sc.fill_all()
        shutil.rmtree(sc.cuts[0])
        self.refused(sc, "EXT-STAGE", "scheduled")

    def test_exact_overlap_refuses(self):
        sc = self.fresh(overlap=2)
        sc.fill_all()
        self.refused(sc, "EXT-CONTAMINATION", "training item")

    def test_arm_of_another_item_set_refuses(self):
        sc = self.fresh()
        sc.fill_all()
        out = sc.candidate / "outputs" / "00000.json"
        doc = json.loads(out.read_text())
        doc["text"] = "another text"
        out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        lines = read_jsonl(sc.candidate / "items.jsonl")
        lines[0]["output_ref"]["sha256"] = file_sha256(out)
        write_jsonl(sc.candidate / "items.jsonl", lines)
        self.refused(sc, "EXT-ARM", "registered item")


class TestLaunch(Case):
    """ext-launch accepts the recipe's requests (launch.py) on a host layout of the scenario: the record is written and
    the command printed, nothing is run."""

    def request(self, sc: TtsScenario, kind: str, out: Path, **kw) -> Path:
        dist = sc.runtime / ".venv" / "lib" / "python3.12" / "site-packages" / "torch-2.5.1+cu124.dist-info"
        dist.mkdir(parents=True, exist_ok=True)
        common = dict(label=f"tts-{kind}", output_dir=out, image=f"docker.io/library/x@sha256:{'ab' * 32}",
                      host_work=sc.host_work, gpu_est_h=0.05, runtime=str(sc.runtime), runtime_container=RT_C)
        if kind == "train":
            req = L.train_request(pairs=sc.inputs["pairs"], candidates=sc.inputs["cand"],
                                  ref_cache=sc.inputs["ref"] / "ref_logps.jsonl", **common, **kw)
        else:
            req = L.eval_request(**common, **kw)
        out.mkdir(parents=True)
        path = sc.root / f"{kind}-{out.name}.request.yaml"
        path.write_text(yaml.safe_dump(req, sort_keys=False), encoding="utf-8")
        return path

    def launch(self, sc: TtsScenario, request: Path, profile: str, *extra: str):
        from gatedtrain.testing.support import run_cli
        rec = request.with_suffix(".launch.json")
        args = ["ext-launch", "--adapter", ADAPTER, "--profile", profile, "--request", str(request), "--gpu-log",
                str(sc.host_work / "gpu-log.md"), "--out-record", str(rec), *extra]
        return run_cli(args, cwd=sc.root, env=sc.env()), rec

    def test_train_and_eval_requests_launch(self):
        sc = self.fresh(n_items=12)
        base_out = sc.output_root / "cosyvoice-dpo-tts"
        req = self.request(sc, "train", base_out / "train-smoke", profile="smoke")
        proc, rec = self.launch(sc, req, "smoke", "--spec", str(sc.root / "spec.yaml"))
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        record = json.loads(rec.read_text())
        self.assertEqual(record["script_path"], f"{RT_C}/train_dpo.py")
        self.assertIn("gpu-step.sh", proc.stdout)
        self.assertIn("--max-steps", record["argv"])
        req = self.request(sc, "eval", base_out / "eval-smoke-base", profile="smoke", llm="base",
                           item_set="items-hard-test", limit=20)
        proc, rec = self.launch(sc, req, "smoke")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(json.loads(rec.read_text())["script_path"], f"{RT_C}/eval_arm.py")
        req = self.request(sc, "eval", base_out / "eval-cut", profile="full", llm=str(sc.final_cut),
                           item_set="items-hard-test")
        smoke = sc.smoke
        proc, rec = self.launch(sc, req, "full", "--smoke-record", str(smoke), "--spec", str(sc.root / "spec.yaml"))
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn({"host": str(sc.final_cut), "container": "/inputs/cut", "mode": "ro"},
                      json.loads(rec.read_text())["mounts"])

    def test_common_mounts_the_given_runtimes_interpreter(self):
        """common() derives the interpreter mounts from its `runtime` argument's venv, not from the module's runtime
        (GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RUNTIME or this checkout's, which may have a venv of its own)."""
        root = Path(tempfile.mkdtemp(prefix="gatedtrain-tts-common-"))
        self.addCleanup(shutil.rmtree, root, True)
        rt = root / "runtime"
        tree = root / "python" / "cpython-3.12.16-linux-x86_64-gnu"
        (tree / "bin").mkdir(parents=True)
        (rt / ".venv" / "lib" / "python3.12" / "site-packages" / "torch-2.5.1+cu124.dist-info").mkdir(parents=True)
        (rt / ".venv" / "pyvenv.cfg").write_text(f"home = {tree}/bin\nversion_info = 3.12.16\n")
        req = L.common(label="x", output_dir=root / "out", image="i", host_work=root / "hw", gpu_est_h=0.1,
                       runtime=str(rt))
        self.assertEqual(req["mounts"], [{"host": str(root / "python"), "container": str(root / "python"), "mode": "ro"},
                                         {"host": str(rt), "container": str(rt), "mode": "ro"}])
        bare = root / "bare"
        (bare / ".venv" / "lib" / "python3.12" / "site-packages" / "torch-2.5.1+cu124.dist-info").mkdir(parents=True)
        req = L.common(label="x", output_dir=root / "out", image="i", host_work=root / "hw", gpu_est_h=0.1,
                       runtime=str(bare))
        self.assertEqual(req["mounts"], [{"host": str(bare), "container": str(bare), "mode": "ro"}])

    def test_full_launch_checks_every_clause_evaluator(self):
        """R18-8: with --spec, the smoke record needs a passing baseline of each clause's evaluator."""
        from gatedtrain.testing.ext import write_canonical
        from _tts import smoke_record
        sc = self.fresh(n_items=12)
        bad = smoke_record(base_model="sha256:" + "0" * 64, item_set=sc.item_set_hash)
        del bad["baseline_scores_by_evaluator"]
        path = write_canonical(sc.root / "smoke-old.json", bad)
        req = self.request(sc, "eval", sc.output_root / "cosyvoice-dpo-tts" / "eval-x", profile="full", llm="base",
                           item_set="items-hard-test")
        proc, _ = self.launch(sc, req, "full", "--smoke-record", str(path), "--spec", str(sc.root / "spec.yaml"))
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn("baseline_scores_by_evaluator", proc.stderr)

    def test_train_request_without_the_pairs_mount_refuses_with_spec(self):
        sc = self.fresh(n_items=12)
        out = sc.output_root / "cosyvoice-dpo-tts" / "train-x"
        path = self.request(sc, "train", out, profile="smoke")
        req = yaml.safe_load(path.read_text())
        req["mounts"] = [m for m in req["mounts"] if m["container"] != "/inputs/pairs"]
        path.write_text(yaml.safe_dump(req, sort_keys=False))
        proc, _ = self.launch(sc, path, "smoke", "--spec", str(sc.root / "spec.yaml"))
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue(any(e["rule"] == "EXT-DATA" for e in parse_errors(proc.stderr)), proc.stderr)


if __name__ == "__main__":
    unittest.main()
