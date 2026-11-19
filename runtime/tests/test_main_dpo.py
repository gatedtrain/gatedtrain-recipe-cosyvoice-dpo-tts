"""train_dpo.py (main venv, CPU, tiny random CosyVoice3LM; the real weights are never loaded).

The key test: the teacher-forced log-prob of a sampled sequence under train_dpo's input builder equals the step-wise
log-probs that the model's own `inference()` computed while sampling it (captured by sample_candidates.LLMCapture).
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path

import _support
import numpy as np
import torch

import sample_candidates as sc
import train_dpo as td


def sample_with_inference(llm, prompt_text, text, prompt_speech, seed):
    """Run llm.inference as llm_job does; return (tokens, stop, step logps, captured inputs)."""
    cap = sc.LLMCapture(llm)
    try:
        torch.manual_seed(seed)
        t = lambda ids: torch.tensor([ids], dtype=torch.int32)  # noqa: E731
        n = lambda ids: torch.tensor([len(ids)], dtype=torch.int32)  # noqa: E731
        list(llm.inference(text=t(text), text_len=n(text), prompt_text=t(prompt_text), prompt_text_len=n(prompt_text),
                           prompt_speech_token=t(prompt_speech), prompt_speech_token_len=n(prompt_speech),
                           embedding=torch.zeros(1, 192), uuid="x"))
        tokens, stop, logps = cap.result()
        return tokens, stop, logps, cap.inputs
    finally:
        cap.restore()


class TeacherForcingMatchesInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.llm = _support.tiny_llm(seed=3)
        size = cls.llm.speech_token_size
        cls.prompt_text = _support.toy_tokens("you are a helpful assistant") + [_support.ENDOFPROMPT] + \
            _support.toy_tokens("the prompt transcript says something long enough")
        cls.text = _support.toy_tokens("the quick brown fox jumps")
        cls.prompt_speech = [(7 * i + 3) % size for i in range(11)]
        cls.samples = [sample_with_inference(cls.llm, cls.prompt_text, cls.text, cls.prompt_speech, s)
                       for s in range(4)]

    def ex(self, tokens, stop, prompt_speech=None):
        return td.build_example(self.prompt_text, self.text, self.prompt_speech if prompt_speech is None else
                                prompt_speech, tokens, stop)

    def test_samples_are_nontrivial(self):
        lens = [len(s[0]) for s in self.samples]
        self.assertTrue(all(n >= 5 for n in lens), lens)
        self.assertEqual(len(set(map(tuple, (s[0] for s in self.samples)))), len(self.samples))

    def test_tokenwise_match(self):
        for tokens, stop, logps, _ in self.samples:
            with torch.no_grad():
                lp = td.token_logps(self.llm, [self.ex(tokens, stop)], "cpu")[0]
            self.assertEqual(lp.numel(), len(logps))
            np.testing.assert_allclose(lp.numpy(), np.asarray(logps), rtol=0, atol=2e-4)

    def test_batched_with_padding_matches(self):
        exs = [self.ex(t, s) for t, s, _, _ in self.samples]
        with torch.no_grad():
            batched, n = td.sequence_logps(self.llm, exs, "cpu")
            single = torch.stack([td.sequence_logps(self.llm, [e], "cpu")[0][0] for e in exs])
        np.testing.assert_allclose(batched.numpy(), single.numpy(), rtol=0, atol=1e-3)
        np.testing.assert_allclose(batched.numpy(), [sum(s[2]) for s in self.samples], rtol=0, atol=2e-3)
        self.assertEqual(n.tolist(), [len(s[2]) for s in self.samples])

    def test_stop_and_max_len(self):
        # Make eos likely so a sample ends with the stop token; then make stops impossible so it runs to max_len.
        llm = _support.tiny_llm(seed=5)
        eos = llm.eos_token
        with torch.no_grad():
            llm.llm_decoder.weight[eos] = llm.llm_decoder.weight[: llm.speech_token_size].mean(0) * 0 + \
                llm.llm_decoder.weight[0] * 1.0
        stops = []
        for s in range(12):
            tokens, stop, logps, _ = sample_with_inference(llm, self.prompt_text, self.text, self.prompt_speech, s)
            if stop is not None:
                stops.append(stop)
                with torch.no_grad():
                    lp = td.token_logps(llm, [td.build_example(self.prompt_text, self.text, self.prompt_speech,
                                                               tokens, stop)], "cpu")[0]
                np.testing.assert_allclose(lp.numpy(), logps, atol=2e-4)
        self.assertTrue(stops, "no sample stopped")
        self.assertTrue(all(st in llm.stop_token_ids for st in stops))
        llm2 = _support.tiny_llm(seed=6)
        with torch.no_grad():
            llm2.llm_decoder.weight[llm2.speech_token_size:] = 0
            llm2.llm_decoder.weight[: llm2.speech_token_size].mul_(4)
        tokens, stop, logps, _ = sample_with_inference(llm2, self.prompt_text, self.text, self.prompt_speech, 0)
        self.assertIsNone(stop)
        self.assertEqual(len(tokens), 20 * len(self.text))                  # max_len = 20 x text tokens
        with torch.no_grad():
            lp = td.token_logps(llm2, [td.build_example(self.prompt_text, self.text, self.prompt_speech, tokens,
                                                        None)], "cpu")[0]
        np.testing.assert_allclose(lp.numpy(), logps, rtol=1e-5, atol=2e-4)       # logits ~ -260 here

    def test_prompt_speech_is_context_not_target(self):
        tokens, stop, _, _ = self.samples[0]
        ex = self.ex(tokens, stop)
        self.assertEqual(ex["targets"], tokens + ([stop] if stop is not None else []))
        x, first = td.embed_example(self.llm, ex, "cpu")
        self.assertEqual(x.shape[0], 1 + len(self.prompt_text) + len(self.text) + 1 + len(self.prompt_speech)
                         + len(tokens))
        self.assertEqual(first, len(self.prompt_text) + len(self.text) + 1 + len(self.prompt_speech))
        # Changing the prompt speech changes the log-probs (it is conditioning) ...
        with torch.no_grad():
            a = td.sequence_logps(self.llm, [ex], "cpu")[0]
            b = td.sequence_logps(self.llm, [self.ex(tokens, stop, prompt_speech=self.prompt_speech[::-1])], "cpu")[0]
        self.assertGreater(abs(float(a - b)), 1e-3)

    def test_captured_inputs(self):
        _, _, _, inputs = self.samples[0]
        self.assertEqual(inputs["prompt_text"].reshape(-1).tolist(), self.prompt_text)
        self.assertEqual(inputs["text"].reshape(-1).tolist(), self.text)
        self.assertEqual(inputs["prompt_speech_token"].reshape(-1).tolist(), self.prompt_speech)

    def test_bf16_autocast_close(self):
        tokens, stop, logps, _ = self.samples[0]
        with torch.no_grad():
            s, _ = td.sequence_logps(self.llm, [self.ex(tokens, stop)], "cpu", torch.bfloat16)
        self.assertLess(abs(float(s[0]) - sum(logps)) / len(logps), 0.1)


class Loss(unittest.TestCase):
    def test_dpo_loss_values(self):
        pc, pr = torch.tensor([-10.0, -20.0]), torch.tensor([-12.0, -15.0])
        rc, rr = torch.tensor([-11.0, -20.0]), torch.tensor([-11.0, -15.0])
        loss, st = td.dpo_loss(pc, pr, rc, rr, beta=0.1)
        m = torch.tensor([(1.0) - (-1.0), 0.0])
        self.assertAlmostEqual(loss.item(), float(-torch.nn.functional.logsigmoid(0.1 * m).mean()), places=6)
        self.assertEqual(st["reward_acc"], 0.5)
        self.assertAlmostEqual(st["chosen_reward"], 0.05, places=6)
        self.assertAlmostEqual(st["rejected_reward"], -0.05, places=6)
        loss0, _ = td.dpo_loss(rc, rr, rc, rr, beta=0.1)
        self.assertAlmostEqual(loss0.item(), math.log(2), places=6)
        loss_nll, _ = td.dpo_loss(pc, pr, rc, rr, beta=0.1, nll_weight=0.5, n_chosen=torch.tensor([5.0, 10.0]))
        self.assertAlmostEqual(loss_nll.item() - loss.item(), 0.5 * float((10 / 5 + 20 / 10) / 2), places=5)

    def test_gradient_flows_only_through_policy(self):
        llm = _support.tiny_llm(seed=1)
        ex = td.build_example([5, _support.ENDOFPROMPT], [7, 8, 9], [1, 2, 3], [4, 5, 6, 7], llm.eos_token)
        s, _ = td.sequence_logps(llm, [ex, ex], "cpu")
        loss, _ = td.dpo_loss(s[:1], s[1:], torch.tensor([0.0]), torch.tensor([0.0]), 0.1)
        loss.backward()
        self.assertIsNotNone(llm.llm_decoder.weight.grad)
        self.assertIsNotNone(llm.speech_embedding.weight.grad)

    def test_gradient_checkpointing_same_gradients(self):
        ex = td.build_example([5, _support.ENDOFPROMPT], [7, 8, 9], [1, 2, 3], [4, 5, 6, 7], 121)
        grads = []
        for ckpt in (False, True):
            llm = _support.tiny_llm(seed=1).train()
            if ckpt:
                llm.llm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            s, _ = td.sequence_logps(llm, [ex], "cpu")
            s.sum().backward()
            grads.append(llm.llm.model.model.layers[0].mlp.up_proj.weight.grad.clone())
        torch.testing.assert_close(grads[0], grads[1])

    def test_schedule_deterministic_and_complete(self):
        a = td.schedule(10, 2, 2, 3, 11, None)
        b = td.schedule(10, 2, 2, 3, 11, None)
        self.assertEqual(a, b)
        flat = [i for step in a for m in step for i in m]
        self.assertEqual(sorted(flat), sorted(list(range(10)) * 2))
        self.assertEqual(len(td.schedule(10, 2, 2, 3, 11, 2)), 2)
        self.assertNotEqual(td.schedule(10, 1, 10, 1, 11, None), td.schedule(10, 1, 10, 1, 12, None))


def write_candidates(root: Path, llm, items: list[dict], k: int, text_split: str = "train") -> tuple[Path, Path]:
    """A candidates dir made by sample_candidates.run with FakeCV, plus a pairs dir (chosen k=0, rejected k=1)."""
    cand = root / "cand"
    cv = _support.FakeCV(llm)
    sc.run(cv, items, cand, k, 11, root, torch.manual_seed, log=lambda s: None)
    voice = {"train": "train_clean_100", "dev": "dev_clean", "test": "test_clean"}[text_split]
    (cand / "candidates_meta.json").write_text(json.dumps({"text_split": "train" if text_split == "dev" else text_split,
                                                           "voice_split": voice, "items_sha256": "sha256:x"}))
    pairs = root / "pairs"
    pairs.mkdir()
    (pairs / "pairs.jsonl").write_text("".join(json.dumps({"item": it["item"], "chosen_k": 0, "rejected_k": 1}) + "\n"
                                               for it in items))
    (pairs / "pairs_meta.json").write_text(json.dumps({"candidates_sha256": td.sha256(cand / "candidates.jsonl")}))
    return cand, pairs


ITEMS = [{"item": f"hard-train:x:{i}", "text": f"word{i} alpha beta gamma delta", "prompt_text": "a prompt here",
          "prompt_wav": f"p{i % 2}.wav"} for i in range(4)]


class TrainLoop(unittest.TestCase):
    def args(self, **kw):
        base = dict(beta=0.1, lr=1e-3, epochs=2, micro_batch=2, grad_accum=1, warmup_steps=1, max_steps=None,
                    nll_weight=0.0, clip=1.0, seed=11, amp="none", save_every=0)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_end_to_end_tiny(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            llm = _support.tiny_llm(seed=2)
            cand, pairs_dir = write_candidates(d, llm, ITEMS, k=2)
            pairs, _ = td.load_pairs(cand, pairs_dir)
            self.assertEqual(len(pairs), 4)
            ref = td.reference_logps(llm, pairs, "cpu", None, batch=3)
            for p in pairs:      # teacher-forced reference == sampling-time log-probs (fp32)
                self.assertAlmostEqual(ref[td.pair_key(p, "chosen")], p["chosen_sample_logp"], delta=5e-3)
            torch.manual_seed(0)
            log = d / "log.jsonl"
            last = td.train(llm, pairs, ref, self.args(), "cpu", log)
            recs = [json.loads(l) for l in log.read_text().splitlines()]
            self.assertEqual(len(recs), 4)                                  # 4 pairs x 2 epochs / 2 per step
            self.assertAlmostEqual(recs[0]["loss"], math.log(2), places=4)  # step 0: policy == reference
            self.assertLess(last["loss"], recs[0]["loss"])                   # lr 1e-3 on a tiny model: it learns
            for k in ("reward_acc", "margin", "chosen_reward", "rejected_reward", "logp_chosen", "logp_rejected",
                      "kl_proxy", "grad_norm", "lr", "tokens"):
                self.assertIn(k, recs[0])

    def test_refuses_non_train_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for split in ("test", "dev"):
                cand, pairs_dir = write_candidates(d / split, _support.tiny_llm(seed=2), ITEMS[:1], k=2,
                                                   text_split=split)
                with self.assertRaises(SystemExit):
                    td.load_pairs(cand, pairs_dir)

    def test_main_cli_with_tiny_model(self):
        import sys
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            llm = _support.tiny_llm(seed=2)
            cand, pairs_dir = write_candidates(d, llm, ITEMS, k=2)
            model_dir = d / "model"
            model_dir.mkdir()
            torch.save(llm.state_dict(), model_dir / "llm.pt")

            def fake_load(md, f, device):
                m = _support.tiny_llm(seed=99)
                m.load_state_dict(torch.load(f, map_location="cpu", weights_only=True), strict=True)
                return m
            argv = ["train_dpo.py", "--model-dir", str(model_dir), "--candidates", str(cand), "--pairs", str(pairs_dir),
                    "--amp", "none", "--lr", "1e-3", "--micro-batch", "2", "--grad-accum", "1", "--epochs", "1",
                    "--warmup-steps", "1", "--save-every", "1", "--device", "cpu"]
            with mock.patch.object(td, "load_llm", fake_load), \
                    mock.patch.object(sys, "argv", argv + ["--out", str(d / "o1")]):
                td.main()
            out = d / "o1"
            sd = torch.load(out / "checkpoint-2" / "llm.pt", map_location="cpu", weights_only=True)
            fresh = _support.tiny_llm(seed=5)
            fresh.load_state_dict(sd, strict=True)                      # as gen_tts.py --llm PATH loads it
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["checkpoint-1", "checkpoint-2", "env-freeze.txt", "env-runtime.json", "ref_logps.jsonl",
                              "run_args.json", "samples.jsonl", "train_log.jsonl", "train_meta.json"])
            self.assertTrue(any(not torch.equal(sd[k], v) for k, v in llm.state_dict().items()))
            meta = json.loads((out / "train_meta.json").read_text())
            for k in ("script_sha256", "llm_init_sha256", "pairs_sha256", "candidates_jsonl_sha256", "items_sha256",
                      "llm_dpo_sha256", "seed", "versions", "ref_vs_sampling", "cuts", "profile"):
                self.assertIn(k, meta)
            self.assertEqual(meta["llm_dpo_sha256"], td.sha256(out / "checkpoint-2" / "llm.pt"))
            self.check_layout(out, d, cand, pairs_dir, model_dir, n_steps=2, cuts=(1, 2))
            self.assertLess(meta["ref_vs_sampling"]["max_abs_logp_diff_per_token"], 1e-3)
            log = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
            self.assertEqual(len(log), 2)
            self.assertAlmostEqual(log[0]["loss"], math.log(2), places=4)
            # Reusing the reference cache gives the same first step.
            with mock.patch.object(td, "load_llm", fake_load), mock.patch.object(
                    sys, "argv", argv + ["--out", str(d / "o2"), "--ref-cache", str(out / "ref_logps.jsonl"),
                                         "--max-steps", "1"]):
                td.main()
            log2 = [json.loads(l) for l in (d / "o2" / "train_log.jsonl").read_text().splitlines()]
            self.assertEqual(len(log2), 1)
            self.assertAlmostEqual(log2[0]["loss"], log[0]["loss"], places=6)

    def check_layout(self, out: Path, d: Path, cand: Path, pairs_dir: Path, model_dir: Path, n_steps: int, cuts):
        """The gatedtrain launcher layout (run_args.json, env files, samples.jsonl, checkpoint-<N>/ with cut.json)."""
        args = json.loads((out / "run_args.json").read_text())
        self.assertEqual((args["profile"], args["dry_run"], args["seed"], args["n_steps"]), ("full", False, 11, n_steps))
        self.assertEqual(args["pairs_sha256"], td.sha256(pairs_dir / "pairs.jsonl"))
        self.assertEqual(args["n_pairs"], len(ITEMS))
        self.assertEqual(args["candidates_jsonl_sha256"], td.sha256(cand / "candidates.jsonl"))
        self.assertEqual(args["llm_init_sha256"], td.sha256(model_dir / "llm.pt"))
        self.assertEqual(args["base_ref"]["files"], {"llm.pt": td.sha256(model_dir / "llm.pt")})
        self.assertFalse(args["base_ref"]["verified"])                  # no models/MODELS.json beside the model dir
        self.assertEqual(args["script_sha256"], td.sha256(Path(td.__file__)))
        rt = json.loads((out / "env-runtime.json").read_text())
        self.assertEqual(rt["script_sha256"], args["script_sha256"])
        freeze = (out / "env-freeze.txt").read_text().splitlines()
        self.assertTrue(any(line.startswith("torch==") for line in freeze))
        # samples.jsonl: exactly the schedule's pairs, in order, before each step
        sched = td.schedule(len(ITEMS), args["settings"]["epochs"], args["settings"]["micro_batch"],
                            args["settings"]["grad_accum"], 11, args["settings"]["max_steps"])
        lines = (pairs_dir / "pairs.jsonl").read_text().splitlines()
        samples = [json.loads(l) for l in (out / "samples.jsonl").read_text().splitlines()]
        want, k = [], 0
        for step, micros in enumerate(sched, start=1):
            for mb, m in enumerate(micros):
                for pos, i in enumerate(m):
                    want.append({"step": step, "epoch": k // len(ITEMS), "micro_batch": mb, "position": pos,
                                 "pair_line": i, "item": json.loads(lines[i])["item"], "chosen_k": 0, "rejected_k": 1,
                                 "row": "sha256:" + __import__("hashlib").sha256(lines[i].encode()).hexdigest()})
                    k += 1
        self.assertEqual(samples, want)
        for step in cuts:
            c = out / f"checkpoint-{step}"
            self.assertEqual(sorted(p.name for p in c.iterdir()), ["base_ref.json", "cut.json", "llm.pt"])
            doc = json.loads((c / "cut.json").read_text())
            self.assertEqual(doc["global_step"], step)
            self.assertEqual(doc["pairs_consumed"], sum(1 for x in samples if x["step"] <= step))
            self.assertEqual(doc["files"], {n: td.sha256(c / n) for n in ("llm.pt", "base_ref.json")})
            self.assertEqual(doc["deployable"], ["llm.pt"])
            self.assertEqual((doc["seed"], doc["profile"], doc["script_sha256"]), (11, "full", args["script_sha256"]))
            ref = json.loads((c / "base_ref.json").read_text())
            self.assertEqual(ref["files"], args["base_ref"]["files"])

    def test_out_must_be_empty(self):
        import sys
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "o").mkdir()
            (d / "o" / "x").write_text("x")
            argv = ["train_dpo.py", "--model-dir", str(d), "--candidates", str(d), "--pairs", str(d), "--out",
                    str(d / "o")]
            with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                td.main()

    def test_base_binding_checks_models_json(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "models"
            md = root / "m"
            (md / "CosyVoice-BlankEN").mkdir(parents=True)
            (md / "llm.pt").write_bytes(b"w")
            (md / "cosyvoice3.yaml").write_text("y")
            (md / "CosyVoice-BlankEN" / "config.json").write_text("{}")
            (md / "flow.pt").write_bytes(b"f")                                    # not part of the LLM
            files = {r: td.sha256(md / r) for r in ("llm.pt", "cosyvoice3.yaml", "CosyVoice-BlankEN/config.json")}
            entry = {"repo": "r/m", "revision": "a" * 40,
                     "files": {r: {"sha256": h.split(":", 1)[1]} for r, h in files.items()}}
            (root / "MODELS.json").write_text(json.dumps({"models": {"m": entry}}))
            ref = td.base_binding(md, md / "llm.pt")
            self.assertEqual(ref["files"], files)
            self.assertTrue(ref["verified"] and ref["llm_init_is_base"])
            self.assertEqual((ref["repo"], ref["revision"]), ("r/m", "a" * 40))
            (md / "cosyvoice3.yaml").write_text("changed")
            with self.assertRaises(SystemExit):
                td.base_binding(md, md / "llm.pt")

    def test_state_dict_round_trip(self):
        llm = _support.tiny_llm(seed=4)
        sd = {k: v.detach().clone() for k, v in llm.state_dict().items()}
        other = _support.tiny_llm(seed=9)
        other.load_state_dict(sd, strict=True)
        self.assertIn("llm.model.lm_head.weight", sd)
        self.assertIn("llm.model.model.embed_tokens.weight", sd)
        self.assertIn("speech_embedding.weight", sd)
        self.assertIn("llm_decoder.weight", sd)


if __name__ == "__main__":
    unittest.main()
