"""Area "TTS" test helpers: a whole DPO run, its two evaluation arms and the records ext-collect reads, as the runtime
writes them (runtime/train_dpo.py `main`, `save_cut`, `write_env`; runtime/eval_arm.py; eval/score_tts.py; gen_tts.py),
with stand-in weights and audio: nothing in gatedtrain opens them (MM-12). Small real pieces: `fixtures/tts/env-freeze.txt`
(the runtime venv's distributions, listed as train_dpo.py `write_env` does) and the item rows' shape.

The pair order is train_dpo.py's own `schedule`, taken from runtime/train_dpo.py's source (`load_schedule`), so a change
of the trainer's order shows here.

Layout of `TtsScenario(root)`:
- `<root>/rt/` the runtime stand-in (mounted read-only at /inputs/rt): `.venv/bin/python`, `train_dpo.py`,
  `eval_arm.py` (stand-ins, never executed), `uv.lock`, `models/fun-cosyvoice3-0.5b-2512/` (the base's file names, stand-in
  contents), `items-hard-test/` (N items, 6 speakers), `items-hard-train/` (the training items of the pairs);
- `<root>/out/cosyvoice-dpo-tts/inputs/{pairs,cand,ref}`: the reused pairs, candidates and reference log-probs;
- `<root>/out/cosyvoice-dpo-tts/train-1/`: the run (12 pairs, 2 epochs, micro-batch 2 x accumulation 2: 6 steps, cuts 2,
  4, 6), `<root>/eval/{base,cand}/{gen,score}`: the evaluations, `<root>/arms/{base,cand}/<metric>`: the arms;
- spec, registration, trainer.yaml, launch record, smoke record, the host work's GPU logs.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pwd
import random
import unittest
from pathlib import Path
from typing import Any, Callable

import yaml

from gatedtrain.ext.arm import META_SCHEMA  # noqa: F401  (re-exported for the tests)
from gatedtrain.ext.launch import RECORD_SCHEMA
from gatedtrain.ext.smoke import SMOKE_SCHEMA
from gatedtrain.testing.ext import (ENV_HOST_WORK, ENV_INPUT_ROOTS, ENV_INTERPRETER, ENV_INTERPRETER_TREES,
                                 ENV_OUTPUT_ROOT, IMAGE_ID, gpu_step_script, host_config_sha256, launch_label, sha_of,
                                 write_canonical)
from gatedtrain.testing.support import canonical_bytes, file_sha256, manifest_entries, manifest_hash, run_cli
from gatedtrain_recipe_cosyvoice_dpo_tts import arms, pins
from gatedtrain_recipe_cosyvoice_dpo_tts.rows import RUNTIME_ENV, SAMPLER, SIM_ID, WER_ID

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tts"
ADAPTER = "cosyvoice_dpo"
TRAINER_ID = "dpo.cosyvoice3_llm@0.1"
SEED = 11
RT_C = "/inputs/rt"
BASE_NAME = "fun-cosyvoice3-0.5b-2512"
BASE_FILES = ("cosyvoice3.yaml", "llm.pt", "llm.rl.pt", "flow.pt", "hift.pt", "campplus.onnx",
              "speech_tokenizer_v3.onnx", "speech_tokenizer_v3.batch.onnx", "config.json", "README.md",
              "CosyVoice-BlankEN/config.json", "CosyVoice-BlankEN/tokenizer_config.json", "CosyVoice-BlankEN/vocab.json")
BOUND = ("llm.pt", "cosyvoice3.yaml", "CosyVoice-BlankEN/config.json", "CosyVoice-BlankEN/tokenizer_config.json",
         "CosyVoice-BlankEN/vocab.json")
REVISION = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
LABEL = "mm-a11-tts-train-full"
ENV_NAMES = ["HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "TOKENIZERS_PARALLELISM",
             "XDG_CACHE_HOME"]
SETTINGS = {"beta": 0.1, "lr": 1e-06, "epochs": 2, "micro_batch": 2, "grad_accum": 2, "warmup_steps": 10,
            "max_steps": None, "nll_weight": 0.0, "clip": 1.0, "seed": SEED, "amp": "bf16", "grad_ckpt": False,
            "ref_batch": 8, "save_every": 2}
N_PAIRS = 12
SPEAKERS = 12
ZIP = b"PK\x03\x04"


def run_user() -> str:
    """The container's `<uid>:<gid>` in fixture requests and records: the uid running the tests and its primary group,
    so the output directories the tests create are owned by the container's uid (EXT-63) whoever runs them. The launch
    rules refuse root, so a test that needs it skips under uid 0."""
    uid = os.getuid()
    if uid == 0:
        raise unittest.SkipTest("the launch rules refuse a container running as root (uid 0); run the tests as a "
                                "non-root user")
    return f"{uid}:{pwd.getpwuid(uid).pw_gid}"


def sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def compact_hash(obj: Any) -> str:
    return sha(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def write_json(path: Path, doc: Any) -> Path:
    """train_dpo.py `write_json`: indent 1, sorted keys, final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, docs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in docs), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def load_schedule() -> Callable:
    """train_dpo.py's `schedule` function, compiled from the runtime's source (it needs only `random`)."""
    src = (REPO / "runtime" / "train_dpo.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "schedule")
    ns: dict[str, Any] = {"random": random}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "train_dpo.schedule", "exec"), ns)
    return ns["schedule"]


# ---------------------------------------------------------------------------
# The runtime stand-in: base, item sets
# ---------------------------------------------------------------------------

def build_base(d: Path, tag: str = "") -> Path:
    for rel in BASE_FILES:
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ZIP if rel.endswith(".pt") else b"") + f"fixture stand-in {rel} {tag}\n".encode())
    return d


def item_row(split: str, cat: str, i: int, *, speaker: str, ref_text: bool = False) -> dict:
    h = hashlib.sha256(f"{split}{cat}{i}".encode()).hexdigest()[:12]
    row = {"category": cat, "item": f"hard-{split}:{cat}:{h}", "prompt_id": f"{speaker}_1_{i:06d}",
           "prompt_text": f"A prompt sentence number {i} read by speaker {speaker}.",
           "prompt_wav": f"unpacked-x/{speaker}/{speaker}_1_{i:06d}.wav", "source_id": None, "speaker": speaker,
           "text": f"The {split} sentence {i} costs ${i}.{i % 10}0 on the {i % 28 + 1}th."}
    if ref_text:
        row["ref_text"] = row["text"].replace("$", "") + " dollars"
    return row


def build_items(d: Path, split: str, n: int, *, texts: dict[int, str] | None = None) -> list[dict]:
    cats = ("number", "money", "date", "unit", "natural_long")
    rows = [item_row(split, cats[i % len(cats)], i, speaker=str(100 + i % SPEAKERS), ref_text=(i % 7 == 3))
            for i in range(n)]
    for i, t in (texts or {}).items():
        rows[i]["text"] = t
    write_jsonl(d / "items.jsonl", rows)
    write_json(d / "item_set_meta.json", {"n_items": n, "set": split, "items_sha256": file_sha256(d / "items.jsonl")})
    return rows


# ---------------------------------------------------------------------------
# Inputs and the training run (train_dpo.py main)
# ---------------------------------------------------------------------------

def build_inputs(d: Path, train_rows: list[dict], train_items: Path) -> dict[str, Path]:
    cand, pairs, ref = d / "cand", d / "pairs", d / "ref"
    rows = [{"audio_seconds": 3.0, "error": None, "item": r["item"], "k": k, "n_tokens": 90, "stop": 6562}
            for r in train_rows[:N_PAIRS] for k in range(2)]
    write_jsonl(cand / "candidates.jsonl", rows)
    write_json(cand / "candidates_meta.json", {"text_split": "train", "voice_split": "train_clean_100",
                                                "items_sha256": file_sha256(train_items / "items.jsonl"),
                                                "candidates_sha256": file_sha256(cand / "candidates.jsonl")})
    pair_rows = [{"category": r["category"], "chosen_k": 1, "item": r["item"], "rejected_k": 0, "wer_chosen": 0.0,
                  "wer_rejected": 0.2} for r in train_rows[:N_PAIRS]]
    write_jsonl(pairs / "pairs.jsonl", pair_rows)
    write_json(pairs / "pairs_meta.json", {"candidates_sha256": file_sha256(cand / "candidates.jsonl"),
                                            "items_sha256": file_sha256(train_items / "items.jsonl"),
                                            "pairs_sha256": file_sha256(pairs / "pairs.jsonl")})
    ref.mkdir(parents=True)
    return {"cand": cand, "pairs": pairs, "ref": ref}


def base_ref_doc(base: Path) -> dict:
    files = {rel: file_sha256(base / rel) for rel in BOUND}
    return {"repo": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "revision": REVISION, "files": files,
            "files_sha256": compact_hash(files)}


def build_cut(out: Path, step: int, *, pairs_consumed: int, base: Path, script_sha256: str, profile: str = "full",
              seed: int = SEED, tag: str = "") -> Path:
    """`checkpoint-<step>/` as train_dpo.py save_cut writes it, cut.json last."""
    d = out / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "llm.pt").write_bytes(ZIP + f"trained llm step {step} {tag}".encode())
    write_json(d / "base_ref.json", base_ref_doc(base))
    write_json(d / "cut.json", {
        "schema": "gatedtrain_recipe_cosyvoice_dpo_tts.cut", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "global_step": step, "pairs_consumed": pairs_consumed,
        "files": {n: file_sha256(d / n) for n in ("llm.pt", "base_ref.json")}, "deployable": ["llm.pt"],
        "script_sha256": script_sha256, "seed": seed, "profile": profile})
    return d


def simulate(pair_lines: list[str], s: dict) -> list[dict]:
    """samples.jsonl of train_dpo.py `train`: one line per pair of each step, in the schedule's order."""
    steps = load_schedule()(len(pair_lines), s["epochs"], s["micro_batch"], s["grad_accum"], s["seed"], s["max_steps"])
    out, k = [], 0
    for n, micros in enumerate(steps, start=1):
        for mb, m in enumerate(micros):
            for pos, i in enumerate(m):
                p = json.loads(pair_lines[i])
                out.append({"step": n, "epoch": k // len(pair_lines), "micro_batch": mb, "position": pos,
                            "pair_line": i, "item": p["item"], "chosen_k": p["chosen_k"],
                            "rejected_k": p["rejected_k"], "row": sha(pair_lines[i].encode("utf-8"))})
                k += 1
    return out


def build_run(out: Path, *, base: Path, inputs: dict[str, Path], script_sha256: str, profile: str = "full",
              s: dict | None = None) -> Path:
    s = dict(s or SETTINGS)
    out.mkdir(parents=True)
    lines = [x for x in (inputs["pairs"] / "pairs.jsonl").read_text(encoding="utf-8").splitlines() if x]
    samples = simulate(lines, s)
    n_steps = max(x["step"] for x in samples)
    llm_sha = file_sha256(base / "llm.pt")
    ref = base_ref_doc(base)
    write_json(out / "run_args.json", {
        "schema": "gatedtrain_recipe_cosyvoice_dpo_tts.run_args", "schema_version": "1.0", "profile": profile,
        "dry_run": False, "seed": s["seed"], "settings": s, "n_steps": n_steps, "argv": [],
        "model_dir": f"{RT_C}/models/{BASE_NAME}", "llm_init": f"{RT_C}/models/{BASE_NAME}/llm.pt",
        "llm_init_sha256": llm_sha, "base_ref": {**ref, "verified": True, "llm_init_is_base": True},
        "pairs_file": "/inputs/pairs/pairs.jsonl", "pairs_sha256": file_sha256(inputs["pairs"] / "pairs.jsonl"),
        "pairs_meta_sha256": file_sha256(inputs["pairs"] / "pairs_meta.json"), "n_pairs": len(lines),
        "candidates_dir": "/inputs/cand", "candidates_jsonl_sha256": file_sha256(inputs["cand"] / "candidates.jsonl"),
        "candidates_meta_sha256": file_sha256(inputs["cand"] / "candidates_meta.json"),
        "ref_cache": "/inputs/ref/ref_logps.jsonl", "ref_cache_sha256": None, "script_sha256": script_sha256,
        "code": f"src/CosyVoice@{pins.COSYVOICE_COMMIT}"})
    (out / "env-freeze.txt").write_bytes((FIXTURES / "env-freeze.txt").read_bytes())
    write_json(out / "env-runtime.json", {"python": "3.12.16", "executable": f"{RT_C}/.venv/bin/python",
                                          "script_sha256": script_sha256, "code": f"src/CosyVoice@{pins.COSYVOICE_COMMIT}",
                                          "env_freeze_sha256": file_sha256(out / "env-freeze.txt")})
    head = {"meta": {"llm_sha256": llm_sha, "pairs_sha256": file_sha256(inputs["pairs"] / "pairs.jsonl"), "amp": "bf16"}}
    (out / "ref_logps.jsonl").write_text(json.dumps(head) + "\n" + "".join(
        json.dumps({"key": f"{json.loads(x)['item']}#{k}", "logp": -100.0}) + "\n" for x in lines for k in (0, 1)))
    write_jsonl(out / "samples.jsonl", samples)
    write_jsonl(out / "train_log.jsonl", [{"step": n, "loss": 0.69, "seconds": 1.5, "peak_gpu_gib": 15.0}
                                          for n in range(1, n_steps + 1)])
    every = s["save_every"]
    for step in sorted({k for k in range(every, n_steps + 1, every)} | {n_steps}):
        build_cut(out, step, pairs_consumed=sum(1 for x in samples if x["step"] <= step), base=base,
                  script_sha256=script_sha256, profile=profile, seed=s["seed"])
    write_json(out / "train_meta.json", {"script_sha256": script_sha256, "n_pairs": len(lines)})
    return out


# ---------------------------------------------------------------------------
# Evaluations (eval_arm.py: gen/ by gen_tts.py, score/ by score_tts.py) and their arms (arms.py)
# ---------------------------------------------------------------------------

def base_wer(i: int) -> float:
    return round(0.1 + 0.05 * (i % 4), 4)


def cand_wer(i: int) -> float:
    return round(max(0.0, base_wer(i) - 0.06 - 0.02 * (i % 2)), 4)


def base_sim(i: int) -> float:
    return round(0.78 + 0.01 * (i % 5), 4)


def cand_sim(i: int) -> float:
    return round(base_sim(i) + 0.002 * (i % 3), 4)


def build_eval(out: Path, items_dir: Path, item_rows: list[dict], *, llm_sha256: str, wer: Callable[[int], float],
               sim: Callable[[int], float], gen_errors: tuple[int, ...] = (), limit: int | None = None) -> Path:
    rows = item_rows[:limit]
    gen, score = out / "gen", out / "score"
    gen_rows, scores = [], []
    for i, it in enumerate(rows):
        name = it["item"].replace(":", "__") + ".wav"
        if i in gen_errors:
            gen_rows.append({"item": it["item"], "error": "RuntimeError: CUDA out of memory", "wall_seconds": 1.0})
            scores.append({"item": it["item"], "speaker": it["speaker"],
                           "error": "RuntimeError: generation error: RuntimeError: CUDA out of memory"})
            continue
        (gen / "wav").mkdir(parents=True, exist_ok=True)
        (gen / "wav" / name).write_bytes(b"RIFF" + f"stand-in audio {out.name} {i}".encode())
        gen_rows.append({"item": it["item"], "error": None, "wav": f"wav/{name}", "audio_seconds": 3.2,
                         "wall_seconds": 2.0})
        scores.append({"item": it["item"], "speaker": it["speaker"], "error": None, "wer_qwen_canon": wer(i),
                       "wer_whisper_canon": wer(i), "wer_qwen": wer(i), "wer_whisper": wer(i), "sim_prompt": sim(i),
                       "asr_qwen": f"asr qwen {i}", "asr_whisper": f"asr whisper {i}",
                       "ref_source": "ref_text" if it.get("ref_text") else "text", "audio_seconds": 3.2})
    write_jsonl(gen / "gen.jsonl", gen_rows)
    write_json(gen / "gen_meta.json", {"llm_sha256": llm_sha256, "items_sha256": file_sha256(items_dir / "items.jsonl"),
                                        "n_items": len(rows), "seed": SEED, "text_frontend": False,
                                        "script_sha256": pins.SCRIPTS["gen_tts.py"], "peak_gpu_gib": 3.6})
    write_jsonl(score / "scores.jsonl", scores)
    write_json(score / "score_meta.json", {
        "items_sha256": file_sha256(items_dir / "items.jsonl"), "gen_jsonl_sha256": file_sha256(gen / "gen.jsonl"),
        "script_sha256": pins.SCRIPTS["eval/score_tts.py"], "wer_canon_sha256": pins.SCRIPTS["eval/wer_canon.py"],
        "canon_version": "canon-1", "versions": {"qwen-asr": "0.0.6", "speechbrain": "1.1.2", "torch": "2.8.0+cu128"}})
    return out


def smoke_record(*, base_model: str, item_set: str, **overrides) -> dict[str, Any]:
    """A PASS smoke.json (MM-38, R18-8): 20 baseline scores of each clause evaluator, none sharing a value."""
    wer = [0.01 * (i + 1) for i in range(20)]
    rec = {"schema": SMOKE_SCHEMA, "adapter": ADAPTER, "area": "tts", "base_model_hash": base_model,
           "item_set_hash": item_set, "peak_gpu_mem_gb": 17.0, "host_ram_peak_gb": 20.0, "swap_peak_gb": 0.0,
           "seconds_per_step": 1.5, "projected_gpu_h": 0.1, "gpu_log_total_h": 3.12,
           "baseline_scores": wer, "error_count": 0, "verdict": "PASS", "reasons": [],
           "baseline_scores_by_evaluator": {WER_ID: wer, SIM_ID: [0.70 + 0.01 * i for i in range(20)]}}
    rec.update(overrides)
    return rec


def guard_log(host_work: Path, label: str) -> Path:
    """gpu-step.sh's log: one finished attempt of 600 s, before the run's files were written."""
    p = host_work / "gpu-guard.log"
    p.write_text(f"2026-11-11T09:00:00Z START {label} attempt=1\n2026-11-11T09:10:00Z END {label} attempt=1 rc=0\n",
                 encoding="utf-8")
    return p


class TtsScenario:
    """A whole TTS DPO run collected against the base (module docstring)."""

    def __init__(self, root: Path, *, n_items: int = 60, overlap: int = 0, s: dict | None = None,
                 reg_mutate: Callable[[dict], None] | None = None):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.runtime = root / "rt"
        self.items_dir = self.runtime / "items-hard-test"
        self.train_items = self.runtime / "items-hard-train"
        self.train_rows = build_items(self.train_items, "train", N_PAIRS + 4)
        texts = {i: self.train_rows[i]["text"] for i in range(overlap)}
        self.item_rows = build_items(self.items_dir, "test", n_items, texts=texts)
        self.base = build_base(self.runtime / "models" / BASE_NAME)
        self.script = self.runtime / "train_dpo.py"
        self.script.write_text("# fixture stand-in for runtime/train_dpo.py; never executed\n", encoding="utf-8")
        (self.runtime / "eval_arm.py").write_text("# fixture stand-in for runtime/eval_arm.py\n", encoding="utf-8")
        (self.runtime / "uv.lock").write_text("# lock stand-in\n", encoding="utf-8")
        self.interpreter = self.runtime / ".venv" / "bin" / "python"
        self.interpreter.parent.mkdir(parents=True)
        self.interpreter.write_text("# fixture stand-in for the runtime venv's interpreter\n", encoding="utf-8")
        (root / "trainer.yaml").write_text(f"backend: {ADAPTER}\nprofile: full\n", encoding="utf-8")
        self.host_work = root / "host-work"
        self.output_root = root / "out"
        gpu_step_script(self.host_work)
        (self.host_work / "gpu-log.md").write_bytes((FIXTURES / "gpu-log.fixture.md").read_bytes())
        self.inputs = build_inputs(self.output_root / "cosyvoice-dpo-tts" / "inputs", self.train_rows,
                                   self.train_items)
        self.data = self.inputs["pairs"]
        self.settings = dict(s or SETTINGS)
        self.stage_out = build_run(self.output_root / "cosyvoice-dpo-tts" / "train-1", base=self.base,
                                   inputs=self.inputs, script_sha256=file_sha256(self.script), s=self.settings)
        (self.inputs["ref"] / "ref_logps.jsonl").write_bytes((self.stage_out / "ref_logps.jsonl").read_bytes())
        self.cuts = sorted(self.stage_out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        self.final_cut = self.cuts[-1]
        self.item_set_hash = manifest_hash(self.items_dir)
        self.reg_mutate = reg_mutate
        self.render()
        self.base_eval = build_eval(root / "eval" / "base", self.items_dir, self.item_rows,
                                    llm_sha256=file_sha256(self.base / "llm.pt"), wer=base_wer, sim=base_sim)
        self.cand_eval = build_eval(root / "eval" / "cand", self.items_dir, self.item_rows,
                                    llm_sha256=file_sha256(self.final_cut / "llm.pt"), wer=cand_wer,
                                    sim=cand_sim)
        self.arms = {}
        for tag, ev, ck in (("base", self.base_eval, self.base), ("cand", self.cand_eval, self.final_cut)):
            built = arms.build(gen=ev / "gen", score=ev / "score", items=self.items_dir, checkpoint=ck,
                               metrics=["wer_qwen_canon", "sim_prompt"])
            arms.write(root / "arms" / tag, built)
            self.arms[tag] = root / "arms" / tag
        self.baseline = self.arms["base"] / "wer_qwen_canon"
        self.candidate = self.arms["cand"] / "wer_qwen_canon"
        self.baseline_sim = self.arms["base"] / "sim_prompt"
        self.candidate_sim = self.arms["cand"] / "sim_prompt"
        self.launch = write_canonical(root / "launch.json", self.launch_record())
        self.label = self.launch_record()["label"]
        guard_log(self.host_work, self.label)
        self._smoke: Path | None = None

    def env(self) -> dict[str, str]:
        return {ENV_HOST_WORK: str(self.host_work), ENV_OUTPUT_ROOT: str(self.output_root),
                ENV_INPUT_ROOTS: str(self.root), ENV_INTERPRETER: f"{RT_C}/.venv/bin/python",
                ENV_INTERPRETER_TREES: str(self.interpreter.parent), RUNTIME_ENV: str(self.runtime)}

    @property
    def smoke(self) -> Path:
        if self._smoke is None:
            from gatedtrain_recipe_cosyvoice_dpo_tts.adapter import CosyVoiceDpoAdapter

            base_model = CosyVoiceDpoAdapter().identity(self.base)["model_hash"]
            self._smoke = write_canonical(self.root / "smoke.json", smoke_record(base_model=base_model,
                                                                                 item_set=self.item_set_hash))
        return self._smoke

    def render(self, reg_mutate: Callable[[dict], None] | None = None) -> None:
        from gatedtrain.ext import registry

        text = (FIXTURES / "registration.template.yaml").read_text(encoding="utf-8")
        text = text.replace("{item_set_hash}", self.item_set_hash).replace("{sampler}", json.dumps(SAMPLER))
        text = text.replace("{wer_hash}", registry.evaluator_identity_hash(WER_ID))
        text = text.replace("{sim_hash}", registry.evaluator_identity_hash(SIM_ID))
        mutate = reg_mutate or self.reg_mutate
        if mutate is not None:
            reg = yaml.safe_load(text)
            mutate(reg)
            text = yaml.safe_dump(reg, sort_keys=False)
        (self.root / "registration.yaml").write_text(text, encoding="utf-8")
        spec = (FIXTURES / "spec.template.yaml").read_text(encoding="utf-8")
        spec = spec.replace("{registration_sha256}", file_sha256(self.root / "registration.yaml")[len("sha256:"):])
        spec = spec.replace("{base}", str(self.base)).replace("{data}", str(self.data))
        spec = spec.replace("{items}", str(self.items_dir))
        (self.root / "spec.yaml").write_text(spec, encoding="utf-8")

    def mounts(self) -> list[dict]:
        return [{"host": str(self.runtime), "container": RT_C, "mode": "ro"},
                {"host": str(self.inputs["pairs"]), "container": "/inputs/pairs", "mode": "ro"},
                {"host": str(self.inputs["cand"]), "container": "/inputs/cand", "mode": "ro"},
                {"host": str(self.inputs["ref"]), "container": "/inputs/ref", "mode": "ro"},
                {"host": str(self.stage_out), "container": "/out", "mode": "rw"}]

    def argv(self, profile: str = "full") -> list[str]:
        return [f"{RT_C}/.venv/bin/python", f"{RT_C}/train_dpo.py", "--model-dir", f"{RT_C}/models/{BASE_NAME}",
                "--candidates", "/inputs/cand", "--pairs", "/inputs/pairs", "--ref-cache", "/inputs/ref/ref_logps.jsonl",
                "--out", "/out", "--profile", profile, "--seed", str(SEED), "--save-every", "2"]

    def launch_record(self, **overrides) -> dict[str, Any]:
        rec = {"schema": RECORD_SCHEMA, "schema_version": "2.0", "tags": ["prototype", "unpromotable"],
               "adapter": ADAPTER, "area": 11, "profile": "full", "label": LABEL, "host_work": str(self.host_work),
               "runtime": {"kind": "venv_in_container", "image": IMAGE_ID,
                           "venv_lock_sha256": file_sha256(self.runtime / "uv.lock")},
               "framework": {"name": "torch", "version": "2.5.1+cu124", "commit": None},
               "training_seed": SEED, "gpus": True, "argv": self.argv(), "mounts": self.mounts(), "tmpfs": ["/tmp"],
               "network": "none", "network_reason": None, "user": run_user(), "env_names": list(ENV_NAMES),
               "gpu_est_h": 0.1, "gpu_used_h": 3.12, "output_dir": str(self.stage_out),
               "output_root": str(self.output_root),
               "gpu_step_sha256": file_sha256(self.host_work / "bin" / "gpu-step.sh"), "allow_network": False,
               "interpreter_path": f"{RT_C}/.venv/bin/python", "interpreter_sha256": file_sha256(self.interpreter),
               "script_path": f"{RT_C}/train_dpo.py", "script_sha256": file_sha256(self.script),
               "host_config_sha256": host_config_sha256(self.host_work, self.output_root, [self.root],
                                                         interpreter=f"{RT_C}/.venv/bin/python",
                                                         interpreter_trees=[self.interpreter.parent])}
        rec.update(overrides)
        rec["label"] = launch_label(rec)
        return rec

    def rewrite_launch(self, **overrides) -> None:
        rec = self.launch_record(**overrides)
        self.launch.write_bytes(canonical_bytes(rec))
        self.label = rec["label"]
        guard_log(self.host_work, self.label)

    def edit_json(self, path: Path, fn: Callable[[dict], None]) -> None:
        doc = json.loads(path.read_text(encoding="utf-8"))
        fn(doc)
        write_json(path, doc)

    def reseal(self, cut: Path) -> None:
        """Rewrite cut.json's file hashes after a cut file was edited (the cut stays complete)."""
        self.edit_json(cut / "cut.json", lambda d: d.update(files={n: file_sha256(cut / n)
                                                                   for n in ("llm.pt", "base_ref.json")}))

    def fill(self, arm: Path, checkpoint: Path, *, base: Path | None = None):
        args = ["ext-arm-fill", "--adapter", ADAPTER, "--checkpoint", str(checkpoint), "--items", str(self.items_dir)]
        if base is not None:
            args += ["--base", str(base)]
        return run_cli(args + [str(arm)], cwd=self.root, env=self.env())

    def fill_all(self) -> None:
        for metric in ("wer_qwen_canon", "sim_prompt"):
            for proc in (self.fill(self.arms["base"] / metric, self.base),
                         self.fill(self.arms["cand"] / metric, self.final_cut, base=self.base)):
                if proc.returncode != 0:
                    raise AssertionError(proc.stderr)

    def collect(self, runs_root: Path, *, baseline: Path | None = None, candidate: Path | None = None,
                launch: Path | None = None, smoke: Path | None = None, timeout: int = 300):
        args = ["ext-collect", str(self.root / "spec.yaml"), "--adapter", ADAPTER, "--stage-out", str(self.stage_out),
                "--launch-record", str(launch or self.launch), "--smoke-record", str(smoke or self.smoke),
                "--eval-baseline", str(baseline or self.baseline), "--eval-candidate", str(candidate or self.candidate),
                "--eval-baseline", str(self.baseline_sim), "--eval-candidate", str(self.candidate_sim),
                "--runs-root", str(runs_root)]
        return run_cli(args, cwd=self.root, timeout=timeout, env=self.env())


def deployed_model_hash(base: Path, cut: Path) -> str:
    """MM-17 by construction: the manifest hash (H7) of a copy of the base whose llm.pt is the cut's."""
    entries = [e for e in manifest_entries(base) if e["path"] != "llm.pt"]
    entries += [e for e in manifest_entries(cut) if e["path"] == "llm.pt"]
    entries.sort(key=lambda e: e["path"].encode("utf-8"))
    return "sha256:" + hashlib.sha256(canonical_bytes(entries)).hexdigest()


__all__ = ["ADAPTER", "BASE_NAME", "SEED", "SETTINGS", "SIM_ID", "TRAINER_ID", "WER_ID", "TtsScenario", "build_cut",
           "build_eval", "compact_hash", "deployed_model_hash", "load_schedule", "read_jsonl", "sha", "sha_of",
           "simulate", "smoke_record", "write_json", "write_jsonl"]
