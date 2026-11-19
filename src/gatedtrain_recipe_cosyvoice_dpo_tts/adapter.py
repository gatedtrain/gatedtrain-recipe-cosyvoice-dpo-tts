"""CosyVoice3 LLM DPO adapter (area "TTS": Fun-CosyVoice3-0.5B, sigmoid DPO on the speech-token LLM with
WER-ranked pairs; MULTIMODAL-SPEC.md section 6).

MM-12: a pure-Python class that imports no framework, torch or pickle loader. It reads JSON, JSONL and the text
`env-freeze.txt`, and hashes everything else; a cut's `llm.pt` (a torch state dict, a pickle in a zip archive) is
hashed and checked for the zip signature only, never opened.

Run layout, read from the runtime's code (`runtime/train_dpo.py`; where a document differs, the code is followed):

- `save_cut` writes `checkpoint-<N>/` (N the optimizer step) into `.checkpoint-<N>.partial/` and renames it:
  `llm.pt` (the trained LLM, fp32, exactly the keys of the base's llm.pt), `base_ref.json` (repo, revision, the SHA-256
  of every base file the LLM is built from: llm.pt, cosyvoice3.yaml, CosyVoice-BlankEN/*, and `files_sha256`), then
  `cut.json` last: `global_step`, `pairs_consumed`, `files` (SHA-256 of llm.pt and base_ref.json), `deployable`
  (["llm.pt"]), `script_sha256`, `seed`, `profile`. Cuts at every `--save-every` multiple and at the last step;
- `samples.jsonl`: one line per pair of each optimizer step, written before `opt.step()`: `{step, epoch, micro_batch,
  position, pair_line, item, chosen_k, rejected_k, row}`, `row` = SHA-256 of the pairs.jsonl line's text;
- `run_args.json`: `profile`, `dry_run`, `seed`, `settings` (beta, lr, epochs, micro_batch, grad_accum, warmup_steps,
  max_steps, nll_weight, clip, seed, amp, grad_ckpt, ref_batch, save_every), `n_steps`, `model_dir` (container path),
  `llm_init`, `llm_init_sha256`, `base_ref` (+ `verified`, `llm_init_is_base`), `pairs_file`, `pairs_sha256`,
  `pairs_meta_sha256`, `n_pairs`, `candidates_dir`, `candidates_jsonl_sha256`, `candidates_meta_sha256`, `ref_cache`,
  `ref_cache_sha256`, `script_sha256`; `env-runtime.json` `script_sha256` (EXT-83); `env-freeze.txt` `name==version`.

The pair order is `schedule()` of train_dpo.py: per epoch e, `list(range(n_pairs))` shuffled by
`random.Random(f"dpo:{seed}:{e}")`, the epochs concatenated, cut into micro-batches of `micro_batch` pairs and those
into optimizer steps of `grad_accum` micro-batches (the last step keeps the rest), the first `max_steps` steps. The
adapter re-derives it with the standard library and requires samples.jsonl to be exactly that order.

Identity (MM-17): a cut deploys the base directory with its `llm.pt` replaced by the cut's. `model_hash` of a cut is
the manifest hash (H7) of the base's entries with the `llm.pt` entry replaced by the cut's, i.e. the manifest hash of
the deployed directory; `adapter_hash` is the hash of the cut's llm.pt entry and `base_ref` names the base, so the
lineage records the composition (MM-18). The base alone: `model_hash` = `cut_hash` = its manifest hash.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from gatedtrain.canonical import entries_hash, file_sha256, manifest_entries, manifest_hash, text_lines
from gatedtrain.errors import Problem
from gatedtrain.ext.registry import EVALUATOR_ROWS, TRAINER_ROWS
from gatedtrain.ext.sdk import (compact_sha256, is_int as _int, plain_file as _plain_file, read_cut_marker,
                             read_json as _read_json, read_jsonl, read_run_args, runtime_argv_problems,
                             runtime_script_problems, single_shard as _shard)

NAME = "cosyvoice_dpo"
LIBRARY = ("the runtime's own sigmoid DPO loop on the CosyVoice3 LLM (train_dpo.py, plain torch) with its cut writer "
           "(save_cut: llm.pt and base_ref.json, cut.json written last, atomic rename)")
CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")        # train_dpo.py save_cut; `.checkpoint-<N>.partial` never
CUT_MARKER = "cut.json"
WEIGHTS = "llm.pt"
BASE_REF = "base_ref.json"
COMPONENTS = tuple(TRAINER_ROWS[NAME]["components"])          # ("llm.pt",)
CUT_FILES = (WEIGHTS, BASE_REF)                               # train_dpo.py CUT_FILES
FRAMEWORK_PACKAGE = "torch"
PINS = tuple(TRAINER_ROWS[NAME]["pins"])
FREEZE_FILE = "env-freeze.txt"
SAMPLES = "samples.jsonl"
REF_LOGPS = "ref_logps.jsonl"
PAIRS = "pairs.jsonl"
PAIRS_META = "pairs_meta.json"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
ZIP_MAGIC = b"PK\x03\x04"                                     # torch.save's zip archive format
# A plain Fun-CosyVoice3-0.5B-2512 directory (models/MODELS.json): the files gen_tts.py's AutoModel loads.
BASE_REQUIRED = ("cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v3.onnx",
                 "CosyVoice-BlankEN/config.json")
# train_dpo.py BASE_FILES and BASE_DIRS: the base files the trained LLM is built from (base_ref.json `files`).
BASE_BOUND_FILES = ("llm.pt", "cosyvoice3.yaml")
BASE_BOUND_DIRS = ("CosyVoice-BlankEN",)
TRAIN_SCRIPT = "train_dpo.py"
EVAL_SCRIPT = "eval_arm.py"
DEFAULT_BASE_REL = "models/fun-cosyvoice3-0.5b-2512"
PROFILES = ("smoke", "full")
EVAL_ITEM_SETS = ("items-hard-test", "items-hard-dev")
EVAL_LLMS = ("base", "rl", "/inputs/cut")
# argparse options of the two scripts (parse_args), for the launch-side argv check: option -> takes a value.
TRAIN_OPTIONS = {"--model-dir": True, "--candidates": True, "--pairs": True, "--out": True, "--profile": True,
                 "--llm": True, "--beta": True, "--lr": True, "--epochs": True, "--micro-batch": True,
                 "--grad-accum": True, "--warmup-steps": True, "--max-steps": True, "--nll-weight": True, "--clip": True,
                 "--seed": True, "--amp": True, "--grad-ckpt": False, "--ref-cache": True, "--ref-batch": True,
                 "--save-every": True, "--device": True}
EVAL_OPTIONS = {"--llm": True, "--item-set": True, "--out": True, "--seed": True, "--limit": True}

CHECKPOINT_COMPLETENESS = {
    "library": LIBRARY,
    "rule": ("a checkpoint-<N> directory is a cut iff its cut.json, written last before an atomic rename from "
             ".checkpoint-<N>.partial, parses with global_step = N and a positive integer pairs_consumed, deployable "
             "[\"llm.pt\"], and files listing exactly the other files of the directory (llm.pt, base_ref.json), each a "
             "plain file with its SHA-256; llm.pt is a zip archive (torch.save); .partial directories are never cuts; "
             "the cut's last step included is N"),
    "complete_dir_pattern": CHECKPOINT_DIR_RE.pattern,
    "sources": ["runtime/train_dpo.py save_cut (cut.json last, os.replace)", "runtime/train_dpo.py CUT_FILES, "
                "DEPLOYABLE", "runtime/train_dpo.py train (save_fn at every --save-every multiple and the last step)"],
    "verified": True,                     # tests run train_dpo.py main on CPU with a tiny CosyVoice3 LLM
}
SEED_DERIVATION = ("train_dpo.py set_seeds(S) seeds random, numpy and torch (CUDA included) with run_args.json `seed`, "
                   "which must equal the launch record's training_seed and every cut's cut.json `seed`; the pair order "
                   "is random.Random(f'dpo:{S}:{epoch}').shuffle per epoch (re-derived here and compared with "
                   "samples.jsonl). Evaluation: gen_tts.py set_all_random_seed(derive_seed('tts', 11, item)) per "
                   "item, the same noise for every arm; K = 1. bf16 autocast and cuDNN are not bit-reproducible, "
                   "so a rerun gives other weights; 05 section 14 H(run_seed, step, data_ref, sample_index) is not "
                   "claimed.")


def schedule(n_pairs: int, epochs: int, micro_batch: int, grad_accum: int, seed: int,
             max_steps: int | None) -> list[list[list[int]]]:
    """train_dpo.py `schedule`, byte for byte in behaviour: optimizer steps as lists of micro-batches of pair indices."""
    order: list[int] = []
    for e in range(epochs):
        idx = list(range(n_pairs))
        random.Random(f"dpo:{seed}:{e}").shuffle(idx)
        order += idx
    micro = [order[i: i + micro_batch] for i in range(0, len(order), micro_batch)]
    steps = [micro[i: i + grad_accum] for i in range(0, len(micro), grad_accum)]
    return steps[:max_steps] if max_steps else steps


def pair_lines(path: Path) -> list[str]:
    """The non-empty lines of pairs.jsonl as train_dpo.py load_pairs reads them (`read_text().splitlines()`)."""
    return [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def line_row(line: str) -> str:
    return "sha256:" + hashlib.sha256(line.encode("utf-8")).hexdigest()


class CosyVoiceDpoAdapter:
    """The MM-12 interface for the CosyVoice3 LLM DPO runs of this recipe's runtime."""

    name = NAME
    trainer_id = TRAINER_ROWS[NAME]["uses"][0]
    evaluator_ids = tuple(TRAINER_ROWS[NAME]["evaluators"])
    framework_name = FRAMEWORK_PACKAGE
    checkpoint_completeness = CHECKPOINT_COMPLETENESS
    seed_derivation = SEED_DERIVATION
    binds_spec = bool(TRAINER_ROWS[NAME]["spec_binding"])
    spec_binding_argv = True              # `ext-launch --spec` binds the training launches, not the evaluations
    consumption_fix = "collect the run's own samples.jsonl as train_dpo.py wrote it (MM-23, MM-27)"
    payload_source = ("the text of a training item whose pair the run applied (exact overlap, MM-11, EXT-107)")

    BASE_FIX = ("train from the spec's Fun-CosyVoice3-0.5B-2512 directory inside a read-only mount (train_dpo.py "
                "--model-dir <runtime>/models/fun-cosyvoice3-0.5b-2512, checked against models/MODELS.json at start), "
                "without --llm")
    DATA_FIX = ("train on the spec's pairs directory inside a read-only mount (train_dpo.py --pairs, its pairs.jsonl "
                "recorded in run_args.json pairs_file, pairs_sha256 and n_pairs) with the candidates it was made from")

    def __init__(self) -> None:
        self._base_cache: dict[str, list[dict]] = {}

    def metric(self, evaluator_id: str) -> dict[str, Any]:
        if evaluator_id not in self.evaluator_ids:
            raise KeyError(evaluator_id)
        row = EVALUATOR_ROWS[evaluator_id]
        return {"metric": row["metric"], "direction": row["direction"], "score_range": tuple(row["score_range"]),
                "error_classes": tuple(row["error_classes"]), "score_outcomes": tuple(row["score_outcomes"]),
                "scorer": dict(row["scorer"]), "binary": row["binary"], "tier": row["tier"]}

    # --- the base directory and cuts (MM-17, MM-18, MM-20) ---

    @staticmethod
    def is_cut(d: Path) -> bool:
        """Whether `d` is (meant to be) a cut: it holds cut.json. The plain base directory never does."""
        return (Path(d) / CUT_MARKER).exists() or (Path(d) / CUT_MARKER).is_symlink()

    def needs_base(self, d: Path) -> bool:
        """ext-arm-fill: a cut's model hash composes the base (MM-18), so `--base` is required."""
        return self.is_cut(d)

    @staticmethod
    def base_dir_problem(d: Path) -> str | None:
        """None when `d` is a plain Fun-CosyVoice3-0.5B-2512 directory (the files gen_tts.py loads)."""
        for rel in BASE_REQUIRED:
            if not _plain_file(Path(d) / rel):
                return f"{rel} is missing (not a Fun-CosyVoice3 model directory)"
        return None

    def _cut_files_problem(self, d: Path) -> str | None:
        """An unreadable file makes the directory incomplete, never an exception (EXT-154)."""
        try:
            return self._cut_files_problem_unguarded(d)
        except OSError as exc:
            return (f"{Path(exc.filename).name if exc.filename else Path(d).name} cannot be read "
                    f"({exc.__class__.__name__}: {exc.strerror or exc})")

    def _cut_files_problem_unguarded(self, d: Path) -> str | None:
        doc, why = read_cut_marker(d)
        if doc is None:
            return why
        files = doc["files"]
        if sorted(files) != sorted(CUT_FILES):
            return f"{CUT_MARKER} lists {sorted(files)}, not {sorted(CUT_FILES)}"
        present = sorted(p.name for p in Path(d).iterdir())
        if present != sorted([*CUT_FILES, CUT_MARKER]):
            extra = sorted(set(present) - set(CUT_FILES) - {CUT_MARKER})
            return f"the directory holds {extra or present}, which {CUT_MARKER} does not list"
        for name in CUT_FILES:
            if not _plain_file(Path(d) / name):
                return f"{name} is not a plain file"
            if file_sha256(Path(d) / name) != files[name]:
                return f"{name} is not the SHA-256 {CUT_MARKER} records"
        if doc.get("deployable") != list(COMPONENTS):
            return f"{CUT_MARKER} deployable is {doc.get('deployable')!r}, not {list(COMPONENTS)}"
        for key in ("global_step", "pairs_consumed"):
            if not (_int(doc.get(key)) and doc[key] >= 1):
                return f"{CUT_MARKER} {key} is {doc.get(key)!r}, not a positive integer"
        with (Path(d) / WEIGHTS).open("rb") as f:
            if f.read(4) != ZIP_MAGIC:
                return f"{WEIGHTS} is not a torch.save zip archive"
        return None

    def model_dir_problem(self, d: Path) -> str | None:
        """None when `d` is a complete cut (it holds cut.json) or a plain base directory."""
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        if self.is_cut(d):
            return self._cut_files_problem(d)
        return self.base_dir_problem(d)

    def cut_problem(self, d: Path, step: int) -> str | None:
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        bad = self._cut_files_problem(d)
        if bad:
            return bad
        doc, _ = read_cut_marker(d)
        if doc.get("global_step") != step:
            return f"{CUT_MARKER} global_step is {doc.get('global_step')!r}, not {step}"
        return None

    def checkpoint_dirs(self, stage_out: Path) -> list[tuple[int, Path]]:
        """Every top-level `checkpoint-<N>` directory, complete or not, in step order."""
        stage_out = Path(stage_out)
        if not stage_out.is_dir():
            return []
        out = []
        for p in stage_out.iterdir():
            m = CHECKPOINT_DIR_RE.match(p.name)
            if m and p.is_dir() and not p.is_symlink():
                out.append((int(m.group(1)), p))
        return sorted(out)

    def find_cuts(self, stage_out: Path) -> tuple[list[tuple[int, Path]], list[Path]]:
        """(complete [(global_step, path)] in step order, incomplete [path]); a cut's last step included is its optimizer
        step (05 section 12.4)."""
        complete, incomplete = [], []
        for step, d in self.checkpoint_dirs(stage_out):
            if self.cut_problem(d, step) is None:
                complete.append((step, d))
            else:
                incomplete.append(d)
        return complete, incomplete

    def _base_entries(self, base: Path) -> list[dict]:
        """The base directory's manifest entries, walked once per adapter instance (about 8.5 GB on the real base)."""
        key = str(Path(base).resolve())
        if key not in self._base_cache:
            self._base_cache[key] = manifest_entries(Path(base))
        return self._base_cache[key]

    def identity(self, cut: Path, base: Any = None, *, entries: list[dict] | None = None) -> dict[str, Any]:
        """MM-17, MM-18. The base alone: `model_hash` = `cut_hash` = its manifest hash. A cut: `model_hash` = the
        manifest hash of the base's entries with llm.pt replaced by the cut's (the deployed directory), `adapter_hash`
        over the cut's llm.pt entry, `cut_hash` over the whole cut, `base_ref` {repo, revision, manifest_hash}. Raises
        ValueError for a cut without `base`."""
        is_cut = self.is_cut(cut)
        if entries is None and not is_cut:
            entries = self._base_entries(Path(cut))
        if entries is None:
            entries = manifest_entries(Path(cut))
        if not is_cut:
            self._base_cache.setdefault(str(Path(cut).resolve()), entries)
            h = entries_hash(entries)
            return {"model_hash": h, "cut_hash": h, "components": [e["path"] for e in entries], "base_ref": None}
        if base is None:
            raise ValueError(f"{cut} is a cut: its model hash composes the base directory (its llm.pt replaced), so "
                             "the base directory is needed (MM-18)")
        comps = [e for e in entries if e["path"] in COMPONENTS]
        if [e["path"] for e in comps] != list(COMPONENTS):
            raise ValueError(f"{cut} holds no {WEIGHTS}")
        base_entries = self._base_entries(Path(base))
        if not any(e["path"] == WEIGHTS for e in base_entries):
            raise ValueError(f"the base {base} holds no {WEIGHTS}")
        deployed = sorted([e for e in base_entries if e["path"] not in COMPONENTS] + comps,
                          key=lambda e: e["path"].encode("utf-8"))
        ref, _ = _read_json(Path(cut) / BASE_REF)
        ref = ref if isinstance(ref, dict) else {}
        return {"model_hash": entries_hash(deployed), "cut_hash": entries_hash(entries),
                "adapter_hash": entries_hash(comps), "components": [e["path"] for e in comps],
                "base_ref": {"repo": ref.get("repo"), "revision": ref.get("revision"),
                             "manifest_hash": entries_hash(base_entries)}}

    # --- consumption (MM-23, MM-27) ---

    def _ledger_plan(self, stage_out: Path, data: Path, cuts: list) -> tuple[list[dict], list[str]]:
        """(the samples.jsonl lines of the steps up to the last cut, in file order) and the problems that refuse the log
        (empty when it is consistent). The schedule is re-derived from run_args.json and the spec's pairs."""
        args, why = read_run_args(stage_out)
        if args is None:
            return [], [why]
        if not cuts:
            return [], ["no complete cut"]
        s = args.get("settings") if isinstance(args.get("settings"), dict) else {}
        ints = {k: s.get(k) for k in ("epochs", "micro_batch", "grad_accum", "seed")}
        bad_ints = [k for k, v in ints.items() if not (_int(v) and v >= (0 if k == "seed" else 1))]
        max_steps = s.get("max_steps")
        if bad_ints or not (max_steps is None or (_int(max_steps) and max_steps >= 1)):
            return [], [f"run_args.json settings {bad_ints or ['max_steps']} are not positive integers (max_steps: "
                        "null or a positive integer)"]
        shard = _shard(data)
        if shard != PAIRS:
            return [], [f"the spec's training data {data} holds {shard or 'not exactly one shard'}, not the one "
                        f"{PAIRS} the run read"]
        try:
            lines = pair_lines(Path(data) / PAIRS)
        except (OSError, UnicodeDecodeError) as exc:
            return [], [f"{PAIRS} cannot be read ({exc.__class__.__name__})"]
        if args.get("n_pairs") != len(lines):
            return [], [f"run_args.json n_pairs {args.get('n_pairs')!r} is not the {len(lines)} lines of {PAIRS}"]
        steps = schedule(len(lines), ints["epochs"], ints["micro_batch"], ints["grad_accum"], ints["seed"], max_steps)
        if args.get("n_steps") != len(steps):
            return [], [f"run_args.json n_steps {args.get('n_steps')!r} is not the {len(steps)} steps its settings "
                        "give"]
        want: list[dict] = []
        k = 0
        items: list[Any] = []
        for line in lines:
            try:
                items.append(json.loads(line))
            except ValueError:
                return [], [f"a line of {PAIRS} does not parse"]
        for n, micros in enumerate(steps, start=1):
            for mb, m in enumerate(micros):
                for pos, i in enumerate(m):
                    p = items[i] if isinstance(items[i], dict) else {}
                    want.append({"step": n, "epoch": k // len(lines), "micro_batch": mb, "position": pos,
                                 "pair_line": i, "item": p.get("item"), "chosen_k": p.get("chosen_k"),
                                 "rejected_k": p.get("rejected_k"), "row": line_row(lines[i])})
                    k += 1
        samples, why = read_jsonl(Path(stage_out) / SAMPLES)
        if samples is None:
            return [], [why]
        out: list[str] = []
        if len(samples) != len(want):
            out.append(f"{SAMPLES} has {len(samples)} lines, not the {len(want)} pairs of the {len(steps)} steps the "
                       "settings give (the run did not finish, or the log belongs to another run)")
        for n_line, (got, exp) in enumerate(zip(samples, want), start=1):
            if got != exp:
                diff = sorted(k for k in set(got) | set(exp) if got.get(k) != exp.get(k))
                out.append(f"{SAMPLES} line {n_line} (step {exp['step']}) differs from train_dpo.py's schedule in "
                           f"{diff}")
                if len(out) >= 8:
                    break
        if out:
            return [], out
        last = 0
        for step, d in cuts:
            doc, why = read_cut_marker(d)
            if doc is None:
                return [], [f"{Path(d).name}: {why}"]
            consumed = sum(1 for x in want if x["step"] <= step)
            if doc.get("pairs_consumed") != consumed:
                out.append(f"{Path(d).name} cut.json pairs_consumed is {doc.get('pairs_consumed')!r}, but the steps up "
                           f"to {step} applied {consumed} pairs")
            last = max(last, step)
        if out:
            return [], out
        return [x for x in samples if x["step"] <= last], []

    def consumption_problems(self, stage_out: Path, data: Path, cuts: list) -> list[str]:
        """Why the observed log cannot give the per-pair consumption up to the last cut."""
        return self._ledger_plan(stage_out, data, cuts)[1]

    def consumption(self, stage_out: Path, data: Path, cuts: list, *, run_id: str = "",
                    attempt_id: str = "0") -> dict[str, Any]:
        """Precision `row`, derivation `observed` (MM-23 with a sample log): one ledger record per samples.jsonl line
        of the steps up to the last cut; `data_ref.row` = the line's `row` (SHA-256 of the pairs.jsonl line), shard
        pairs.jsonl, `sample_id` = H(run, attempt, epoch, step, pair_line) (NUL-joined text); `rewards`, `seed_inputs`
        and `sampler_config_hash` null (the pairs were ranked before training; DPO samples nothing)."""
        from gatedtrain.ledger import WRITER_ID, check_precision

        plan, bad = self._ledger_plan(stage_out, data, cuts)
        if bad:
            raise ValueError("; ".join(bad))
        check_precision("row", "observed")
        snapshot = manifest_hash(data)
        rows = []
        for x in plan:
            key = "\x00".join(str(v) for v in (run_id, attempt_id, x["epoch"], x["step"], x["pair_line"]))
            rows.append({
                "run_id": run_id, "attempt_id": attempt_id, "step": x["step"],
                "sample_id": "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest(),
                "data_ref": {"snapshot": snapshot, "shard": PAIRS, "row": x["row"], "mixture": None, "component": None},
                "parent_ref": None, "outcome": "consumed", "reason": None, "rule_version": None, "gen_version": None,
                "reward_epoch": None, "rewards": None, "sampler_config_hash": None, "seed_inputs": None, "chunk": None,
                "precision": "row", "derivation": "observed", "writer_id": WRITER_ID,
            })
        last = max(step for step, _ in cuts)
        return {"precision": "row", "rows": rows, "derivation": "observed",
                "reason": (f"{SAMPLES} logs every pair of each optimizer step before the step, exactly train_dpo.py's "
                           f"schedule (re-derived) (MM-23, MM-27): {len(rows)} records of steps 1 .. {last}, "
                           f"data_ref.row = SHA-256 of the {PAIRS} line, derivation observed; no pair is filtered")}

    def training_payloads(self, stage_out: Path, base: Path, cuts: list, *,
                          data: Path | None = None) -> tuple[set[str] | None, str | None]:
        """The `text` of every training item whose pair the run applied up to the last cut (EXT-107). The texts come
        from the runtime's items-hard-train, which must be the item set pairs_meta.json names by SHA-256."""
        import os

        from .rows import RUNTIME, RUNTIME_ENV

        if data is None:
            return None, "the spec's pairs directory cannot be read"
        meta, why = _read_json(Path(data) / PAIRS_META)
        if not isinstance(meta, dict):
            return None, f"{PAIRS_META}: {why or 'not an object'}"
        items_file = Path(os.environ.get(RUNTIME_ENV) or RUNTIME) / "items-hard-train" / "items.jsonl"
        try:
            got = file_sha256(items_file)
        except OSError as exc:
            return None, f"{items_file} cannot be read ({exc.__class__.__name__})"
        if got != meta.get("items_sha256"):
            return None, (f"{items_file} is {got}, not the training item set {meta.get('items_sha256')!r} that "
                          f"{PAIRS_META} names")
        rows, why = read_jsonl(items_file)
        if rows is None:
            return None, why
        text_of = {r.get("item"): r.get("text") for r in rows}
        plan, bad = self._ledger_plan(stage_out, data, cuts)
        if bad:
            return None, "; ".join(bad)
        out = {text_of.get(x["item"]) for x in plan}
        if None in out:
            return None, "an applied pair names an item that is not in the training item set"
        return out, None

    # --- pins and checks (MM-21, MM-29, MM-33) ---

    def framework_identity(self, stage_out: Path, launch: dict) -> dict[str, Any]:
        """`torch==<v>` in env-freeze.txt equals the launch record's framework; the freeze pins transformers and
        numpy (MM-33). Raises ValueError with the reason."""
        freeze = Path(stage_out) / FREEZE_FILE
        if not _plain_file(freeze):
            raise ValueError(f"{freeze} is missing")
        versions: dict[str, str] = {}
        for line in text_lines(freeze.read_text(encoding="utf-8")):
            line = line.strip()
            if "==" in line and not line.startswith("#"):
                name, _, version = line.partition("==")
                versions[name.strip().lower().replace("_", "-")] = version.strip()
        missing = [p for p in PINS if p not in versions]
        if missing:
            raise ValueError(f"{freeze} has no `<name>==<version>` line for {', '.join(missing)} (pins of {NAME})")
        version = versions[FRAMEWORK_PACKAGE]
        fw = launch.get("framework") if isinstance(launch.get("framework"), dict) else {}
        if fw.get("name") != FRAMEWORK_PACKAGE or fw.get("version") != version:
            raise ValueError(f"{freeze} pins {FRAMEWORK_PACKAGE}=={version}, but the launch record claims "
                             f"framework {fw.get('name')!r} {fw.get('version')!r}")
        return {"name": FRAMEWORK_PACKAGE, "version": version, "freeze_sha256": file_sha256(freeze)}

    def script_problems(self, stage_out: Path, launch: dict) -> list[Problem]:
        """EXT-83: env-runtime.json `script_sha256` equals the record's."""
        return runtime_script_problems(stage_out, launch)

    def argv_problems(self, argv: Any, *, profile: str | None = None) -> list[str]:
        """ext-launch, --recheck and ext-collect. argv[1] is train_dpo.py or eval_arm.py of the runtime directory (the
        pinned interpreter's). train_dpo.py: `--profile` is the launch profile, `--model-dir` is the runtime's base
        directory and `--llm` is absent (the initial and reference weights are the base's llm.pt). eval_arm.py:
        `--item-set` names a registered item set of the runtime, `--llm` is base, rl or the cut mounted at /inputs/cut,
        and `--limit` is refused under --profile full (a gate arm covers the whole item set)."""
        out, script, parsed = runtime_argv_problems(argv, profile=profile,
                                                    options_of={TRAIN_SCRIPT: TRAIN_OPTIONS, EVAL_SCRIPT: EVAL_OPTIONS},
                                                    eval_script=EVAL_SCRIPT, default_base_rel=DEFAULT_BASE_REL,
                                                    rule="cosyvoice_dpo argv")
        if script is None:
            return out
        root = str(Path(argv[0]).parent.parent.parent)
        seen = {opt: value for opt, value, _ in parsed}
        if script == TRAIN_SCRIPT:
            if seen.get("--profile") != profile:
                out.append(f"{TRAIN_SCRIPT} --profile {seen.get('--profile')!r} is not the launch profile {profile!r} "
                           "(run_args.json and every cut.json record it)")
            if seen.get("--model-dir") != f"{root}/{DEFAULT_BASE_REL}":
                out.append(f"{TRAIN_SCRIPT} --model-dir {seen.get('--model-dir')!r} is not the runtime's base "
                           f"{root}/{DEFAULT_BASE_REL}")
            if "--llm" in seen:
                out.append(f"{TRAIN_SCRIPT} --llm starts from other weights than the base's llm.pt, which ext-collect "
                           "refuses (run_args.json llm_init_is_base)")
        elif script == EVAL_SCRIPT:
            if seen.get("--item-set") not in EVAL_ITEM_SETS:
                out.append(f"{EVAL_SCRIPT} --item-set {seen.get('--item-set')!r} is not one of {list(EVAL_ITEM_SETS)}")
            if seen.get("--llm") not in EVAL_LLMS:
                out.append(f"{EVAL_SCRIPT} --llm {seen.get('--llm')!r} is not one of {list(EVAL_LLMS)} (a cut is "
                           "mounted read-only at /inputs/cut)")
        return out

    def mount_binding_problems(self, mounts: Any, *, base: Path, data: Path, base_field: str, data_field: str,
                               flag: str = "--launch-record", argv: Any = None) -> list[Problem]:
        """`ext-launch --spec` for a training launch: the spec's base directory and its pairs directory lie inside
        read-only mounts. An evaluation launch reads neither the pairs nor (as training) the base's llm.pt."""
        from gatedtrain.ext.sdk import ro_table, within

        if isinstance(argv, list) and len(argv) > 1 and isinstance(argv[1], str) and not argv[1].endswith(
                "/" + TRAIN_SCRIPT):
            return []
        table = ro_table(mounts)
        out = []
        if not any(within(str(Path(base).resolve()), host) for _, host in table):
            out.append(Problem("EXT-BASE", base_field, f"{flag} mounts no read-only directory that is or holds the "
                               f"spec's base directory {base}", self.BASE_FIX))
        if not any(within(str(Path(data).resolve()), host) for _, host in table):
            out.append(Problem("EXT-DATA", data_field, f"{flag} mounts no read-only directory that is or holds the "
                               f"spec's pairs directory {data}", self.DATA_FIX))
        return out

    def _base_files(self, base: Path) -> dict[str, str]:
        """{relative path: "sha256:<hex>"} of the base files the LLM is built from (base_ref.json `files` shape)."""
        out = {}
        for e in self._base_entries(base):
            p = e["path"]
            if p in BASE_BOUND_FILES or any(p.startswith(d + "/") for d in BASE_BOUND_DIRS):
                out[p] = "sha256:" + e["sha256"]
        return out

    def checks(self, *, last_cut: Path, base: Path, launch: dict, base_field: str, stage_out: Path | None = None,
               data: Path | None = None, data_field: str = "data", **_: Any) -> list[Problem]:
        """Base binding, data binding, seed, profile, script, and the run finished with its last cut."""
        from gatedtrain.ext.sdk import container_file

        stage_out = Path(stage_out) if stage_out is not None else Path(last_cut).parent
        out: list[Problem] = []
        args, why = read_run_args(stage_out)
        if args is None:
            return [Problem("EXT-BASE", base_field, f"{why}, so the run cannot be bound to the spec's base",
                            self.BASE_FIX)]
        mounts = launch.get("mounts")
        complete = self.find_cuts(stage_out)[0]
        cuts = [d for _, d in complete] or [Path(last_cut)]
        # Base (MM-21): the run built the LLM from the spec's base through a read-only mount, re-hashed against
        # MODELS.json, and started from its llm.pt; every cut's base_ref.json names those files.
        md = args.get("model_dir")
        hit = container_file(f"{md.rstrip('/')}/{WEIGHTS}", mounts) if isinstance(md, str) and md else None
        if hit is None or Path(hit[1]).parent.resolve() != Path(base).resolve():
            out.append(Problem("EXT-BASE", base_field, f"run_args.json model_dir {md!r} is not, through a read-only "
                               f"mount, the spec's base directory {base} (MM-21)", self.BASE_FIX))
        rb = args.get("base_ref") if isinstance(args.get("base_ref"), dict) else {}
        if rb.get("verified") is not True:
            out.append(Problem("EXT-BASE", base_field, f"run_args.json base_ref.verified is {rb.get('verified')!r}: the "
                               "run did not check the base files against models/MODELS.json", self.BASE_FIX))
        if rb.get("llm_init_is_base") is not True:
            out.append(Problem("EXT-BASE", base_field, "run_args.json base_ref.llm_init_is_base is not true: the run "
                               "started from other weights than the base's llm.pt (--llm)", self.BASE_FIX))
        try:
            files = self._base_files(base)
        except (OSError, ValueError):
            files = None                                  # collect reports an unreadable base (EXT-BASE)
        if files is not None and args.get("llm_init_sha256") != files.get(WEIGHTS):
            out.append(Problem("EXT-BASE", base_field, f"run_args.json llm_init_sha256 {args.get('llm_init_sha256')!r} "
                               f"is not the base's {WEIGHTS} {files.get(WEIGHTS)} (the reference model is the initial "
                               "LLM)", self.BASE_FIX))
        for d in cuts:
            ref, why = _read_json(Path(d) / BASE_REF)
            if not isinstance(ref, dict):
                out.append(Problem("EXT-BASE", base_field, f"{Path(d).name}/{BASE_REF}: {why or 'not an object'}",
                                   self.BASE_FIX))
                continue
            rev = ref.get("revision")
            if not (isinstance(rev, str) and REVISION_RE.fullmatch(rev)):
                out.append(Problem("EXT-BASE", base_field, f"{Path(d).name}/{BASE_REF} revision {rev!r} is not a 40-hex "
                                   "commit (MM-18)", self.BASE_FIX))
            if files is not None and ref.get("files") != files:
                out.append(Problem("EXT-BASE", base_field, f"{Path(d).name}/{BASE_REF} files are not the SHA-256 of the "
                                   f"spec's base files {sorted(files)} (the run built another LLM, MM-21)",
                                   self.BASE_FIX))
            elif not (isinstance(ref.get("files"), dict) and ref.get("files_sha256") == compact_sha256(ref["files"])):
                out.append(Problem("EXT-BASE", base_field, f"{Path(d).name}/{BASE_REF} files_sha256 is not the hash of "
                                   "its files (compact sorted JSON)", self.BASE_FIX))
            if (rb.get("repo"), rb.get("revision"), rb.get("files"), rb.get("files_sha256")) != (
                    ref.get("repo"), rev, ref.get("files"), ref.get("files_sha256")):
                out.append(Problem("EXT-BASE", base_field, f"run_args.json base_ref is not {Path(d).name}/{BASE_REF}",
                                   self.BASE_FIX))
        # Data: the pairs file the run read is the spec's, through a read-only mount, with its hash and row count; the
        # candidates are the ones the pairs were made from; the reference log-probs are of the base and these pairs.
        if data is not None:
            pairs = Path(data) / PAIRS
            path = args.get("pairs_file")
            hit = container_file(path, mounts) if isinstance(path, str) else None
            pairs_sha = None
            try:
                pairs_sha = file_sha256(pairs)
            except OSError:
                pass
            if hit is None or Path(hit[1]).resolve() != pairs.resolve():
                out.append(Problem("EXT-DATA", data_field, f"run_args.json pairs_file {path!r} is not, through a "
                                   f"read-only mount, the spec's {pairs}", self.DATA_FIX))
            elif args.get("pairs_sha256") != pairs_sha:
                out.append(Problem("EXT-DATA", data_field, f"run_args.json pairs_sha256 {args.get('pairs_sha256')!r} "
                                   f"is not the SHA-256 of {pairs}", self.DATA_FIX))
            meta, why = _read_json(Path(data) / PAIRS_META)
            if not isinstance(meta, dict):
                out.append(Problem("EXT-DATA", data_field, f"{PAIRS_META}: {why or 'not an object'}", self.DATA_FIX))
            else:
                if args.get("candidates_jsonl_sha256") != meta.get("candidates_sha256"):
                    out.append(Problem("EXT-DATA", data_field, f"run_args.json candidates_jsonl_sha256 "
                                       f"{args.get('candidates_jsonl_sha256')!r} is not {PAIRS_META} candidates_sha256 "
                                       f"{meta.get('candidates_sha256')!r}: the run read other candidates than the "
                                       "pairs were made from", self.DATA_FIX))
                if meta.get("pairs_sha256") not in (None, pairs_sha):
                    out.append(Problem("EXT-DATA", data_field, f"{PAIRS_META} pairs_sha256 is not the SHA-256 of "
                                       f"{pairs}", self.DATA_FIX))
            ref_lines, why = read_jsonl(stage_out / REF_LOGPS)
            want_head = {"meta": {"llm_sha256": files.get(WEIGHTS) if files else None, "pairs_sha256": pairs_sha,
                                  "amp": (args.get("settings") or {}).get("amp")}}
            if not ref_lines or ref_lines[0] != want_head:
                out.append(Problem("EXT-DATA", data_field, f"{REF_LOGPS} {why or 'first line'} is not the reference "
                                   f"log-probs of the base's {WEIGHTS} on these pairs ({want_head})", self.DATA_FIX))
        # Seed, profile, script.
        claimed, field = launch.get("training_seed"), "--launch-record.training_seed"
        seed = args.get("seed")
        s = args.get("settings") if isinstance(args.get("settings"), dict) else {}
        if not (_int(seed) and seed == claimed and s.get("seed") == seed):
            out.append(Problem("EXT-LAUNCH", field, f"the launch record claims training_seed {claimed!r}, but "
                               f"run_args.json seed is {seed!r} (settings.seed {s.get('seed')!r}) (MM-29)",
                               "launch with the seed the record names"))
        if args.get("profile") != launch.get("profile"):
            out.append(Problem("EXT-PROFILE", "--launch-record.profile", f"run_args.json profile is "
                               f"{args.get('profile')!r}, the launch record's {launch.get('profile')!r}",
                               "collect the run launched with this record's profile"))
        if args.get("dry_run") is not False:
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json dry_run is {args.get('dry_run')!r}, not "
                               "false", "collect a real training run"))
        if args.get("script_sha256") != launch.get("script_sha256"):
            out.append(Problem("EXT-LAUNCH", "--launch-record.script_sha256", f"run_args.json script_sha256 "
                               f"{args.get('script_sha256')!r} is not the launch record's "
                               f"{launch.get('script_sha256')!r} (EXT-83)",
                               "collect the run launched with this record's training script"))
        for d in cuts:
            doc, _ = read_cut_marker(d)
            doc = doc or {}
            if doc.get("seed") != claimed:
                out.append(Problem("EXT-LAUNCH", field, f"{Path(d).name}/cut.json seed is {doc.get('seed')!r}, not the "
                                   f"launch record's training_seed {claimed!r} (MM-29)",
                                   "collect the cuts of the run launched with this record"))
            if doc.get("script_sha256") != launch.get("script_sha256"):
                out.append(Problem("EXT-LAUNCH", "--launch-record.script_sha256", f"{Path(d).name}/cut.json "
                                   f"script_sha256 is {doc.get('script_sha256')!r}, not the launch record's "
                                   f"{launch.get('script_sha256')!r} (EXT-83)",
                                   "collect the cuts of the run launched with this record's training script"))
            if doc.get("profile") != launch.get("profile"):
                out.append(Problem("EXT-PROFILE", "--launch-record.profile", f"{Path(d).name}/cut.json profile is "
                                   f"{doc.get('profile')!r}, the launch record's {launch.get('profile')!r}",
                                   "collect the run launched with this record's profile"))
        # The run finished: its last cut is the last scheduled step, and it saved every scheduled cut.
        n_steps, every = args.get("n_steps"), s.get("save_every")
        m = CHECKPOINT_DIR_RE.match(Path(last_cut).name)
        last = int(m.group(1)) if m else None
        if not (_int(n_steps) and n_steps == last):
            out.append(Problem("EXT-STAGE", "--stage-out", f"the last cut is step {last}, but run_args.json n_steps is "
                               f"{n_steps!r}: the run did not save its last step",
                               "collect a run that finished and saved its last step"))
        elif _int(every):
            want = sorted({k for k in range(every, n_steps + 1, every)} | {n_steps}) if every > 0 else [n_steps]
            got = [step for step, _ in complete]
            if got != want:
                out.append(Problem("EXT-STAGE", "--stage-out", f"the complete cuts are steps {got}, not the scheduled "
                                   f"{want} (save_every {every}, n_steps {n_steps})",
                                   "collect the run's output directory as train_dpo.py wrote it"))
        return out

    # --- arms (EXT-155) and report notes (EXT-246) ---

    def arm_problems(self, arm: Path, records: list[dict], item_rows: list[dict]) -> list[str]:
        """Every output an arm references (`outputs/<n>.json`, written by `gatedtrain_recipe_cosyvoice_dpo_tts.arms` from the
        scorer's records) names its item and the registered row's text and reference, so an evaluation of another
        item set shows here."""
        rows = {r.get("item"): r for r in item_rows}
        out: list[str] = []
        for rec in records:
            ref, item = rec.get("output_ref"), rec.get("item")
            if not (isinstance(ref, dict) and isinstance(ref.get("path"), str)) or item not in rows:
                continue                                  # no output, or an unknown item (the gate reports it)
            doc, why = _read_json(Path(arm) / ref["path"])
            row = rows[item]
            if not isinstance(doc, dict):
                out.append(f"output {ref['path']}: {why or 'not a JSON object'}")
            elif doc.get("item") != item:
                out.append(f"output {ref['path']} names item {doc.get('item')!r}, not {item!r}")
            elif doc.get("text") != row.get("text") or doc.get("reference") != (row.get("ref_text") or row.get("text")):
                out.append(f"output {ref['path']} text or reference is not the registered item's: the evaluation read "
                           "another item set than the registered one")
            if len(out) >= 8:
                break
        return out

    def report_notes(self, records_by_arm: dict[str, list[dict]]) -> list[str]:
        """Per-category mean score of each arm (the category is the item id's second field, `hard-test:<category>:..`);
        descriptive only."""
        lines = []
        cats = sorted({str(r["item"]).split(":")[1] for recs in records_by_arm.values() for r in recs
                       if str(r.get("item", "")).count(":") >= 2})
        for cat in cats:
            parts = []
            for arm in ("baseline", "candidate"):
                vals = [r["score"] for r in records_by_arm.get(arm, []) if r.get("score") is not None
                        and str(r["item"]).split(":")[1:2] == [cat]]
                parts.append(f"{arm} {sum(vals) / len(vals):.4f} (n {len(vals)})" if vals else f"{arm} none")
            lines.append(f"Category {cat} (descriptive): " + ", ".join(parts) + ".")
        return lines
