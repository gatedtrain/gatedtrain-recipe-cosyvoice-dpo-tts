"""`gatedtrain ext-launch` requests of this recipe (runtime/README.md "Launch request facts"), written as YAML.

Host side, no GPU. Every host path comes from the arguments, the runtime directory (`GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RUNTIME`
or this checkout's runtime/) and its venv; nothing here names a machine.

- `train`: train_dpo.py on the reused pairs, candidates and reference log-probs (each mounted read-only under /inputs),
  the published settings (sigmoid DPO, beta 0.1, lr 1e-6, 2 epochs, micro-batch 2 x accumulation 8, warm-up 10, clip
  1.0, bf16, seed 11, a cut every 26 steps); `--profile smoke` stops after SMOKE_STEPS steps with a cut every
  SMOKE_SAVE_EVERY; `--max-steps N` (full) is the retraining to a dev-selected step N;
- `eval`: eval_arm.py (gen_tts.py then score_tts.py) of `base`, `rl` or a cut (mounted read-only at /inputs/cut) on an
  item set of the runtime; `--limit` for the smoke baseline only.

usage: python -m gatedtrain_recipe_cosyvoice_dpo_tts.launch train --profile smoke|full --label L --output-dir DIR
           --pairs DIR --candidates DIR --ref-cache FILE --image IMAGE --host-work DIR [--gpu-est-h H] > request.yaml
       python -m gatedtrain_recipe_cosyvoice_dpo_tts.launch eval --profile smoke|full --label L --output-dir DIR
           --llm base|rl|CUT_DIR --item-set items-hard-test --image IMAGE --host-work DIR [--limit N] [--gpu-est-h H]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from .rows import OUT_NAME, RUNTIME, host_rows

AREA = 11
SEED = 11
SMOKE_STEPS = 6
SMOKE_SAVE_EVERY = 3
TRAIN_SETTINGS = ["--beta", "0.1", "--lr", "1e-6", "--epochs", "2", "--micro-batch", "2", "--grad-accum", "8",
                  "--warmup-steps", "10", "--clip", "1.0", "--seed", str(SEED), "--amp", "bf16"]
ENV = {"HF_HUB_OFFLINE": "1", "HOME": "/tmp/home", "XDG_CACHE_HOME": "/tmp/cache", "PYTHONDONTWRITEBYTECODE": "1",
       "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
BASE_REL = "models/fun-cosyvoice3-0.5b-2512"


def torch_version(runtime: str | Path = RUNTIME) -> str:
    """The runtime venv's torch version as importlib.metadata reports it (what env-freeze.txt records)."""
    hits = sorted((Path(runtime) / ".venv").glob("lib/python3*/site-packages/torch-*.dist-info"))
    if len(hits) != 1:
        raise SystemExit(f"{runtime}/.venv holds {len(hits)} torch distributions, not one")
    return hits[0].name[len("torch-"):-len(".dist-info")]


def _identity(path: str) -> dict[str, str]:
    return {"host": path, "container": path, "mode": "ro"}


def common(*, label: str, output_dir: Path, image: str, host_work: Path, gpu_est_h: float,
           runtime: str = RUNTIME, runtime_container: str | None = None) -> dict[str, Any]:
    """The keys every request of this recipe shares; the runtime is mounted read-only at its own host path
    (`runtime_container` None) or at `runtime_container` (tests: a host path under /tmp cannot be an identity mount
    beside the /tmp tmpfs). The interpreter roots are those of `runtime`'s own venv."""
    roots = [r for r in host_rows(runtime, OUT_NAME)["input_roots"] if not r.startswith("{output_root}")]
    py_roots = [r for r in roots if r != runtime]
    return {"area": AREA, "label": label, "host_work": str(host_work),
            "runtime": {"kind": "venv_in_container", "image": image, "venv_lock": f"{runtime}/uv.lock"},
            "framework": {"name": "torch", "version": torch_version(runtime), "commit": None},
            "training_seed": SEED, "gpus": True, "network": "none", "network_reason": None,
            "user": f"{os.getuid()}:{os.getgid()}", "env": dict(ENV),
            "mounts": [_identity(r) for r in py_roots] + [{"host": runtime, "container": runtime_container or runtime,
                                                           "mode": "ro"}],
            "tmpfs": ["/tmp"], "output_dir": str(output_dir), "gpu_est_h": gpu_est_h, "argv": []}


def train_request(*, profile: str, label: str, output_dir: Path, pairs: Path, candidates: Path, ref_cache: Path,
                  image: str, host_work: Path, gpu_est_h: float, runtime: str = RUNTIME,
                  runtime_container: str | None = None, max_steps: int | None = None) -> dict[str, Any]:
    req = common(label=label, output_dir=output_dir, image=image, host_work=host_work, gpu_est_h=gpu_est_h,
                 runtime=runtime, runtime_container=runtime_container)
    rt = runtime_container or runtime
    ref_cache = Path(ref_cache)
    req["mounts"] += [{"host": str(pairs), "container": "/inputs/pairs", "mode": "ro"},
                      {"host": str(candidates), "container": "/inputs/cand", "mode": "ro"},
                      {"host": str(ref_cache.parent), "container": "/inputs/ref", "mode": "ro"},
                      {"host": str(output_dir), "container": "/out", "mode": "rw"}]
    if profile == "smoke":
        steps = ["--max-steps", str(SMOKE_STEPS), "--save-every", str(SMOKE_SAVE_EVERY)]
    else:                       # --max-steps: the retraining to a dev-selected step (select.py)
        steps = (["--max-steps", str(max_steps)] if max_steps is not None else []) + ["--save-every", "26"]
    req["argv"] = [f"{rt}/.venv/bin/python", f"{rt}/train_dpo.py", "--model-dir", f"{rt}/{BASE_REL}",
                   "--candidates", "/inputs/cand", "--pairs", "/inputs/pairs", "--ref-cache",
                   f"/inputs/ref/{ref_cache.name}", "--out", "/out", "--profile", profile, *TRAIN_SETTINGS, *steps]
    return req


def eval_request(*, profile: str, label: str, output_dir: Path, llm: str, item_set: str, image: str, host_work: Path,
                 gpu_est_h: float, limit: int | None = None, runtime: str = RUNTIME,
                 runtime_container: str | None = None) -> dict[str, Any]:
    if profile == "full" and limit is not None:
        raise SystemExit("--limit is for the smoke baseline; a full arm covers the whole item set")
    req = common(label=label, output_dir=output_dir, image=image, host_work=host_work, gpu_est_h=gpu_est_h,
                 runtime=runtime, runtime_container=runtime_container)
    rt = runtime_container or runtime
    if llm in ("base", "rl"):
        llm_arg = llm
    else:
        req["mounts"].append({"host": str(llm), "container": "/inputs/cut", "mode": "ro"})
        llm_arg = "/inputs/cut"
    req["mounts"].append({"host": str(output_dir), "container": "/out", "mode": "rw"})
    req["argv"] = [f"{rt}/.venv/bin/python", f"{rt}/eval_arm.py", "--llm", llm_arg, "--item-set", item_set,
                   "--out", "/out", "--seed", str(SEED)] + (["--limit", str(limit)] if limit is not None else [])
    return req


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gatedtrain_recipe_cosyvoice_dpo_tts.launch")
    sub = ap.add_subparsers(dest="kind", required=True)
    for kind in ("train", "eval"):
        p = sub.add_parser(kind)
        p.add_argument("--profile", choices=("smoke", "full"), required=True)
        p.add_argument("--label", required=True)
        p.add_argument("--output-dir", required=True)
        p.add_argument("--image", required=True, help="the pinned image, <name>@sha256:<digest>")
        p.add_argument("--host-work", required=True)
        p.add_argument("--gpu-est-h", type=float, default=0.25)
        if kind == "train":
            p.add_argument("--pairs", required=True)
            p.add_argument("--candidates", required=True)
            p.add_argument("--ref-cache", required=True)
            p.add_argument("--max-steps", type=int, help="full profile: stop after this step (dev-selected retraining)")
        else:
            p.add_argument("--llm", required=True)
            p.add_argument("--item-set", required=True)
            p.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    if a.kind == "train":
        req = train_request(profile=a.profile, label=a.label, output_dir=Path(a.output_dir), pairs=Path(a.pairs),
                            candidates=Path(a.candidates), ref_cache=Path(a.ref_cache), image=a.image,
                            host_work=Path(a.host_work), gpu_est_h=a.gpu_est_h, max_steps=a.max_steps)
    else:
        req = eval_request(profile=a.profile, label=a.label, output_dir=Path(a.output_dir), llm=a.llm,
                           item_set=a.item_set, image=a.image, host_work=Path(a.host_work), gpu_est_h=a.gpu_est_h,
                           limit=a.limit)
    sys.stdout.write(yaml.safe_dump(req, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
