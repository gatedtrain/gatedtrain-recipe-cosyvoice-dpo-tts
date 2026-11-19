"""Registry rows of adapter `cosyvoice_dpo`: DPO of Fun-CosyVoice3-0.5B's speech-token LLM (area "TTS").

The training row, the two evaluator rows and the adapter class, loaded by `gatedtrain.ext.registry` through the
`gatedtrain.adapters` entry point (MULTIMODAL-SPEC section 6). Values are read from this repository's runtime
(`runtime/train_dpo.py`, `runtime/eval_arm.py`, `runtime/eval/score_tts.py`, `runtime/README.md`).
"""

from __future__ import annotations

from typing import Any

import os
from pathlib import Path

from gatedtrain.ext.rowkit import CONTAINER_PATHS, python_tree, runtime_dir

from . import pins

NAME = "cosyvoice_dpo"
TRAINER_ID = "dpo.cosyvoice3_llm@0.1"
WER_ID = "tts.wer_qwen_canon@0.1"
SIM_ID = "tts.sim_prompt@0.1"
# The runtime directory: this repository's runtime/ (scripts, both venvs, models, data-source, item sets, the CosyVoice
# source; mounted read-only at its own host path), or the override; runs and their reused inputs lie under the host's
# output root as `{output_root}/<OUT_NAME>`.
RUNTIME_ENV = pins.RUNTIME_ENV
OUT_NAME = "cosyvoice-dpo-tts"
RUNTIME = runtime_dir(RUNTIME_ENV, __file__)


def host_rows(runtime: str, out_name: str) -> dict:
    """The host-path keys of the training row (EXT-79, EXT-93, EXT-113), as `rowkit.venv_runtime_rows` gives them,
    for a venv whose `home` may be a minor-version alias of the interpreter tree (uv's `cpython-3.12-<platform>`, a
    link to `cpython-3.12.16-<platform>`): both the alias and the resolved tree are interpreter trees (the
    interpreter resolves into the latter, where the credential file-name scan also walks), their site-packages are
    skipped, and the evaluator environment's venv (eval/.venv) is skipped as the main one is."""
    tree = python_tree(runtime)
    trees: list[str] = []
    version = None
    if tree:
        trees = list(dict.fromkeys([tree, os.path.realpath(tree)]))
        try:
            cfg = (Path(runtime) / ".venv" / "pyvenv.cfg").read_text(encoding="utf-8")
            info = [v.strip() for k, _, v in (line.partition("=") for line in cfg.splitlines())
                    if k.strip() == "version_info"]
            version = ".".join(info[0].split(".")[:2]) if info else None
        except OSError:
            version = None
    roots = list(dict.fromkeys(str(Path(t).parent) for t in trees))
    skip = [f"{runtime}/.venv", f"{runtime}/eval/.venv"]
    if version:
        skip += [f"{t}/lib/python{version}/site-packages" for t in trees]
    return {"input_roots": tuple(roots + [runtime, "{output_root}/" + out_name]), "scan_skip": tuple(skip),
            "interpreter_trees": tuple(trees), "interpreter": f"{runtime}/.venv/bin/python"}


HOST_ROWS = host_rows(RUNTIME, OUT_NAME)

TRAINER_ROWS: dict[str, dict[str, Any]] = {}
EVALUATOR_ROWS: dict[str, dict[str, Any]] = {}

# Full fine-tune of the CosyVoice3 LLM (Qwen2 body, embeddings, speech decoder) with the runtime's own sigmoid DPO loop
# (train_dpo.py: plain torch, no trainer library); the flow decoder and the vocoder are the base's. Cuts by the
# launcher's own writer (train_dpo.py save_cut, cut.json last); consumption observed per pair (samples.jsonl).
TRAINER_ROWS[NAME] = {
    "uses": (TRAINER_ID,),
    "pins": ("torch", "transformers", "numpy"),                       # from env-freeze.txt (MM-33)
    "outputs": ("checkpoint-<step>/ cuts (llm.pt, base_ref.json, cut.json written last)", "run_args.json",
                "env-freeze.txt", "env-runtime.json", "samples.jsonl", "train_log.jsonl", "ref_logps.jsonl",
                "train_meta.json"),
    # The deployable part of a cut: the trained LLM state dict, which replaces the base directory's llm.pt (MM-17);
    # base_ref.json is not deployable.
    "components": ("llm.pt",),
    # Observed per-pair log (MM-23 with a sample log); a log that does not match the run refuses (no fallback).
    "consumption": {"target": "row", "fallback": None},
    "gpu_h_estimate": 0.1,                                            # the published 130-step run: 0.07 GPU-h
    "evaluators": (WER_ID, SIM_ID),
    "framework": "torch",
    "input_roots": HOST_ROWS["input_roots"],
    # The base, the pairs and the candidates lie in read-only mounts; --spec optional, checked when given (training).
    "spec_binding": False,
    # runtime/README.md "Launch request facts": every name, none a credential.
    "env_names": ("HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "TOKENIZERS_PARALLELISM",
                  "XDG_CACHE_HOME"),
    "scan_skip": HOST_ROWS["scan_skip"],
    "interpreter_trees": HOST_ROWS["interpreter_trees"],
    "interpreter": HOST_ROWS["interpreter"],
    "container_paths": CONTAINER_PATHS,
    "env_paths": ("HOME", "XDG_CACHE_HOME"),
}

# The deployment sampler every arm evaluates with (gen_tts.py; registration `sampler`): zero-shot, non-streaming,
# CosyVoice3's own sampling settings, one seed per item.
SAMPLER = {
    "decoding": "CosyVoice3LM.inference with cosyvoice3.yaml ras_sampling (top_p 0.8, top_k 25, win_size 10, tau_r 0.1)",
    "mode": "zero-shot, stream false, speed 1.0",
    "prompt": "You are a helpful assistant.<|endofprompt|> + prompt_text; prompt audio prompt_wav",
    "text_frontend": False,
    "seed_rule": "set_all_random_seed(derive_seed('tts', eval_seed, item))",
    "flow_and_vocoder": "the base directory's flow.pt and hift.pt",
    "code": f"gen_tts.py {pins.SCRIPTS['gen_tts.py']}, src/CosyVoice@{pins.COSYVOICE_COMMIT}",
}

# The item set: one hard-text sentence (or a held-out LibriTTS-R sentence) spoken in the voice of one held-out
# speaker's prompt recording (runtime/make_hard_items.py). `text` is what the model reads and the exact-overlap key.
_ITEMS = {
    "binary": False,
    "env_version": False,
    "normalizer": False,                # the text normalizer is part of the scorer (wer_canon.py), not of the model
    "payload_text_field": "text",
    "item_payload": ("text", "prompt_text", "prompt_wav", "speaker", "category"),
    # MM-4: gen_error = gen_tts.py recorded an error for the item (no audio); score_error = score_tts.py raised on the
    # item's audio, or the WER is undefined (the canonical reference is empty).
    "error_classes": ("gen_error", "score_error"),
    "framework": "torch",
    "samples": "450, 1",
    # EXT-106: the registration's `sampler` is exactly SAMPLER's keys (eval_arm.py runs gen_tts.py with it).
    "sampler_keys": tuple(sorted(SAMPLER)),
}
SCORE_TTS = pins.SCRIPTS["eval/score_tts.py"]
WER_CANON = pins.SCRIPTS["eval/wer_canon.py"]
REV = pins.MODEL_REVISIONS

EVALUATOR_ROWS[WER_ID] = {
    **_ITEMS,
    "metric": "wer_qwen_canon",
    "direction": "lower_is_better",
    "score_range": (0, None),           # an insertion-heavy transcript exceeds 1
    "score_outcomes": ("empty_transcript",),        # scored: WER 1 against a non-empty reference
    # 05 section 6.1, R15-1: another ASR (Qwen3-ASR-1.7B) than the reward's (Whisper-large-v3-turbo), never used to
    # build pairs, so T3; the same canon-1 WER formula as the reward, so construct-aligned, and the registration and
    # the report say what a pass shows.
    "tier": "T3",
    "construct_aligned": {
        "reward": ("wer_whisper_canon: canon-1 WER (wer_canon.py) of Whisper-large-v3-turbo's transcript, the ranking "
                   "of the DPO pairs (eval/score_candidates.py)"),
        "statement": ("wer_qwen_canon is the training reward's formula (canon-1 WER against the item's reference) "
                      "computed with another ASR, Qwen3-ASR-1.7B, which never scored a training candidate: a pass shows "
                      "that the reward's gain in ASR-judged reading accuracy transfers to an independent ASR on held-out "
                      "texts and voices, not a general naturalness or quality gain; the two ASRs can share biases "
                      "(05 section 6.1, R15-1)")},
    "scorer": {"name": "wer_qwen_canon",
               "version": (f"score_tts v2, Qwen3-ASR-1.7B {REV['Qwen/Qwen3-ASR-1.7B']} (qwen-asr 0.0.6, bf16, "
                           f"language English), Whisper EnglishTextNormalizer with whisper-large-v3-turbo "
                           f"{REV['openai/whisper-large-v3-turbo']} normalizer.json, canon-1 (wer_canon.py "
                           f"{WER_CANON}), jiwer 3.1.0 WER; reference ref_text else text"),
               "weights_sha256": None, "code_sha256": SCORE_TTS},
}

EVALUATOR_ROWS[SIM_ID] = {
    **_ITEMS,
    "metric": "sim_prompt",
    "direction": "higher_is_better",
    "score_range": (-1, 1),             # cosine similarity
    "score_outcomes": (),
    "tier": "T3",                       # a speaker-verification model the training never saw; not the reward
    "scorer": {"name": "sim_prompt",
               "version": (f"score_tts v2, SpeechBrain ECAPA-TDNN spkrec-ecapa-voxceleb "
                           f"{REV['speechbrain/spkrec-ecapa-voxceleb']} (speechbrain 1.1.2), cosine of the embeddings "
                           "of the output and the prompt recording, both mono 16 kHz"),
               "weights_sha256": None, "code_sha256": SCORE_TTS},
}

PLUGIN = {"adapters": {NAME: "gatedtrain_recipe_cosyvoice_dpo_tts.adapter:CosyVoiceDpoAdapter"}, "trainer_rows": TRAINER_ROWS,
          "evaluator_rows": EVALUATOR_ROWS}
