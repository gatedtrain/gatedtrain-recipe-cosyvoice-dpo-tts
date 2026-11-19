"""The runtime scripts and item sets of this checkout, and what the 2026-11-17 gate ran with: the LLM weights, the
source pins and the scripts' hashes the run recorded.

The values are the hashes the run recorded (`gen_meta.json`, `score_meta.json`, `train_meta.json`, the gate result)
and the gate registration names (results/2026-11-17/gate/registration-v2.yaml `code`, `arms`, `eval_distribution`),
except train_dpo.py (rewritten to write the gatedtrain launcher layout, same training computation; the published run's
hash is `PUBLISHED_TRAIN_DPO_SHA256`), eval_arm.py (new) and the three scripts edited for the rename to gatedtrain 0.3.0
(train_dpo.py, eval_arm.py: names in docstrings and the record schema ids they write; fetch_models.py: its User-Agent;
the hashes before are `RERUN_SCRIPTS` and `PUBLISHED_FETCH_MODELS_SHA256`). `tests/test_pins.py` checks them against
the files of this checkout. A script whose bytes change gets a new pin here and a note in RESULT.md.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from gatedtrain.ext.rowkit import runtime_dir

# The runtime directory: this repository's runtime/, or the override.
RUNTIME_ENV = "GATEDTRAIN_RECIPE_COSYVOICE_DPO_TTS_RUNTIME"
RUNTIME = runtime_dir(RUNTIME_ENV, __file__)

# Runtime scripts (path relative to the runtime directory -> SHA-256). The four the gate pins are marked.
SCRIPTS = {
    "gen_tts.py": "sha256:8f4d5c0796e21afdaa3e45e19f5f76188939af9209e11c0d4e23f962e737d4fc",          # registered
    "eval/score_tts.py": "sha256:4a5f29d65cad7a9a22d1e0edd970c22a47c3d33193726ab67fb4b72600549d34",   # registered (v2)
    "eval/wer_canon.py": "sha256:54610fa0019d9955846d0fcaabc3774828e6c1b68c86f605831f778e7feb7b20",   # registered
    "analyze_gate.py": "sha256:7b815399aa6f01e6dc330fbf852bcb288fd3aa454cac6d6af47c78b026fa9c9a",     # gate result
    "train_dpo.py": "sha256:bc2daa7c6540e806f6752569c48e491de72dc91bff54f41d0efe0ae05cc8e032",      # gatedtrain layout
    "eval_arm.py": "sha256:319b7f26a2efe1119e98cabe8cad10e982b67e29ffd9547aad06eba900802b81",       # gatedtrain arm launch
    "sample_candidates.py": "sha256:1e5921f5c2bf123fef2381bad5f0452c9155fc7479a9badb208bd826453f9931",
    "eval/score_candidates.py": "sha256:2e99edbf2b4f5ba6b72c01e048fed75732ded3a449096f24800914f772e795c8",
    "make_pairs.py": "sha256:67ed70b4a960e576601f880c385e09567d3920f08609ecdd78833c7ae8bb8d2d",
    "make_hard_items.py": "sha256:8bf0d9cf1900422db125f5a9cd0078c9f372d7a7ad073a5cce4b9db44cea9773",
    "make_items.py": "sha256:5aa8c0e0f86e00250bb6a795620d1c3037f7d420f3903e49291921a371f387dc",
    "fetch_models.py": "sha256:4626dc28e8dfdd18cf6011f9fab6d0ef82a169ee27812eaa8bc8e4fb6b54ae44",
}
# The train_dpo.py of the published run (dpo-train-k6, train_meta.json `script_sha256`): the same training computation,
# writing llm_dpo.step<N>.pt and llm_dpo.pt instead of the gatedtrain layout; in this repository's history (the initial
# commit), replaced by the current one.
PUBLISHED_TRAIN_DPO_SHA256 = "sha256:e19f6cc8c28b6deb9c87c4b0475c2061379e3d9c28133cc8c96b3a247dbd8f2e"
# The scripts the 2026-11-18 rerun ran (its launch records' `script_sha256`, under ptc-hub 0.2.0), before their
# re-pin for the rename to gatedtrain 0.3.0; in this repository's history.
RERUN_SCRIPTS = {
    "train_dpo.py": "sha256:9813b91417e3336e86f1f61f64fbe796003c3dfd8e50a526901b30f4f0ae2e67",
    "eval_arm.py": "sha256:a07b851a1edc5147cf2e322b8dcbdd90e70ddac840cb99e2feda77840b3f5452",
}
# The fetch_models.py the published runs' downloads used (User-Agent `ptc-area11-fetch`), before the rename.
PUBLISHED_FETCH_MODELS_SHA256 = "sha256:8cd1d28c8c82964e0281ad26cbc2a77164acbdf64049a1f34b1d53d6a08ff7fc"
# The score_tts.py of registration v1 (sha256 ac51251b...), replaced by the amendment; not in this repository.
SCORE_TTS_V1_SHA256 = "sha256:552ee513371374819ce176bca93de0962c9f69398e562d5502a2685837a28ebf"

# Item sets: SHA-256 of <set>/items.jsonl (what the scripts and the registration record).
ITEM_SETS = {
    "items-hard-test": "sha256:33cf0b871919d49889fd8def5af33f58a532b82dc7871bbf952931e13c98d090",    # registered
    "items-hard-train": "sha256:bcd7909438786169c3a0023cf2ab5a0553b09b765269cee369c7e6c5abad4488",
    "items-hard-dev": "sha256:b9ac4220b723e351f5f49cac537d95cd873c4a9f05ec3a840a576393af1cb1a4",
    "items-hard-pilot": "sha256:c68e12fa7c7826090c4bb2f20a431d91af2a1693a89a4e756841cc47f1a07e7c",
    "items-smoke": "sha256:6136abb55a519f731c27edc6769e590f3a08bc88e884999e3007f1bbecab4e46",
}

# LLM weights of the three arms (state dicts; hashed, never loaded here).
LLM_SHA256 = {
    "base": "sha256:69f43bd545131c30e98947fb360ea8b4dc9916d8e83dded7757c7ea4f5a24970",        # models/.../llm.pt
    "candidate": "sha256:65cd9a45463fa4af9d24e6ea2da61dae570a634978cc181feceb9e873346f794",   # dpo-train-k6/llm_dpo.pt
    "rl": "sha256:74d34b01a80c7154670ae75ac372d1b1712c78bceae9f467eb9f1f6f61ec764f",          # models/.../llm.rl.pt
}

# Sources (runtime/fetch_cosyvoice.sh, runtime/fetch_models.py).
COSYVOICE_COMMIT = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
MATCHA_TTS_COMMIT = "dd9105b34bf2be2230f4aa1e4769fb586a3c824e"
MODEL_REVISIONS = {
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512": "29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
    "openai/whisper-large-v3-turbo": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
    "Qwen/Qwen3-ASR-1.7B": "7278e1e70fe206f11671096ffdd38061171dd6e5",
    "speechbrain/spkrec-ecapa-voxceleb": "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def script_problems(runtime: str | Path = RUNTIME) -> list[str]:
    """Every pinned script or item set of `runtime` whose SHA-256 is not its pin; [] when all match."""
    out = []
    for rel, want in {**SCRIPTS, **{f"{k}/items.jsonl": v for k, v in ITEM_SETS.items()}}.items():
        p = Path(runtime) / rel
        got = file_sha256(p) if p.is_file() else "missing"
        if got != want:
            out.append(f"{rel}: {got}, pinned {want}")
    return out
