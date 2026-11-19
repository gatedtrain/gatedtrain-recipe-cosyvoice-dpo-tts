"""Download the area "TTS" models and data at pinned revisions and verify every file (standard library only).

Models (Hugging Face, no token): each file of the pinned revision is downloaded to models/<name>/ and its SHA-256 is
checked against the Hub's LFS oid (git blob SHA-1 for small non-LFS files); models/MODELS.json records repo, revision,
license and each file's SHA-256 and size. Data (OpenSLR 141, LibriTTS-R, CC BY 4.0): the archives are downloaded to
data-source/ and checked against OpenSLR's md5sum.txt, then unpacked; data-source/DATA.json records them.

usage: python3 fetch_models.py [--models] [--data] [--verify]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = {
    # name: (repo, revision, license from the card metadata)
    "fun-cosyvoice3-0.5b-2512": ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
                                 "apache-2.0"),
    "whisper-large-v3-turbo": ("openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9", "mit"),
    "qwen3-asr-1.7b": ("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5", "apache-2.0"),
    "spkrec-ecapa-voxceleb": ("speechbrain/spkrec-ecapa-voxceleb", "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
                              "apache-2.0"),
}
OPENSLR = "https://www.openslr.org/resources/141"
DATA = {  # archive: md5 (OpenSLR 141 md5sum.txt, read 2026-11-17)
    "test_clean.tar.gz": "4d373d453eb96c0691e598061bbafab7",
    "dev_clean.tar.gz": "2c1f5312914890634cc2d15783032ff3",
    "train_clean_100.tar.gz": "6df668d8f5f33e70876bfa33862ad02b",
}
CHUNK = 1 << 20


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "gatedtrain-recipe-cosyvoice-dpo-tts-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def download(url: str, dest: Path, hashes: list) -> None:
    """Stream `url` to `dest` (via dest.part), updating each hashlib object in `hashes`; retries 3 times."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            part = dest.with_name(dest.name + ".part")
            req = urllib.request.Request(url, headers={"User-Agent": "gatedtrain-recipe-cosyvoice-dpo-tts-fetch"})
            with urllib.request.urlopen(req, timeout=120) as r, part.open("wb") as f:
                while True:
                    block = r.read(CHUNK)
                    if not block:
                        break
                    f.write(block)
                    for h in hashes:
                        h.update(block)
            part.rename(dest)
            return
        except OSError as exc:
            print(f"  retry {attempt + 1} for {url}: {exc}", flush=True)
            time.sleep(5)
    raise SystemExit(f"download failed: {url}")


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(p: Path) -> str:
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch_models(verify_only: bool) -> None:
    out = {}
    for name, (repo, rev, lic) in MODELS.items():
        info = get_json(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true")
        files = {}
        for s in info["siblings"]:
            rel = s["rfilename"]
            dest = HERE / "models" / name / rel
            lfs = s.get("lfs")
            if not verify_only and not dest.exists():
                print(f"{name}: {rel} ({(s.get('size') or 0) / 1e6:.1f} MB)", flush=True)
                download(f"https://huggingface.co/{repo}/resolve/{rev}/{rel}", dest, [])
            if lfs:
                got = file_sha256(dest)
                if got != lfs["sha256"]:
                    raise SystemExit(f"{name}/{rel}: sha256 {got} is not the Hub's LFS oid {lfs['sha256']}")
            elif s.get("blobId") and git_blob_sha1(dest) != s["blobId"]:
                raise SystemExit(f"{name}/{rel}: git blob sha1 is not the Hub's {s['blobId']}")
            files[rel] = {"sha256": file_sha256(dest), "size": dest.stat().st_size}
        out[name] = {"repo": repo, "revision": rev, "license": lic, "files": files}
        print(f"{name}: {len(files)} files verified", flush=True)
    (HERE / "models" / "MODELS.json").write_text(json.dumps({"models": out}, indent=1, sort_keys=True) + "\n")


def fetch_data(verify_only: bool) -> None:
    out = {}
    for arc, md5 in DATA.items():
        dest = HERE / "data-source" / arc
        if not verify_only and not dest.exists():
            print(f"LibriTTS-R {arc}", flush=True)
            download(f"{OPENSLR}/{arc}", dest, [])
        h = hashlib.md5()
        with dest.open("rb") as f:
            for block in iter(lambda: f.read(CHUNK), b""):
                h.update(block)
        if h.hexdigest() != md5:
            raise SystemExit(f"{arc}: md5 {h.hexdigest()} is not OpenSLR's {md5}")
        unpacked = HERE / "data-source" / ("unpacked-" + arc.split(".")[0])
        if not unpacked.exists():
            tmp = unpacked.with_name(unpacked.name + ".part")
            with tarfile.open(dest) as t:
                t.extractall(tmp, filter="data")
            tmp.rename(unpacked)
        out[arc] = {"url": f"{OPENSLR}/{arc}", "md5": md5, "sha256": file_sha256(dest), "size": dest.stat().st_size,
                    "license": "CC BY 4.0 (LibriTTS-R, OpenSLR 141)"}
        print(f"{arc}: verified and unpacked", flush=True)
    (HERE / "data-source" / "DATA.json").write_text(json.dumps({"data": out}, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--verify", action="store_true", help="verify what is on disk, download nothing")
    a = ap.parse_args()
    if "HF_TOKEN" in os.environ:
        sys.exit("unset HF_TOKEN: the downloads are anonymous")
    if a.models or not a.data:
        fetch_models(a.verify)
    if a.data or not a.models:
        fetch_data(a.verify)
