"""Paths of the checkout the recipe tests read (no model, audio or GPU needed)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNTIME = REPO / "runtime"
GATE = REPO / "results" / "2026-11-17" / "gate"
EVIDENCE = REPO / "results" / "2026-11-17" / "evidence"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "area11"


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
