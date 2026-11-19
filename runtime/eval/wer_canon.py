"""WER with Whisper's English text normalizer plus a canonicalization of known normalizer artifacts (canon-1).

Shared by score_tts.py (deciding Qwen3-ASR WER, reported Whisper WER) and score_candidates.py (Whisper reward WER), so
the reward and the deciding metric compare text the same way. Both sides of every comparison (reference and ASR
hypothesis) go through the same `canonical()`; nothing here looks at which side a string is.

canonical(text) = post(normalizer(pre(text))):
- pre, on the raw string (before the normalizer):
  - `H:00` -> `H` ("9:00 a.m." is read "nine a m", which the normalizer writes `9 a m`);
  - `H:MM` -> `HMM` ("6:10" and "10:05"; the normalizer would write `6 10` and `10 5`, dropping the minutes'
    leading zero, and `1:11` as `one 11`, while the spoken "six ten" / "ten oh five" / "one eleven" normalize to
    `610` / `1005` / `111`);
  - `H.MM` directly before a.m./p.m./am/pm -> `HMM` (Whisper writes times as `6.10 a.m.`); a dotted number
    anywhere else stays a decimal, so `54.74` vs `54,274` and `$5.94` vs `$594` still differ.
- post, on the normalized string:
  - `H MM` directly before `a m`/`p m`/`am`/`pm` -> `HMM` (an ASR that writes "6 10 a.m.");
  - a run of two or more single-letter tokens is joined (`l e d` -> `led`, `a m` -> `am`, `c p r` -> `cpr`): a
    dotted or spelled acronym and its plain spelling become one token.

Known limits (both sides alike, so they never favour one side): the normalizer writes a lone letter `o` as `0`
(`U N E S C O` -> `u n e s c 0`), which breaks the letter run; two genuinely separate one-letter words in a row
(rare: "i a") would also be joined.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CANON_VERSION = "canon-1"

_TIME_00 = re.compile(r"\b(\d{1,2}):00\b")
_TIME_COLON = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_TIME_DOT = re.compile(r"\b(\d{1,2})\.(\d{2})(?=\s*[ap]\.?\s?m\b)", re.IGNORECASE)
_TIME_SPACED = re.compile(r"\b(\d{1,2}) (\d{2})(?= (?:a m|p m|am|pm)\b)")
_LETTER = re.compile(r"[a-z]")


def pre(text: str) -> str:
    """Raw-text step (before the normalizer): times to one HMM number."""
    text = _TIME_00.sub(r"\1", text)
    text = _TIME_COLON.sub(r"\1\2", text)
    return _TIME_DOT.sub(r"\1\2", text)


def post(normalized: str) -> str:
    """Normalized-text step: spaced times before am/pm, then single-letter runs joined."""
    s = _TIME_SPACED.sub(r"\1\2", normalized)
    out: list[str] = []
    run: list[str] = []
    for tok in s.split():
        if len(tok) == 1 and _LETTER.fullmatch(tok):
            run.append(tok)
            continue
        out.extend(["".join(run)] if len(run) >= 2 else run)
        run = []
        out.append(tok)
    out.extend(["".join(run)] if len(run) >= 2 else run)
    return " ".join(out)


def load_normalizer(models_dir: Path):
    """Whisper's EnglishTextNormalizer with the pinned whisper-large-v3-turbo spelling map (as score_tts.py)."""
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    return EnglishTextNormalizer(json.loads((Path(models_dir) / "whisper-large-v3-turbo" / "normalizer.json").read_text()))


def canonical(text: str, norm) -> str:
    return post(norm(pre(text)))


def wer_plain(ref: str, hyp: str, norm) -> float:
    """The pre-canon metric (normalizer only); NaN when the normalized reference is empty."""
    import jiwer
    r, h = norm(ref), norm(hyp)
    return float(jiwer.wer(r, h)) if r.strip() else float("nan")


def wer_canon(ref: str, hyp: str, norm) -> float:
    """canon-1 WER; NaN when the canonical reference is empty."""
    import jiwer
    r, h = canonical(ref, norm), canonical(hyp, norm)
    return float(jiwer.wer(r, h)) if r.strip() else float("nan")


def reference(item: dict) -> str:
    """The text an item is scored against: `ref_text` (spoken form of abbreviated units) when present, else `text`."""
    return item.get("ref_text") or item["text"]
