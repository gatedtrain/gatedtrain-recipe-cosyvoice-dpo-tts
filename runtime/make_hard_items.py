"""Build a hard-text zero-shot TTS item set (deterministic): programmatic hard sentences plus long natural
LibriTTS-R sentences, spoken in the voices of a LibriTTS-R split.

Plain LibriTTS-R text leaves the base model at the WER floor (smoke: 12 of 14 items at 0), so the items stress what
TTS gets wrong: numbers, dates, times, money, units, initialisms, homographs, tongue twisters, rare names and long
multi-clause sentences. Two disjoint text sources keep training and evaluation apart: `--text-split train` and
`--text-split test` use different templates and word lists (no template and no lexicon entry is shared; asserted at
import), and the natural sentences come from different LibriTTS-R splits (train-clean-100 for training, test-clean or
dev-clean for evaluation; a train text split must take voices and natural sentences from train_clean_100, a test text
split must not). Voices: per speaker of `--voice-split`, the reference is the utterance with the most words among
those of 4-10 s (as make_items.py). Each item is one (speaker, text) pair; texts are assigned to speakers round-robin.
Natural sentences: the longest (most words, then id) utterances of `--natural-split` with 30 to
`--max-natural-words` words (default 70, which keeps the speech under Whisper's 30 s window; the pilot had no cap,
its longest was 71 words / 26 s), never an utterance that is also a voice prompt.

Presets (the builds of the DPO study; explicit flags override the counts):
  --preset train : --text-split train --voice-split train_clean_100 --natural-split train_clean_100
                   --n-hard 1600 --n-natural 400
  --preset test  : --text-split test --voice-split test_clean --natural-split test_clean --n-hard 360 --n-natural 90
  --preset dev   : train templates and lexicon (text split `train`, so no test template leaks into model selection),
                   voices and natural sentences from dev_clean, --n-hard 120 --n-natural 30. Its hard texts come from
                   their own RNG stream (`hard:dev:<seed>`) and every text of the train preset's hard set with the same
                   seed is rejected (the fixed-list categories, all used by train, are therefore absent). Items are
                   named `hard-dev:...`. Only for gen_tts.py + score_tts.py: the candidate sampler, reward scorer,
                   pair builder and trainer accept only text split `train` with voices from train_clean_100.
dev_clean is also the pilot/probe split (items-hard-pilot; sharing speakers with it is fine). The preset builds also
assert that the voice speakers are disjoint from every speaker of the other presets' voice splits (directory
listing). `--check-against DIR [DIR ...]` asserts that the new set shares no speaker, prompt, text, reference text
or natural source with each existing item set (build train, then test with `--check-against items-hard-train`, then
dev with `--check-against items-hard-train items-hard-test`).

Reference text: the item's `text` (numerals as written) is what the model reads; the scorer's Whisper English
normalizer and canonicalization (eval/wer_canon.py) bring transcript and reference to the same written form. An
abbreviated unit (`314 nmi`, `MPa`, `m3`, `kWh`, ...) is read as its full name, which no normalizer recovers, so a
unit item whose text contains an abbreviation also gets `ref_text`: the same sentence with that abbreviation written
as the spoken name (`314 nautical miles`); the scorers use `ref_text` when present.

usage: make_hard_items.py --preset test --seed 11 --out items-hard-test [--check-against items-hard-train]
       make_hard_items.py --preset dev --seed 11 --out items-hard-dev --check-against items-hard-train items-hard-test
       make_hard_items.py --text-split test --voice-split dev_clean --natural-split dev_clean --n-hard 54
                          --n-natural 10 --seed 7 --out items-hard-pilot [--limit-speakers N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
SRC = HERE / "data-source"

# Disjoint lexicons per text split (train / test). Every list is used by one split only.
LEX = {
    "train": {
        "names": ["Thaddeus Okonkwo", "Marisol Vandersteen", "Ignatius Przybylski", "Leocadia Ferreira-Nunes",
                  "Bartholomew Achterberg", "Siobhan Ni Dhomhnaill", "Evander Szczepanski", "Ottoline Merriweather"],
        "places": ["Ouagadougou", "Llanfairfechan", "Worcestershire", "Tegucigalpa", "Saskatoon", "Zaporizhzhia"],
        "units": [("kilometers", "km"), ("milligrams", "mg"), ("square feet", "sq ft"), ("kilowatt hours", "kWh")],
        "abbr": ["NASA", "UNESCO", "HTML", "GDP", "MRI", "DNA", "FAQ", "PDF"],
        "homographs": [("lead", "The plumber found lead in the old pipes, so he will lead the repair crew."),
                       ("wind", "We had to wind the clock before the wind knocked the shutters loose."),
                       ("tear", "A single tear fell when she saw the tear in her wedding dress."),
                       ("bass", "The fisherman caught a bass while humming the bass line of the song.")],
        "twisters": ["Six sleek swans swam swiftly southwards.", "Fresh fried fish, fish fresh fried, fried fish fresh.",
                     "Thin sticks, thick bricks, quick tricks.", "Red lorry, yellow lorry, red lorry, yellow lorry."],
        "clauses": ["although the committee had already adjourned", "because the forecast had changed twice",
                    "while the second shipment was still at customs", "unless the auditors objected in writing"],
    },
    "test": {
        "names": ["Wojciech Kowalczyk-Brennan", "Aoife Mac Giolla Phadraig", "Leopoldine Haverkamp",
                  "Florentyna Zbigniewska", "Cuthbert Oyelaran", "Ysolde Ravenscroft", "Dagobert Quintanilla",
                  "Euphemia Blankenship"],
        "places": ["Antananarivo", "Loughborough", "Ystradgynlais", "Chichicastenango", "Kirkcudbright", "Nuuk"],
        "units": [("nautical miles", "nmi"), ("micrograms", "mcg"), ("cubic meters", "m3"), ("megapascals", "MPa")],
        "abbr": ["FIFA", "NATO", "SQL", "OPEC", "CPR", "LED", "GPS", "USB"],
        "homographs": [("bow", "The violinist took a bow and then tied a bow around the gift."),
                       ("row", "They had a row about who would row the boat across the lake."),
                       ("dove", "The dove flew off as the swimmer dove into the cold water."),
                       ("close", "Please close the gate, since the bus stop is close to the road.")],
        "twisters": ["Peculiar pink pelicans plucked plump purple plums.", "Crisp crusts crackle and crunch.",
                     "Truly rural, truly rural, truly rural.", "Which wristwatches are Swiss wristwatches?"],
        "clauses": ["even though the hearing had been postponed", "since the bridge was closed for repairs",
                    "after the referee reviewed the footage", "whereas the earlier estimate was far lower"],
    },
}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
PRESETS = {
    "train": {"text_split": "train", "voice_split": "train_clean_100", "natural_split": "train_clean_100",
              "n_hard": 1600, "n_natural": 400},
    "test": {"text_split": "test", "voice_split": "test_clean", "natural_split": "test_clean",
             "n_hard": 360, "n_natural": 90},
    "dev": {"text_split": "train", "voice_split": "dev_clean", "natural_split": "dev_clean",
            "n_hard": 120, "n_natural": 30},
}


def _lexicon_entries(split: str) -> set[str]:
    L = LEX[split]
    out = set(L["names"]) | set(L["places"]) | set(L["abbr"]) | set(L["twisters"]) | set(L["clauses"])
    out |= {u for pair in L["units"] for u in pair} | {w for pair in L["homographs"] for w in pair}
    return out


assert not (_lexicon_entries("train") & _lexicon_entries("test")), "train and test lexicons overlap"


def ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th, 11th, 12th, 13th, 21st, 22nd, 23rd (the pilot's template wrote `2th`, `21th`)."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _unit(r: random.Random, units, fmt) -> tuple[str, str]:
    """A unit sentence and its reference: the abbreviation is written as its spoken name in the reference. Draw
    order (unit, first number, second number) is the pilot template's."""
    u = r.choice(units)
    a, b = r.randint(2, 900), r.randint(2, 900)
    return fmt(a, u[1], b, u[0]), fmt(a, u[0], b, u[0])


def templates(split: str):
    """Sentence makers of a text split; the two splits' templates differ in wording and structure. A maker returns
    the text, or (text, ref_text) when the spoken reference differs from the text."""
    L = LEX[split]
    if split == "train":
        return {
            "number": lambda r: f"The warehouse logged {r.randint(1000, 999999):,} parcels and {r.randint(2, 99)}.{r.randint(10, 99)} percent were late.",
            "date": lambda r: f"The contract was signed on {MONTHS[r.randrange(12)]} {r.randint(1, 28)}, {r.randint(1890, 2030)}, at {r.randint(1, 12)}:{r.randint(0, 59):02d} p.m.",
            "money": lambda r: f"Repairs cost ${r.randint(10, 9999):,}.{r.randint(0, 99):02d}, which is about {r.randint(2, 40)} percent over budget.",
            "unit": lambda r: _unit(r, L["units"], lambda a, ua, b, ub: f"The route covers {a} {ua}, roughly {b} {ub} per week."),
            "abbr": lambda r: f"The {r.choice(L['abbr'])} report and the {r.choice(L['abbr'])} summary were filed in {r.randint(1990, 2030)}.",
            "homograph": lambda r: r.choice(L["homographs"])[1],
            "twister": lambda r: r.choice(L["twisters"]),
            "name": lambda r: f"{r.choice(L['names'])} flew from {r.choice(L['places'])} to {r.choice(L['places'])} on flight {r.randint(100, 9999)}.",
            "long": lambda r: f"The board approved the budget, {r.choice(L['clauses'])}, and {r.choice(L['names'])} promised a revised plan by {MONTHS[r.randrange(12)]} {r.randint(1, 28)}, {r.choice(L['clauses'])}.",
        }
    return {
        "number": lambda r: f"Exactly {r.randint(1000, 999999):,} visitors came, and attendance rose by {r.randint(2, 99)}.{r.randint(10, 99)} percent.",
        "date": lambda r: f"Doors opened at {r.randint(1, 12)}:{r.randint(0, 59):02d} a.m. on the {ordinal(r.randint(1, 28))} of {MONTHS[r.randrange(12)]}, {r.randint(1890, 2030)}.",
        "money": lambda r: f"She paid ${r.randint(10, 9999):,}.{r.randint(0, 99):02d} upfront and ${r.randint(1, 999)} each month after that.",
        "unit": lambda r: _unit(r, L["units"], lambda a, ua, b, ub: f"Engineers measured {a} {ua} on the first test and {b} {ub} on the second."),
        "abbr": lambda r: f"According to {r.choice(L['abbr'])}, the {r.choice(L['abbr'])} upgrade ships before {r.randint(1990, 2030)}.",
        "homograph": lambda r: r.choice(L["homographs"])[1],
        "twister": lambda r: r.choice(L["twisters"]),
        "name": lambda r: f"Yesterday {r.choice(L['names'])} drove from {r.choice(L['places'])} toward {r.choice(L['places'])} with {r.randint(2, 60)} passengers.",
        "long": lambda r: f"The museum reopened the east wing, {r.choice(L['clauses'])}, while {r.choice(L['names'])} catalogued {r.randint(100, 9999):,} artifacts, {r.choice(L['clauses'])}.",
    }


def split_root(src: Path, split: str) -> Path:
    return src / f"unpacked-{split}" / "LibriTTS_R" / split.replace("_", "-")


def split_speakers(src: Path, split: str) -> set[str]:
    """Speaker ids of a LibriTTS-R split (directory names; no audio is read)."""
    return {p.name for p in split_root(src, split).iterdir() if p.is_dir() and p.name.isdigit()}


def speakers(split: str, src: Path = SRC) -> dict[str, list[dict]]:
    root = split_root(src, split)
    out: dict[str, list[dict]] = {}
    for wav in sorted(root.glob("*/*/*.wav")):
        txt = wav.with_name(wav.name[:-4] + ".normalized.txt")
        if txt.exists():
            text = txt.read_text(encoding="utf-8").strip()
            info = sf.info(str(wav))
            out.setdefault(wav.parts[-3], []).append({"id": wav.stem, "wav": str(wav.relative_to(src)), "text": text,
                                                      "words": len(text.split()),
                                                      "seconds": info.frames / info.samplerate})
    return out


def hard_texts(text_split: str, n_hard: int, seed: int, stream: str | None = None,
               exclude: set[str] = frozenset()) -> list[dict]:
    """`stream` names the RNG stream (default: the text split); texts in `exclude` are never produced."""
    rng = random.Random(f"hard:{stream or text_split}:{seed}")
    makers = templates(text_split)
    # Round-robin over categories; a category whose maker yields no new text in MAX_TRIES draws (the fixed-list
    # categories: homograph, twister) is exhausted and the remaining items go to the others.
    MAX_TRIES = 1000
    texts: list[dict] = []
    seen: set[str] = set(exclude)
    live = sorted(makers)
    turn = 0
    while len(texts) < n_hard:
        if not live:
            raise SystemExit(f"only {len(texts)} distinct hard texts exist; asked for {n_hard}")
        cat = live[turn % len(live)]
        for _ in range(MAX_TRIES):
            made = makers[cat](rng)
            t, ref = made if isinstance(made, tuple) else (made, made)
            if t not in seen:
                seen.add(t)
                texts.append({"category": cat, "text": t, **({"ref_text": ref} if ref != t else {})})
                turn += 1
                break
        else:
            live.remove(cat)
    return texts


def voice_refs(voices: dict[str, list[dict]], limit_speakers: int | None) -> list[tuple[str, dict]]:
    refs = []
    for spk in sorted(voices, key=int)[:limit_speakers]:
        cands = [u for u in voices[spk] if 4.0 <= u["seconds"] <= 10.0]
        if cands:
            refs.append((spk, max(cands, key=lambda u: (u["words"], u["id"]))))
    return refs


def natural_texts(natural: dict[str, list[dict]], n: int, max_words: int, exclude_ids: set[str]) -> list[dict]:
    pool = sorted((u for us in natural.values() for u in us
                   if 30 <= u["words"] <= max_words and u["id"] not in exclude_ids),
                  key=lambda u: (-u["words"], u["id"]))
    texts, seen = [], set()
    for u in pool:                                   # a sentence read twice in the corpus is one text
        if len(texts) == n:
            break
        if u["text"] not in seen:
            seen.add(u["text"])
            texts.append({"category": "natural_long", "text": u["text"], "source_id": u["id"]})
    if len(texts) < n:
        raise SystemExit(f"only {len(texts)} natural sentences of 30-{max_words} words; asked for {n}")
    return texts


def check_split_rules(text_split: str, voice_split: str, natural_split: str, preset: str | None = None) -> None:
    if preset == "dev":
        if (text_split, voice_split, natural_split) != ("train", "dev_clean", "dev_clean"):
            raise SystemExit("the dev set takes train templates and dev_clean voices and natural sentences")
        return
    if text_split == "train" and (voice_split, natural_split) != ("train_clean_100", "train_clean_100"):
        raise SystemExit("a train text split takes voices and natural sentences from train_clean_100")
    if text_split == "test" and "train_clean_100" in (voice_split, natural_split):
        raise SystemExit("a test text split never takes voices or natural sentences from train_clean_100")


def check_disjoint(rows_a: list[dict], rows_b: list[dict]) -> None:
    """Two item sets share no speaker, no prompt utterance and no text (text or ref_text, either way round)."""
    def texts(rows):
        return {r["text"] for r in rows} | {r["ref_text"] for r in rows if r.get("ref_text")}
    for what, fa, fb in (("speaker", {r["speaker"] for r in rows_a}, {r["speaker"] for r in rows_b}),
                         ("prompt", {r["prompt_id"] for r in rows_a}, {r["prompt_id"] for r in rows_b}),
                         ("text", texts(rows_a), texts(rows_b)),
                         ("source", {r["source_id"] for r in rows_a if r.get("source_id")},
                          {r["source_id"] for r in rows_b if r.get("source_id")})):
        shared = fa & fb
        if shared:
            raise SystemExit(f"item sets share {len(shared)} {what}(s), e.g. {sorted(shared)[:3]}")


def build_rows(text_split: str, voice_split: str, natural_split: str, n_hard: int, n_natural: int, seed: int,
               src: Path = SRC, max_natural_words: int = 70, limit_speakers: int | None = None,
               preset: str | None = None) -> tuple[list, int]:
    check_split_rules(text_split, voice_split, natural_split, preset)
    voices = speakers(voice_split, src)
    refs = voice_refs(voices, limit_speakers)
    if not refs:
        raise SystemExit("no speaker has a 4-10 s reference utterance")
    label = "dev" if preset == "dev" else text_split
    if preset == "dev":
        train_texts = hard_texts("train", PRESETS["train"]["n_hard"], seed)
        texts = hard_texts(text_split, n_hard, seed, stream="dev",
                           exclude={t[f] for t in train_texts for f in ("text", "ref_text") if f in t})
    else:
        texts = hard_texts(text_split, n_hard, seed)
    if n_natural:
        natural = voices if natural_split == voice_split else speakers(natural_split, src)
        texts += natural_texts(natural, n_natural, max_natural_words, {ref["id"] for _, ref in refs})
    rows = []
    for i, t in enumerate(texts):
        spk, ref = refs[i % len(refs)]
        key = hashlib.sha256(f"{label}:{t['text']}".encode()).hexdigest()[:12]
        row = {"item": f"hard-{label}:{t['category']}:{key}", "category": t["category"], "speaker": spk,
               "text": t["text"], "prompt_text": ref["text"], "prompt_wav": ref["wav"], "prompt_id": ref["id"],
               "source_id": t.get("source_id")}
        if t.get("ref_text"):
            row["ref_text"] = t["ref_text"]
        rows.append(row)
    assert len({r["item"] for r in rows}) == len(rows), "duplicate item ids"
    assert len({r["text"] for r in rows}) == len(rows), "duplicate texts"
    return rows, len(refs)


def sha256_file(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--text-split", choices=["train", "test"])
    ap.add_argument("--voice-split", choices=["train_clean_100", "dev_clean", "test_clean"])
    ap.add_argument("--natural-split", choices=["train_clean_100", "dev_clean", "test_clean"])
    ap.add_argument("--n-hard", type=int)
    ap.add_argument("--n-natural", type=int)
    ap.add_argument("--max-natural-words", type=int, default=70)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-speakers", type=int, default=None)
    ap.add_argument("--check-against", nargs="+", default=[])
    ap.add_argument("--data-source", default=str(SRC))
    a = ap.parse_args()
    cfg = dict(PRESETS[a.preset]) if a.preset else {}
    for k in ("text_split", "voice_split", "natural_split", "n_hard", "n_natural"):
        v = getattr(a, k)
        if a.preset and k.endswith("split") and v is not None and v != cfg[k]:
            raise SystemExit(f"--preset {a.preset} fixes --{k.replace('_', '-')} = {cfg[k]}")
        if v is not None:
            cfg[k] = v
        if k not in cfg:
            raise SystemExit(f"--{k.replace('_', '-')} is required without --preset")
    src = Path(a.data_source)
    out = HERE / a.out
    if out.exists():
        raise SystemExit(f"{out} exists: an item set is never overwritten")
    if a.preset:
        for other in sorted({p["voice_split"] for n, p in PRESETS.items() if n != a.preset} - {cfg["voice_split"]}):
            if split_root(src, other).exists():
                shared = split_speakers(src, cfg["voice_split"]) & split_speakers(src, other)
                if shared:
                    raise SystemExit(f"{cfg['voice_split']} and {other} share speakers {sorted(shared)[:5]}")
    rows, n_speakers = build_rows(cfg["text_split"], cfg["voice_split"], cfg["natural_split"], cfg["n_hard"],
                                  cfg["n_natural"], a.seed, src, a.max_natural_words, a.limit_speakers, a.preset)
    check_meta = []
    for other in a.check_against:
        other_dir = HERE / other
        other_rows = [json.loads(l) for l in (other_dir / "items.jsonl").read_text(encoding="utf-8").splitlines() if l]
        check_disjoint(rows, other_rows)
        check_meta.append({"dir": str(other_dir.resolve()), "items_sha256": sha256_file(other_dir / "items.jsonl")})
    out.mkdir(parents=True)
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (out / "items.jsonl").write_text(body, encoding="utf-8")
    meta = {**cfg, "preset": a.preset, "set": "dev" if a.preset == "dev" else cfg["text_split"], "seed": a.seed, "n_items": len(rows), "n_speakers": n_speakers,
            "max_natural_words": a.max_natural_words, "limit_speakers": a.limit_speakers,
            "n_ref_text": sum("ref_text" in r for r in rows), "checked_disjoint_from": check_meta or None,
            "categories": {c: sum(r["category"] == c for r in rows) for c in sorted({r["category"] for r in rows})},
            "items_sha256": "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "script_sha256": sha256_file(Path(__file__)),
            "sources": "programmatic templates (this script); LibriTTS-R (OpenSLR 141, CC BY 4.0)"}
    (out / "item_set_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
