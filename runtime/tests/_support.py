"""Shared test helpers: paths, and (main venv only) a tiny randomly initialized CosyVoice3LM.

Tests never load the real 0.5B weights. Run per venv from anywhere:
  A/.venv/bin/python -m unittest discover -s A/tests -t A/tests -p 'test_main_*.py'
  A/eval/.venv/bin/python -m unittest discover -s A/tests -t A/tests -p 'test_eval_*.py'
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # tests never touch a GPU (imported before torch by every test)
from pathlib import Path

AREA = Path(__file__).resolve().parent.parent
EVAL = AREA / "eval"
for p in (str(AREA), str(EVAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

ENDOFPROMPT = 151646      # CosyVoice3LM.inference asserts this token id is in the text (llm.py:479-481)


def cosyvoice_path() -> None:
    for p in (str(AREA / "src" / "CosyVoice" / "third_party" / "Matcha-TTS"), str(AREA / "src" / "CosyVoice")):
        if p not in sys.path:
            sys.path.insert(0, p)


def tiny_llm(speech_token_size: int = 120, hidden: int = 32, layers: int = 2, seed: int = 0):
    """A CosyVoice3LM with a tiny random Qwen2 (real vocabulary size, so <|endofprompt|> exists), the real
    ras_sampling settings of cosyvoice3.yaml, and stop-token logits pushed down so sampled sequences are long."""
    cosyvoice_path()
    from functools import partial

    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from cosyvoice.llm.llm import CosyVoice3LM, Qwen2Encoder
    from cosyvoice.utils.common import ras_sampling

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=151936, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=True, attn_implementation="sdpa")
    enc = Qwen2Encoder.__new__(Qwen2Encoder)
    torch.nn.Module.__init__(enc)
    enc.model = Qwen2ForCausalLM(cfg)
    llm = CosyVoice3LM(llm_input_size=hidden, llm_output_size=hidden, speech_token_size=speech_token_size, llm=enc,
                       sampling=partial(ras_sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1))
    with torch.no_grad():
        llm.llm_decoder.weight.mul_(20.0)                          # spread the speech-token logits
        llm.llm_decoder.weight[speech_token_size:].mul_(0.02)      # stop tokens rarely in the top-k
    return llm.eval()


def toy_tokens(text: str, salt: str = "") -> list[int]:
    """Deterministic fake tokenizer: one id per word, in [1000, 1999]."""
    import hashlib
    return [1000 + int(hashlib.sha256(f"{salt}{w}".encode()).hexdigest()[:6], 16) % 1000 for w in text.split()]


class FakeCV:
    """Stands in for AutoModel on CPU: the frontend is a toy tokenizer, the LLM is the real (tiny) CosyVoice3LM
    called the way CosyVoice3Model.llm_job calls it (model.py:110-119, kwargs), the 'audio' is 960 samples per
    token. Records every call for assertions."""

    sample_rate = 24000

    def __init__(self, llm, n_prompt_speech: int = 7):
        from types import SimpleNamespace
        self.model = SimpleNamespace(llm=llm)
        self.n_prompt_speech = n_prompt_speech
        self.calls = []

    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, zero_shot_spk_id="", stream=False, speed=1.0,
                            text_frontend=True):
        import torch
        self.calls.append({"text": tts_text, "prompt_text": prompt_text, "prompt_wav": prompt_wav, "stream": stream,
                           "text_frontend": text_frontend})
        head, _, tail = prompt_text.partition("<|endofprompt|>")
        pt = toy_tokens(head) + [ENDOFPROMPT] + toy_tokens(tail)
        size = self.model.llm.speech_token_size
        ps = [t % size for t in toy_tokens(prompt_wav + " x" * self.n_prompt_speech, "ps")][: self.n_prompt_speech]
        text = torch.tensor([toy_tokens(tts_text)], dtype=torch.int32)
        ptt = torch.tensor([pt], dtype=torch.int32)
        pst = torch.tensor([ps], dtype=torch.int32)
        gen = self.model.llm.inference(text=text, text_len=torch.tensor([text.shape[1]], dtype=torch.int32),
                                       prompt_text=ptt, prompt_text_len=torch.tensor([ptt.shape[1]], dtype=torch.int32),
                                       prompt_speech_token=pst,
                                       prompt_speech_token_len=torch.tensor([pst.shape[1]], dtype=torch.int32),
                                       embedding=torch.ones(1, 192), uuid="u")
        toks = list(gen)
        yield {"tts_speech": torch.zeros(1, 960 * max(1, len(toks)))}
