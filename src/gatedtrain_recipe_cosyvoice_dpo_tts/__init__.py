"""gatedtrain recipe: Fun-CosyVoice3-0.5B DPO on the LLM speech tokens with a Whisper WER reward. `pins` holds the pinned
runtime scripts, item sets and weights; `rows` registers the adapter `cosyvoice_dpo` (`adapter.CosyVoiceDpoAdapter`);
`arms` turns score_tts.py output into gatedtrain arms; `smoke` writes the smoke record."""
