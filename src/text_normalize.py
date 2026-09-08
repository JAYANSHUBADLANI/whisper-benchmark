"""
Normalizes text before WER computation: lowercase, strip punctuation,
collapse whitespace. LibriSpeech ground truth is upper case with no
punctuation; Whisper's raw output has natural casing and punctuation.
Comparing them unnormalized would inflate WER with formatting differences
that have nothing to do with transcription accuracy.
"""

import re


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
