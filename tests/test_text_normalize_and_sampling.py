import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_normalize import normalize


def test_normalize_lowercases():
    assert normalize("HELLO WORLD") == "hello world"


def test_normalize_strips_punctuation():
    assert normalize("Hello, world! How are you?") == "hello world how are you"


def test_normalize_keeps_apostrophes():
    assert normalize("don't can't won't") == "don't can't won't"


def test_normalize_collapses_whitespace():
    assert normalize("hello    world\n\nfoo") == "hello world foo"


def test_normalize_matches_librispeech_style_ground_truth():
    # LibriSpeech ground truth is upper case, no punctuation, single spaces
    ground_truth = "HE HOPED THERE WOULD BE STEW FOR DINNER"
    whisper_output = "He hoped there would be stew for dinner."
    assert normalize(ground_truth) == normalize(whisper_output)


def test_collect_utterances_reads_trans_file(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import importlib
    sample_data = importlib.import_module("sample_data")

    speaker_dir = tmp_path / "1234" / "5678"
    speaker_dir.mkdir(parents=True)
    trans_file = speaker_dir / "1234-5678.trans.txt"
    trans_file.write_text(
        "1234-5678-0000 HELLO THERE\n1234-5678-0001 GENERAL KENOBI\n"
    )
    (speaker_dir / "1234-5678-0000.flac").write_bytes(b"fake audio")
    (speaker_dir / "1234-5678-0001.flac").write_bytes(b"fake audio")

    utterances = sample_data.collect_utterances(tmp_path)
    assert len(utterances) == 2
    ids = {u["utt_id"] for u in utterances}
    assert ids == {"1234-5678-0000", "1234-5678-0001"}
    ground_truths = {u["ground_truth"] for u in utterances}
    assert ground_truths == {"HELLO THERE", "GENERAL KENOBI"}


def test_collect_utterances_skips_missing_audio(tmp_path):
    import importlib
    sample_data = importlib.import_module("sample_data")

    speaker_dir = tmp_path / "1" / "2"
    speaker_dir.mkdir(parents=True)
    trans_file = speaker_dir / "1-2.trans.txt"
    trans_file.write_text("1-2-0000 THIS FILE HAS NO AUDIO\n")
    # deliberately do not create the .flac file

    utterances = sample_data.collect_utterances(tmp_path)
    assert len(utterances) == 0
