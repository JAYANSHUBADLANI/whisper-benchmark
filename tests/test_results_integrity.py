import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def test_raw_transcriptions_has_all_combinations():
    with open(RESULTS_DIR / "raw_transcriptions.json") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    combos = df.groupby(["model_size", "split"]).size()
    assert len(combos) == 6  # 3 model sizes x 2 splits
    assert (combos == 40).all()


def test_all_wer_values_are_non_negative():
    with open(RESULTS_DIR / "raw_transcriptions.json") as f:
        records = json.load(f)
    for r in records:
        assert r["wer"] is None or r["wer"] >= 0


def test_summary_shows_accuracy_improves_with_model_size():
    summary = pd.read_csv(RESULTS_DIR / "summary_by_model_split.csv")
    for split in ["test-clean", "test-other"]:
        sub = summary[summary["split"] == split].set_index("model_size")
        assert sub.loc["small", "mean_wer"] < sub.loc["tiny", "mean_wer"]


def test_test_other_gap_shrinks_with_model_size():
    gap = pd.read_csv(RESULTS_DIR / "clean_vs_other_gap.csv", index_col=0)
    assert gap.loc["small", "gap"] < gap.loc["tiny", "gap"]


def test_worst_transcriptions_file_has_high_wer_entries():
    worst = pd.read_csv(RESULTS_DIR / "worst_10_transcriptions.csv")
    assert len(worst) == 10
    assert worst["wer"].min() > 0.5
