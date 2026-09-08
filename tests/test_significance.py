import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def test_significance_result_excludes_zero():
    with open(RESULTS_DIR / "significance_test.json") as f:
        result = json.load(f)
    ci_low, ci_high = result["tiny_minus_small_gap_diff_ci_95"]
    assert ci_low > 0, "the gap-shrinking claim should hold at the 95% level given the current data"
    assert result["significant_at_95pct_two_sided"] is True


def test_significance_result_has_all_three_model_sizes():
    with open(RESULTS_DIR / "significance_test.json") as f:
        result = json.load(f)
    sizes = {row["model_size"] for row in result["per_model_gap_ci"]}
    assert sizes == {"tiny", "base", "small"}


def test_gap_confidence_intervals_are_all_positive():
    # every model size's clean-vs-other gap CI should not include zero,
    # since test-other being harder than test-clean is not in question,
    # only whether the size of that gap differs meaningfully by model size
    with open(RESULTS_DIR / "significance_test.json") as f:
        result = json.load(f)
    for row in result["per_model_gap_ci"]:
        assert row["gap_ci_low"] > 0, f"{row['model_size']}'s gap CI should be entirely positive"
