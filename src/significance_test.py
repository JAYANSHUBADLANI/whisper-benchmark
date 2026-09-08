"""
Bootstraps a confidence interval on the clean versus other WER gap for
each model size, and on the difference between tiny's gap and small's
gap directly, rather than reporting the point estimates in README.md and
PROGRESS.md as if 40 utterances per split settled the question. Added
after those files had already stated the gap shrinking as a clean
finding: point estimates alone overstated how settled that was at this
sample size, see PROGRESS.md for the full account.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
N_BOOT = 5000
SEED = 42


def bootstrap_gap(df: pd.DataFrame, model_size: str, rng: np.random.Generator) -> np.ndarray:
    clean = df[(df.model_size == model_size) & (df.split == "test-clean")]["wer"].dropna().values
    other = df[(df.model_size == model_size) & (df.split == "test-other")]["wer"].dropna().values
    gaps = np.empty(N_BOOT)
    for i in range(N_BOOT):
        c = rng.choice(clean, size=len(clean), replace=True).mean()
        o = rng.choice(other, size=len(other), replace=True).mean()
        gaps[i] = o - c
    return gaps


def main() -> None:
    with open(RESULTS_DIR / "raw_transcriptions.json") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    rng = np.random.default_rng(SEED)

    per_size_gaps = {}
    rows = []
    for size in ["tiny", "base", "small"]:
        gaps = bootstrap_gap(df, size, rng)
        per_size_gaps[size] = gaps
        rows.append({
            "model_size": size,
            "gap_mean": float(gaps.mean()),
            "gap_ci_low": float(np.percentile(gaps, 2.5)),
            "gap_ci_high": float(np.percentile(gaps, 97.5)),
        })
        print(f"{size}: gap={gaps.mean():.4f}  95% CI=({np.percentile(gaps, 2.5):.4f}, "
              f"{np.percentile(gaps, 97.5):.4f})")

    # paired bootstrap on tiny_gap - small_gap directly, resampling each
    # group's utterances independently per draw rather than differencing
    # the two already-bootstrapped arrays, since the two model sizes were
    # run on the same underlying 40 utterances per split, not independent
    # samples
    tiny_clean = df[(df.model_size == "tiny") & (df.split == "test-clean")]["wer"].dropna().values
    tiny_other = df[(df.model_size == "tiny") & (df.split == "test-other")]["wer"].dropna().values
    small_clean = df[(df.model_size == "small") & (df.split == "test-clean")]["wer"].dropna().values
    small_other = df[(df.model_size == "small") & (df.split == "test-other")]["wer"].dropna().values
    n_clean, n_other = len(tiny_clean), len(tiny_other)

    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        clean_idx = rng.integers(0, n_clean, n_clean)
        other_idx = rng.integers(0, n_other, n_other)
        tiny_gap = tiny_other[other_idx].mean() - tiny_clean[clean_idx].mean()
        small_gap = small_other[other_idx].mean() - small_clean[clean_idx].mean()
        diffs[i] = tiny_gap - small_gap

    p_tiny_gap_larger = float((diffs > 0).mean())
    ci_low, ci_high = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    print(f"\nP(tiny's clean-vs-other gap > small's) = {p_tiny_gap_larger:.4f}")
    print(f"difference in gaps (tiny - small): mean={diffs.mean():.4f}, "
          f"95% CI=({ci_low:.4f}, {ci_high:.4f})")
    significant_at_95 = not (ci_low <= 0 <= ci_high)
    print(f"95% CI excludes zero: {significant_at_95}")

    output = {
        "per_model_gap_ci": rows,
        "tiny_minus_small_gap_diff_mean": float(diffs.mean()),
        "tiny_minus_small_gap_diff_ci_95": [ci_low, ci_high],
        "p_tiny_gap_larger_than_small": p_tiny_gap_larger,
        "significant_at_95pct_two_sided": significant_at_95,
        "n_bootstrap": N_BOOT,
        "seed": SEED,
    }
    with open(RESULTS_DIR / "significance_test.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nwritten to {RESULTS_DIR / 'significance_test.json'}")


if __name__ == "__main__":
    main()
