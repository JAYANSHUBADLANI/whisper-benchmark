"""
Goes past a single headline WER number per model size: looks at whether
errors correlate with audio duration or reference length, whether the
tiny to small gap is bigger on test-other than test-clean (the gap a
"just use the biggest model" recommendation would need to survive), and
surfaces the worst individual transcriptions so the failure mode is
visible, not just its rate.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def get_duration_seconds(audio_path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return float("nan")


def main() -> None:
    with open(RESULTS_DIR / "raw_transcriptions.json") as f:
        records = json.load(f)
    with open(ROOT / "data" / "sample_manifest.json") as f:
        manifest = json.load(f)

    path_by_utt = {
        u["utt_id"]: u["audio_path"] for split in manifest.values() for u in split
    }

    df = pd.DataFrame(records)
    df["duration_seconds"] = df["utt_id"].map(
        lambda u: get_duration_seconds(path_by_utt[u])
    )
    df["reference_word_count"] = df["ground_truth"].str.split().str.len()

    summary = (
        df.groupby(["model_size", "split"])
        .agg(mean_wer=("wer", "mean"), median_wer=("wer", "median"),
             mean_transcribe_seconds=("transcribe_seconds", "mean"),
             n=("wer", "count"))
        .reset_index()
    )
    print(summary.to_string(index=False))

    corr_by_group = (
        df.groupby(["model_size", "split"])
        .apply(lambda g: g["wer"].corr(g["duration_seconds"]), include_groups=False)
        .reset_index(name="wer_duration_correlation")
    )
    print("\ncorrelation between WER and utterance duration, per group:")
    print(corr_by_group.to_string(index=False))

    clean_vs_other_gap = (
        summary.pivot(index="model_size", columns="split", values="mean_wer")
        .assign(gap=lambda d: d["test-other"] - d["test-clean"])
    )
    print("\ntest-other minus test-clean mean WER gap, per model size:")
    print(clean_vs_other_gap.to_string())

    worst = df.nlargest(10, "wer")[
        ["model_size", "split", "utt_id", "ground_truth", "hypothesis_raw", "wer"]
    ]

    df.to_csv(RESULTS_DIR / "transcriptions_with_duration.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary_by_model_split.csv", index=False)
    clean_vs_other_gap.to_csv(RESULTS_DIR / "clean_vs_other_gap.csv")
    worst.to_csv(RESULTS_DIR / "worst_10_transcriptions.csv", index=False)
    corr_by_group.to_csv(RESULTS_DIR / "wer_duration_correlation.csv", index=False)

    print(f"\nresults written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
