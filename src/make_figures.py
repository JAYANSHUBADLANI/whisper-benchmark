import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


def plot_wer_by_model_and_split():
    df = pd.read_csv(RESULTS_DIR / "summary_by_model_split.csv")
    order = ["tiny", "base", "small"]
    df["model_size"] = pd.Categorical(df["model_size"], categories=order, ordered=True)
    df = df.sort_values("model_size")

    fig, ax = plt.subplots(figsize=(7, 5))
    for split in df["split"].unique():
        sub = df[df["split"] == split]
        ax.plot(sub["model_size"], sub["mean_wer"], marker="o", label=split)
    ax.set_xlabel("Whisper model size")
    ax.set_ylabel("mean word error rate")
    ax.set_title("Word error rate by model size, test-clean vs test-other")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "wer_by_model_size.png", dpi=150)
    plt.close()


def plot_speed_vs_accuracy():
    df = pd.read_csv(RESULTS_DIR / "summary_by_model_split.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"test-clean": "o", "test-other": "s"}
    for split in df["split"].unique():
        sub = df[df["split"] == split]
        ax.scatter(sub["mean_transcribe_seconds"], sub["mean_wer"],
                   marker=markers.get(split, "o"), s=80, label=split)
        for _, row in sub.iterrows():
            ax.annotate(row["model_size"], (row["mean_transcribe_seconds"], row["mean_wer"]),
                       textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("mean transcribe time per utterance, seconds")
    ax.set_ylabel("mean word error rate")
    ax.set_title("Speed versus accuracy tradeoff")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "speed_vs_accuracy.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    plot_wer_by_model_and_split()
    plot_speed_vs_accuracy()
    print(f"figures written to {FIG_DIR}")
