"""
Transcribes the sampled LibriSpeech utterances with three Whisper model
sizes (tiny, base, small) on both test-clean and test-other, measuring
word error rate and wall clock time per utterance for each. Loads each
model once and reuses it across all utterances in that model's pass,
rather than reloading per file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jiwer
import whisper

from text_normalize import normalize


def load_model_with_retry(model_size: str, max_attempts: int = 5):
    # this network has been intermittently dropping mid download all session;
    # retry rather than losing an already completed model size's results
    for attempt in range(1, max_attempts + 1):
        try:
            return whisper.load_model(model_size)
        except Exception as e:
            if attempt == max_attempts:
                raise
            print(f"  load_model({model_size}) failed on attempt {attempt}: {e}, retrying in 5s")
            time.sleep(5)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_SIZES = ["tiny", "base", "small"]


def transcribe_split(model, utterances: list[dict], model_size: str, split: str) -> list[dict]:
    records = []
    for i, utt in enumerate(utterances):
        t0 = time.time()
        result = model.transcribe(utt["audio_path"], language="en", fp16=False)
        elapsed = time.time() - t0

        hypothesis = normalize(result["text"])
        reference = normalize(utt["ground_truth"])
        wer = jiwer.wer(reference, hypothesis) if reference else None

        records.append({
            "model_size": model_size,
            "split": split,
            "utt_id": utt["utt_id"],
            "ground_truth": utt["ground_truth"],
            "hypothesis_raw": result["text"],
            "wer": wer,
            "transcribe_seconds": elapsed,
        })
        if (i + 1) % 10 == 0:
            print(f"  {model_size}/{split}: {i + 1}/{len(utterances)} done")
    return records


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "sample_manifest.json") as f:
        manifest = json.load(f)

    partial_path = RESULTS_DIR / "raw_transcriptions.json"
    all_records = []
    done_model_sizes = set()
    if partial_path.exists():
        with open(partial_path) as f:
            all_records = json.load(f)
        done_model_sizes = {r["model_size"] for r in all_records}
        if done_model_sizes:
            print(f"resuming: already have results for {sorted(done_model_sizes)}")

    for model_size in MODEL_SIZES:
        if model_size in done_model_sizes:
            print(f"skipping {model_size}, already in {partial_path.name}")
            continue

        print(f"loading whisper model: {model_size}")
        t0 = time.time()
        model = load_model_with_retry(model_size)
        print(f"  loaded in {time.time() - t0:.1f}s")

        for split, utterances in manifest.items():
            print(f"transcribing {split} with {model_size} ({len(utterances)} utterances)")
            records = transcribe_split(model, utterances, model_size, split)
            all_records.extend(records)
            with open(partial_path, "w") as f:
                json.dump(all_records, f, indent=2)

        del model  # free memory before loading the next size

    with open(RESULTS_DIR / "raw_transcriptions.json", "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\n{len(all_records)} total transcriptions written to "
          f"{RESULTS_DIR / 'raw_transcriptions.json'}")


if __name__ == "__main__":
    main()
