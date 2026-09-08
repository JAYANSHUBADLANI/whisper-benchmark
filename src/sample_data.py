"""
Samples a fixed number of real utterances from LibriSpeech test-clean and
test-other, with ground truth transcripts, and saves a manifest the
benchmark script reads from. A fixed seed keeps the sample the same across
reruns.

Full test-clean is 2620 utterances and test-other 2939, both real audio.
Transcribing all of them with three model sizes on a laptop CPU would take
hours; sampling a fixed subset keeps runtime reasonable while still being
real recorded speech with real ground truth, not synthetic audio.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRISPEECH_DIR = ROOT / "data" / "raw" / "LibriSpeech"
DATA_DIR = ROOT / "data"
N_SAMPLES_PER_SPLIT = 40
SEED = 13


def collect_utterances(split_dir: Path) -> list[dict]:
    utterances = []
    for trans_file in split_dir.rglob("*.trans.txt"):
        chapter_dir = trans_file.parent
        for line in trans_file.read_text().splitlines():
            utt_id, transcript = line.split(" ", 1)
            audio_path = chapter_dir / f"{utt_id}.flac"
            if audio_path.exists():
                utterances.append({
                    "utt_id": utt_id,
                    "audio_path": str(audio_path),
                    "ground_truth": transcript.strip(),
                })
    return utterances


def main() -> None:
    rng = random.Random(SEED)
    manifest = {}
    for split in ["test-clean", "test-other"]:
        split_dir = LIBRISPEECH_DIR / split
        all_utts = collect_utterances(split_dir)
        print(f"{split}: {len(all_utts)} total utterances found")
        sample = rng.sample(all_utts, min(N_SAMPLES_PER_SPLIT, len(all_utts)))
        manifest[split] = sample
        print(f"{split}: sampled {len(sample)} utterances (seed {SEED})")

    with open(DATA_DIR / "sample_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest written to {DATA_DIR / 'sample_manifest.json'}")


if __name__ == "__main__":
    main()
