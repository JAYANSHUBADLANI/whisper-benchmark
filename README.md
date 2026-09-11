# Does a bigger Whisper model just buy accuracy, or also robustness

[![tests](https://github.com/JAYANSHUBADLANI/whisper-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/JAYANSHUBADLANI/whisper-benchmark/actions/workflows/tests.yml)

A word error rate and speed benchmark of Whisper tiny, base, and small on
real recorded speech, LibriSpeech test-clean and test-other, entirely
local, no API keys, no cloud.

## Headline result

Bigger Whisper models are not just more accurate on easy audio, they are
specifically more robust to hard audio: the accuracy gap between clean and
noisy speech shrinks from 23.2 points at tiny down to 12.9 points at
small, even though both individual numbers also improve. A paired
bootstrap on the same 40 utterances both model sizes were run on (see
`src/significance_test.py`) puts a 95 percent confidence interval of
(0.018, 0.184) on that difference, excluding zero: this is not just a
point estimate that happens to move in the expected direction. Small costs
roughly 4 times tiny's inference time per utterance for that improvement.
Zero setup beyond what was already on this machine plus two small,
public, keyless downloads.

## Data

LibriSpeech test-clean and test-other, the standard ASR benchmark pair,
downloaded directly from openslr.org, no account or key needed. test-other
is specifically curated by the dataset's creators to be the harder,
noisier condition, which gives a real difficulty axis without having to
construct one. Sampled 40 utterances from each split with a fixed seed
(13) rather than transcribing the full 2620 and 2939 utterance sets, to
keep three model sizes' worth of CPU inference to a reasonable runtime.
See `src/sample_data.py`.

## Method

Whisper tiny, base, and small, each loaded once and run over both 40
utterance samples. Word error rate computed with `jiwer` after normalizing
both the ground truth and Whisper's output (lower case, punctuation
stripped, whitespace collapsed, apostrophes kept) so formatting
differences between LibriSpeech's upper case, unpunctuated transcripts and
Whisper's naturally cased, punctuated output do not inflate the error
rate. See `src/text_normalize.py` and `src/run_benchmark.py`.

## Results

| Model | test-clean WER | test-other WER | Gap | Mean seconds/utterance |
|---|---|---|---|---|
| tiny | 9.98% | 33.19% | 23.20 pts | 0.17-0.22s |
| base | 7.76% | 25.78% | 18.01 pts | 0.29s |
| small | 5.22% | 18.08% | 12.87 pts | 0.77-0.78s |

See `figures/wer_by_model_size.png` for the two split lines converging as
model size increases, and `figures/speed_vs_accuracy.png` for the
accuracy gained per second of extra inference time.

Word error rate correlates negatively with utterance duration in every
model and split, -0.17 to -0.30 (`results/wer_duration_correlation.csv`):
partly a real effect, partly a measurement artifact, since one wrong word
in a five word utterance is a 20 percent error rate while the same single
error in a forty word utterance is 2.5 percent, so short utterances
mechanically pull toward more extreme WER values. This means a single
short utterance's WER should not be read as precisely as a long one's.

## Statistical validity of the headline claim

A first attempt at a confidence interval treated tiny's and small's
results as independent samples and bootstrapped each model size's gap
separately: the two individual 95 percent intervals overlapped, and the
naive difference between them included zero, which would not count as a
significant result. That approach was wrong for this experiment's actual
design: tiny and small were both run on the identical 40 utterances per
split, not two separate samples, so a hard utterance being hard for both
models is a real correlation the naive approach discards as noise. The
correct paired bootstrap, resampling the same utterance indices for both
model sizes on every draw, gives the (0.018, 0.184) interval reported
above. Both attempts, and why the first one was the wrong tool rather
than just a more conservative one, are in `PROGRESS.md`.

## What the aggregate number hides

The ten worst individual transcriptions (`results/worst_10_transcriptions.csv`)
are almost entirely test-other with tiny or base, and several are not near
miss word substitutions, they are complete breakdowns onto unrelated
words: "KNOWEST THOU WHITHER HE WENT" became "No was that with a he
went." LibriSpeech is built from public domain audiobooks, and several of
the worst cases involve archaic or literary diction, thee, thou, whither,
spake, that a model would see far less of in typical training data. The
average WER treats this the same as a model that is uniformly a little
worse on hard audio; it is not the same failure mode, and a deployment
whose real audio contains unusual vocabulary would see this kind of
complete miss more often than the aggregate number suggests.

## Sharpest ways this could be wrong

40 utterances per split is enough for the paired bootstrap above to
exclude zero, but the interval, (0.018, 0.184), is still wide relative to
the 0.10 point estimate; a larger sample would narrow it, and the exact
percentage point values here should not be read as more precise than that
interval implies.

Whisper medium and large were not tested. Medium is roughly 1.5GB and
large over 2.9GB, meaningfully heavier to download and slower on CPU, for
a question the tiny to small range already answers directionally; whether
the gap keeps narrowing, flattens, or reverses above small was not
checked.

The archaic diction failure mode is correlated with test-other in this
sample, but whether unusual vocabulary or noisier recording conditions is
the actual driver was not separately tested, since LibriSpeech's split
design varies both at once rather than one at a time.

## Reproducing this

```
cd whisper-benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd data/raw
curl -o test-clean.tar.gz https://www.openslr.org/resources/12/test-clean.tar.gz
curl -o test-other.tar.gz https://www.openslr.org/resources/12/test-other.tar.gz
tar -xzf test-clean.tar.gz && tar -xzf test-other.tar.gz
cd ../../src
python3 sample_data.py
python3 run_benchmark.py
python3 error_analysis.py
python3 significance_test.py
python3 make_figures.py
cd ..
python3 -m pytest tests/ -v
```

`run_benchmark.py` saves its results incrementally per model size and
retries model loading on a network error, since this network dropped mid
download twice while this project was built; see `PROGRESS.md`. Needs
`ffmpeg` on PATH, already required for Whisper's own audio decoding.

## Test suite

15 of 15 tests pass: text normalization, the LibriSpeech transcript
parsing logic (with synthetic files, no real audio needed), integrity
checks on the saved benchmark results including that accuracy actually
improves with model size and that the clean versus other gap actually
shrinks, and that the paired bootstrap confidence interval on the headline
gap difference still excludes zero, so a future change that silently
broke any of these patterns would fail a test rather than only show up in
a chart nobody re-checked.
