# Progress log

The question: does Whisper model size only buy accuracy, or does it also
buy robustness to harder audio conditions, and what does that cost in
speed, on real recorded speech with real ground truth rather than one
file transcribed and called done.

Picked LibriSpeech test-clean and test-other, the standard ASR benchmark
pair, over recording my own audio: real speakers, real ground truth
transcripts, and test-other is specifically curated to be the harder,
noisier condition, so clean versus other gives a genuine difficulty axis
for free instead of having to construct one. Downloaded both full test
sets, 2620 and 2939 utterances, then sampled 40 from each with a fixed
seed rather than transcribing all of them, since running three model
sizes over roughly 5500 utterances on a laptop CPU would take far longer
than the accuracy question needs.

Normalized text before computing WER: lower case, punctuation stripped,
whitespace collapsed. LibriSpeech ground truth is upper case with no
punctuation and Whisper's output has natural casing and periods, and
comparing them unnormalized would inflate WER with formatting differences
that have nothing to do with transcription accuracy. Kept apostrophes in
normalization since "don't" and "dont" are a real transcription
difference, not formatting noise.

Network dropped mid download twice during this project, once during the
LibriSpeech download itself and once while Whisper was downloading the
small model checkpoint, both times as a connection reset partway through
a file. This wifi has been flaky lately, not specific to this project. Added a retry wrapper around model loading and made
run_benchmark.py save its transcription results incrementally per model
size rather than only at the very end, so a network drop after the tiny
and base sizes finished would not throw away work already done. The
second run of run_benchmark.py hit a connection reset loading the small
model on the first attempt and succeeded on retry without losing the tiny
or base results that had already run.

Results, 40 utterances per split, mean word error rate:

tiny:  test-clean 9.98%, test-other 33.19%, gap 23.20 points
base:  test-clean 7.76%, test-other 25.78%, gap 18.01 points
small: test-clean 5.22%, test-other 18.08%, gap 12.87 points

Accuracy improving with model size is the expected, unsurprising part.
The gap between test-clean and test-other shrinking as model size goes up
is the finding worth stating plainly: going from tiny to small does not
just move both numbers down by roughly the same amount, it specifically
narrows the accuracy penalty for harder audio, meaning the larger model is
not just more accurate on easy audio, it degrades less on hard audio. A
recommendation to use a bigger model "for accuracy" undersells what is
actually happening if the real use case involves variable audio quality
rather than clean, close mic conditions.

Speed: tiny about 0.17 to 0.22 seconds per utterance, base about 0.29,
small about 0.77 to 0.78, so small is roughly 2.7 times slower than base
and 4 times slower than tiny for this sample. Whether that tradeoff is
worth it depends entirely on whether the deployment's real audio looks
more like test-clean or test-other, which is exactly the kind of thing a
single aggregate WER number from one condition would hide.

Checked whether WER correlates with utterance duration: negative
correlation in every group, -0.17 to -0.30, meaning longer utterances tend
to have a lower word error rate. This is at least partly a measurement
artifact rather than a pure statement about audio difficulty: WER for a
short utterance is noisy, one wrong word out of five words is a 20 percent
error, one wrong word out of forty words is 2.5 percent, so short
utterances mechanically pull the distribution toward more extreme WER
values in both directions. Recorded as a real caveat on how much to trust
any single utterance's WER rather than something the sample size here can
resolve.

Looked at the ten worst individual transcriptions rather than only the
aggregate numbers. Nearly all of them are test-other with tiny or base,
and several are not near miss substitution errors, they are full breakdown
onto unrelated words: "KNOWEST THOU WHITHER HE WENT" became "No was that
with a he went," "THOU AMIABLE ONE" became "Now, stand up one." LibriSpeech
audio comes from public domain audiobooks, and several of the worst cases
involve archaic or literary diction, thee, thou, whither, spake, that a
transcription model trained mostly on contemporary speech would see far
less of. This is a real, specific failure mode worth naming rather than
folding into a single WER average: the model is not uniformly a little
worse on hard audio, it can fail completely on a specific kind of input
the average obscures.

Went back and actually computed a confidence interval on the gap rather
than leaving that as a stated limitation, since the gap shrinking is the
headline claim and a claim that central to the project deserves more than
a point estimate. First attempt at this, in significance_test.py, treated
tiny's and small's results as independent samples and bootstrapped each
model size's gap separately: their individual 95 percent intervals
overlapped substantially, and a naive difference between two
independently resampled gap arrays gave a 95 percent interval of
(-0.033, 0.241), which includes zero, meaning that version of the check
would not call the gap difference significant.

That version was wrong, not just conservative: tiny and small were run on
the identical 40 utterances per split, not two separate samples, so their
per utterance errors are correlated, a hard utterance tends to be hard for
both models. Treating them as independent throws away that correlation
and adds noise the actual experiment does not have. Redid it as a paired
bootstrap, resampling the utterance indices once per draw and applying the
same resampled indices to both model sizes' arrays before computing each
one's gap, which is what the experiment's actual design supports. That
version gives a 95 percent confidence interval on the tiny minus small gap
difference of (0.018, 0.184), which excludes zero, and P(tiny's gap is
larger) of 0.992 rather than the naive check's 0.930. The headline finding
holds up under the statistically appropriate test; it did not hold up
clearly under the first, wrong test, and I am recording both attempts
rather than only the one that agreed with what I already believed, since
finding the same answer either way would not have told me the first check
was flawed.

Left undone: Did not test Whisper's medium or large models, since medium
alone is roughly 1.5GB and large over 2.9GB, meaningfully heavier
downloads and slower CPU inference for what the tiny to small range
already answers about the direction of the tradeoff. Did not separately
test whether the archaic diction failure mode is really about vocabulary
rarity specifically versus just being correlated with the noisier
recording conditions test-other selects for; distinguishing those two
explanations would need a dataset built to vary one without the other,
which LibriSpeech was not built for.
