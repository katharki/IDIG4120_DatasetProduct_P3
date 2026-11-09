# IDIG4120 P1-P3 synthesis

**Course:** IDIG4120 Digital Information Processing  
**Authors:** Katharina Kivle and Nikolai O. Stubø

## 1. Which dataset, and which question does your project answer?

We used *UrbanSound8K*, 8,732 sound excerpts (at most 4 s) of urban
sounds in WAV format. The question the project answers is *what
happens if you make a lossy FLAC?*.

The codec keeps the FLAC approach, predictive coding with Rice entropy
and mid-side stereo, and adds quantization of the prediction errors.
P1 audits what the data actually is, P2 builds the codec and measures
exactly that, and P3 packages the dataset with the codec as a product.

## 2. How did each part shape the next?

Each part left the next one with something it needed. P1's audit
showed the excerpts arrive in a gamut of WAV encodings: *16-bit,
24-bit, 8-bit unsigned, float, even ADPCM*. That was the report's
implementation concern, and it is why P2 normalizes everything to
int16 PCM before the codec sees it, with a loader that routes by
subtype instead of trusting one decoder path.

P2 taught us what pays and what does not: prediction with Rice coding
brings lossless audio down to about half the PCM size, but the
advanced predictors never earned their keep, so the simple fixed
predictor is still what P3 ships, with step 4 as the lossy comparison.
P3 imports the codec unchanged, and the datasheet's facts about the
files come from the P1 audit.

## 3. What is your strongest evidence, and the main limitation across the three parts?

The strongest evidence is **the lossy range**: it gets below FLAC's
size with the error bounded by design, at most half the quantization
step per sample. The full-release sweep, all 8732 excerpts, puts numbers
on each quality level. Step 2 compresses to 52.7% of PCM at a median SNR
of 73.0 dB. Step 4 is the first below FLAC, 46.4% of PCM against 47.4%,
at 67.6 dB. The remaining steps live in the results notebook's tables.
What the loss does to the sound is not in these numbers, so the results
notebook plays the same clip at several steps, and the note there records
where the difference becomes audible.

The main limitation is **the lossless side**: FLAC is simply better
there, 47.4% of PCM against our 57.0%, and the advanced machinery never
closed the gap. LPC loses to the simple fixed predictor on most of the
clips we tested, and LMS is a little better at many times the cost. What
we added over FLAC is the lossy range, not a better lossless codec.

## 4. Is the dataset fit for use: use, use with controls, investigate, or do not use? What would change it?

**Use with controls.** Though not having any problems in terms of
invalid, mislabeled, corrupted or otherwise deviant data, we found the
dataset was harder to work with than we initially expected due to the
gamut of different WAV encodings.

To simplify the task of making a codec we introduced the
normalization step to 16-bit PCM before the codec runs: converting
the full release cut its storage by 17.43%, from 6,760.26 to
5,581.96 MiB, with that conversion being one of the controls now,
next to digest verification before processing, no resampling, and
the predefined folds kept intact.

For the purposes of making an audio codec this data set has some
drawbacks; the short length of clips and a bit limited sampling in
terms of "types" of noise (the classes). For our purposes a dataset
committed to a single wav encoding and more variation in terms of
types and lengths of clips would be more fitting.

## 5. What is still unresolved?

There are some things we would have liked to implement to more
completely reach our goal of a *lossy* FLAC. These are mostly
concerning optimizations and features of FLAC that we did not end up
implementing. Among these the most noteworthy are our choice to
always apply mid-side pairing when encoding audio channels in pairs
and improve the more advanced predictors as they are currently not
providing much improvement.

For future work we would also like to add support for more
WAV-formats than only PCM16 to more accurately mirror FLAC.
