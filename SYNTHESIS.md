# IDIG4120 P1-P3 synthesis

**Course:** IDIG4120 Digital Information Processing  
**Authors:** Katharina Kivle and Nikolai O. Stubø

## 1. Which dataset, and which question does your project answer?

We used *UrbanSound8K*, 8,732 sound excerpts (at most 4 s) of urban
sounds in WAV format. The question the project answers is **what
happens if you make a lossy FLAC**.

The codec keeps the FLAC approach, predictive coding with Rice entropy
and mid-side stereo, and adds quantization of the prediction errors.
P1 audits what the data actually is, P2 builds the codec and measures
exactly that, and P3 packages the dataset with the codec as a product.

## 2. How did each part shape the next?

Each part left the next one with something it needed. P1's audit
showed the excerpts arrive in a gamut of WAV encodings: 16-bit,
24-bit, 8-bit unsigned, float, even ADPCM. That was the report's
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
step per sample. Step 4 already lands around FLAC's size (42.5% of
PCM against 42.1%) and the deeper steps go clearly under it, down to
24.4% at step 32. What the loss does to the sound is not in these
numbers, so the results notebook plays the same clip at several
steps, and the note there records where the difference becomes
audible.

The main limitation is **the lossless side**: FLAC is simply better
there, 42.1% of PCM against our 51.6%, and the advanced machinery
never closed the gap; LPC loses to the simple fixed predictor on most
of the clips we tested, and LMS is a little better at many times the
cost. What we added over FLAC is the lossy range, not a better
lossless codec.

## 4. Is the dataset fit for use: use, use with controls, investigate, or do not use? What would change it?

**Use with controls.** The dataset was harder to work with than it
looks, because the excerpts arrive in a gamut of WAV encodings:
16-bit, 24-bit, 8-bit unsigned, float, even ADPCM. The answer was the
normalization step to 16-bit PCM before the codec runs: converting
fold1's 873 files cut their storage by a fifth, 769.76 to 604.31 MiB,
and that conversion is one of the controls now, next to digest
verification before processing, no resampling, and the predefined
folds kept intact.

What keeps the verdict from plain use is that the class labels have
not been verified against the audio, and the licence (CC BY-NC 4.0)
restricts commercial use. Verified labels would change it; the licence
is the dataset authors' choice and not ours to change.

## 5. What is still unresolved?

The shortcuts we know about are the stereo and predictor choices. We
apply mid-side to every stereo pair without adjusting anything, and
the results notebook measured what that costs on this material,
roughly 3 to 5% against coding the channels separately; choosing
mid-side or separate per file, the way FLAC does, is the first thing
we would change.

The predictors are the same story: LPC almost never earns the
coefficient bytes it puts on the wire, and LMS would have to get much
faster before its small win is worth it.

The codec also stops at int16, so the precision in the dataset's wider
and float files is spent at load; keeping it would mean a wider
working format. These are the things we would do differently given
more time.