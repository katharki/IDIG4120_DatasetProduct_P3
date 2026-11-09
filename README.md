# IDIG4120 P1-P3

The course project for IDIG4120 Digital Information Processing:
UrbanSound8K, from dataset audit to packaged dataset product. Three parts
build on each other: the audit fixes the baseline representation, the codec
compresses it, the product packages the dataset with a reproducible
pipeline.

## Layout

p1/ holds the dataset audit report. p2/ holds the APAC audio codec, its
source, tests and documentation. p3/ holds the dataset product built on the
codec; p3/README.md covers the setup, the entry point, the tests and the
notebook.

## Requirements

Python 3.10 or newer. pip install -r p3/requirements.txt (numpy, soundfile,
pandas, pytest) covers p2 and p3. The p1 notebook also uses scipy and
matplotlib.

## The codec (p2)

APAC (Another Predictive Audio Codec) is a lossless and lossy audio codec
for 16-bit PCM: predictive error encoding with Rice, stereo pairs go
mid-side first. quant_step 1 is lossless and bit exact, larger steps are
lossy with bounded error (max step/2 LSB). From p2:

```text
pip install -e .                                     # installs the apac command
apac --compress -i in.wav -o out.apac --quality 100  # lossless
apac --decompress -i out.apac -o back.wav
python evaluation.py in.wav out.apac                 # ratio, bitrate, SNR
```

python -m audio_codec.main works without the install. p2/README.md covers
the settings, the wire format and the full evaluation.

## The dataset product (p3)

The product packages the dataset with a SHA-256 manifest and one entry
point. The codec is APAC from the p2 folder, imported as a dependency. From
p3:

```text
python run.py
python run.py --verify-only --dataset-dir PATH/to/full/archive
```

The default run verifies the configured clip against the manifest, encodes
with APAC, decodes, asserts a bit-exact round trip and writes
compressed.apac, decoded.wav and results.json into output/. The manifest
covers the full release, UrbanSound8K.csv plus 8,732 excerpts. The
repository carries 253 of the 873 fold1 excerpts in p2/data/raw. Missing
files are not errors, so a subset works.

## Evaluation

Over the full release, all 10 folds and 8732 excerpts, lossless APAC is
57.0% of 16-bit PCM against FLAC's 47.4%, about 1.2x, with 8732/8732 bit
exact round trips. Step 4 is the first lossy step under FLAC: 46.4% of
PCM at a median SNR of 67.6 dB. The full tables live in p2/README.md.

## Tests

```text
python -m pytest test      # from p2: 38 format and codec tests
python -m pytest tests     # from p3: 9 integrity, pipeline and failure tests
```

## Licence

The dataset licence (CC BY-NC 4.0, see the datasheet in the p3 notebook)
restricts commercial use. Cite the dataset paper when using the data.
