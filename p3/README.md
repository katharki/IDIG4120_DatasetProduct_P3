# UrbanSound8K dataset product (P3)

UrbanSound8K packaged with a reproducible compression pipeline: a datasheet, a quality assurance report, a SHA-256 manifest, one entry point, failure handling tests, and recorded results. The codec is APAC from the p2 folder of this repository, imported as a dependency.

## Layout

```text
run.py                     the entry point
config.json                every knob: dataset dir, input clip, quantization step, output dir
requirements.txt           numpy, soundfile, pandas, pytest
manifest/                  urbansound8k_manifest.json: relative path, byte size, SHA-256 per file
src/p3/
    config.py              product root discovery, config loading, derived paths
    integrity.py          manifest build, per-file and per-dataset digest verification
    pipeline.py           the pipeline steps, shape contract, all-at-once writes, benchmark
IDIG4120_P3_Kivle_Stubø.ipynb   the single notebook: datasheet, schema, QA report, pipeline and results
tests/                     integrity checks and the four required failure cases
output/                    run outputs and log, not tracked
```

## Requirements

Python 3.10 or newer, then:

```text
pip install -r requirements.txt
```

The p3 package imports the codec from p2/src; the p2 folder sits next to p3 in this repository.

## Run the pipeline

```text
python run.py
```

Verifies the configured clip against the manifest, loads it, encodes with APAC (quant_step 1 = lossless), decodes, asserts a bit-exact round trip of the loaded samples, and writes compressed.apac, decoded.wav and results.json into output/, each file written all at once or not at all. Every step appends one line to output/run.log.

Overrides: `--input PATH` (relative to the dataset dir), `--quant-step N`, `--dataset-dir PATH`.

Exit codes: 0 ok, 2 missing input, 3 digest mismatch, 4 invalid shape, 5 failed round trip, 130 interrupted.

## Verify a dataset copy

```text
python run.py --verify-only
python run.py --verify-only --dataset-dir PATH/to/full/archive
```

The manifest covers the full release: UrbanSound8K.csv plus 8,732 excerpts across fold1 to fold10. This repository carries part of fold1 inside p2/data/raw: 253 of the 873 excerpts, including the configured clip. Absent files are reported without being treated as errors, so a subset works; a digest mismatch or an unexpected file fails with exit code 3.

## Tests

```text
python -m pytest tests
```

Covers the integrity layer and the four required failure cases: missing input, digest mismatch, invalid shape, interrupted run.

## Notebook

IDIG4120_P3_Kivle_Stubø.ipynb holds the datasheet, the schema, the QA report and the pipeline results in one document. It runs from anywhere inside p3 and prints every number from the live run.

## Configuration

| Key | Meaning |
|---|---|
| dataset_dir | dataset root, relative to the product root |
| input_clip | the clip the pipeline runs, relative to dataset_dir, as listed in the manifest |
| quant_step | APAC quantization step, 1 = lossless |
| output_dir | where outputs and run.log go, relative to the product root |

## Determinism

The pipeline has no random steps. The same input, config and codec version produce the same output bytes, so no seed is needed. Wall-clock times in results.json vary between runs and machines.

## Licence

The dataset licence (CC BY-NC 4.0, see the datasheet, area 4) restricts commercial use. Cite the dataset paper when using the data.
