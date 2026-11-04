#Codec-layer tests: entropy, prediction, planes, block modes, end-to-end.

from pathlib import Path

import pytest
import numpy as np
import soundfile as sf

from audio_codec.codec import entropy
from audio_codec.codec.codec import (compress, decode_blocks, decode_pipeline, decompress, dct_quant_steps,
                                     encode_blocks, encode_pipeline, from_planes, quantize_residual,
                                     reconstruct_sample, to_planes)
from audio_codec.codec.format import BlockError, read_apac
from audio_codec.config import DEFAULT_SETTINGS, DCTMode, MODE_CONSTANT, Predictor
from audio_codec.main import compress_file
from audio_codec.shared.load_file import load_apac, load_wav, save_apac, save_wav


def test_zigzag_roundtrip():
    '''zigzag maps into the non-negative domain injectively and unzigzag inverts it.'''
    xs = np.concatenate([
        np.arange(-50_000, 50_001, dtype=np.int64),
        np.array([2**31 - 1, -(2**31 - 1), -2**31], dtype=np.int64),   #int32 extremes, outside the range above
    ])
    zs = entropy.zigzag_np(xs)
    assert (zs >= 0).all()
    assert len(np.unique(zs)) == len(zs)                     #injective: distinct inputs stay distinct
    back = np.array([entropy.unzigzag(int(z)) for z in zs])
    assert np.array_equal(back, xs)
    assert entropy.zigzag(0) == 0 and entropy.zigzag(1) == 2 and entropy.zigzag(-1) == 1


def test_rice_roundtrip():
    '''Rice encode -> decode returns the original residuals for representative k values; int32 extremes at high k.'''
    extremes = np.array([-(2**31), 2**31 - 1, -1, 0, 1], dtype=np.int64)
    for k in (0, 1, 2, 7, 15, 30):
        rng = np.random.default_rng(1000 + k)
        residuals = rng.integers(-2048, 2049, 120)
        enc = entropy.rice_encode(residuals, k)
        assert np.array_equal(entropy.rice_decode(enc, len(residuals), k), residuals)
    for k in (16, 30):            #extremes only at high k: the unary quotient zz >> k stays small there
        enc = entropy.rice_encode(extremes, k)
        assert np.array_equal(entropy.rice_decode(enc, len(extremes), k), extremes)


def test_rice_decode_rejects_truncated_stream():
    '''Every truncation of a valid Rice stream either decodes all residuals exactly or raises BlockError.'''
    rng = np.random.default_rng(1234)
    residuals = rng.integers(-64, 65, 24)
    for k in (0, 1, 5, 13, 30):
        enc = entropy.rice_encode(residuals, k)
        for cut in range(len(enc) + 1):
            try:
                decoded = entropy.rice_decode(enc[:cut], len(residuals), k)
                assert np.array_equal(decoded, residuals)    #only padding lost: still exact
            except BlockError as error:
                assert str(error).startswith("missing data: expected more residuals")


def test_best_k_picks_exact_argmin():
    '''best_k is the exact argmin of the bit cost, ties break to the lowest k.'''
    rng = np.random.default_rng(7)
    residuals = rng.integers(-1000, 1001, 500)
    costs = np.array([entropy.rice_size(residuals, k) for k in range(31)])
    assert entropy.best_k(residuals) == int(np.argmin(costs))
    assert entropy.best_k(np.zeros(10, dtype=np.int64)) == 0     #all-zero residuals: k = 0 is optimal


def test_quantize_reconstruct_bounds():
    '''Quantization/reconstruction round a residual to the nearest multiple of the step; step 1 is the identity.'''
    for res in (-100, -9, -4, 0, 4, 9, 100):
        assert abs(res - reconstruct_sample(0, quantize_residual(res, 8), 8)) <= 4
    assert quantize_residual(-7, 1) == -7 and reconstruct_sample(0, -7, 1) == -7


def test_predictors_lossless_at_step_one():
    '''Every predictor/order variant reconstructs a clipped random walk exactly at quant_step = 1.'''
    for predictor, order in [
        (Predictor.NONE, 0),
        (Predictor.FIXED, 1), (Predictor.FIXED, 2), (Predictor.FIXED, 4),
        (Predictor.LPC, 1), (Predictor.LPC, 4),
        (Predictor.LMS, 1), (Predictor.LMS, 4),
    ]:
        rng = np.random.default_rng(2000 + int(predictor) * 10 + order)
        x = np.clip(np.cumsum(rng.integers(-40, 41, 300)), -32768, 32767).astype(np.int16)
        payload = encode_blocks([x.astype(np.int32)], [x], predictor, order, 1, 64)
        decoded, _ = decode_blocks(payload, 1, 300, predictor, order, 1, 64)
        assert np.array_equal(decoded[0], x.astype(np.int32))


def test_mid_side_roundtrip_exact():
    '''Mid/side planes merge back into the original channels exactly for mono, pairs and an odd leftover.'''
    for channels in (1, 2, 3, 6):
        rng = np.random.default_rng(3000 + channels)
        x = rng.integers(-32768, 32768, (257, channels), dtype=np.int16)
        planes, sources = to_planes(x)
        assert len(planes) == channels                          #every channel yields exactly one plane
        payload = encode_blocks(planes, sources, Predictor.FIXED, 2, 1, 64)
        decoded, masks = decode_blocks(payload, channels, 257, Predictor.FIXED, 2, 1, 64)
        assert np.array_equal(from_planes(decoded, masks, channels, 64), x)


def test_raw_fallback_on_noise():
    '''White noise costs more in Rice than raw int16, so every block falls back.'''
    rng = np.random.default_rng(66)
    x = rng.integers(-32768, 32768, 256, dtype=np.int16)
    payload = encode_blocks([x.astype(np.int32)], [x], Predictor.NONE, 0, 1, 64)
    assert len(payload) == 4 * (1 + 2 * 64)                      #control + 2 bytes/sample, every block raw
    decoded, masks = decode_blocks(payload, 1, 256, Predictor.NONE, 0, 1, 64)
    assert masks == [[True] * 4]
    assert np.array_equal(decoded[0], x.astype(np.int32))


def test_from_planes_raw_matrix():
    '''All four raw/Rice combinations of a plane pair reconstruct L and R exactly.'''
    rng = np.random.default_rng(42)
    bs = 64
    frames = 4 * bs
    left = rng.integers(-30000, 30001, frames, dtype=np.int16)
    right = rng.integers(-30000, 30001, frames, dtype=np.int16)
    mid = left.astype(np.int32) + right
    side = left.astype(np.int32) - right
    raw_l = left.astype(np.int32)                           #a raw plane segment carries its source channel
    raw_r = right.astype(np.int32)
    planes = [
        np.concatenate([mid[:bs], raw_l[bs:2 * bs], mid[2 * bs:3 * bs], raw_l[3 * bs:]]),
        np.concatenate([side[:bs], side[bs:2 * bs], raw_r[2 * bs:3 * bs], raw_r[3 * bs:]]),
    ]
    masks = [[False, True, False, True], [False, False, True, True]]
    assert np.array_equal(from_planes(planes, masks, 2, bs), np.stack([left, right], axis=1))


def test_raw_rice_combinations_end_to_end():
    '''Forced raw/Rice plane mixes appear on the wire (masks) and reconstruct exactly.'''
    rng = np.random.default_rng(43)
    bs = 64
    n = 2 * bs
    noise = rng.integers(-29000, 29001, n, dtype=np.int16)
    ramp = (np.arange(n) * 2000 // n).astype(np.int16)
    #pair A: M = 2*noise (raw fallback), S = 2*ramp (Rice-friendly); pair B is the mirror
    x = np.stack([noise + ramp, noise - ramp, ramp + noise, ramp - noise], axis=1).astype(np.int16)
    planes, sources = to_planes(x)
    payload = encode_blocks(planes, sources, Predictor.FIXED, 1, 1, bs)
    decoded, masks = decode_blocks(payload, 4, n, Predictor.FIXED, 1, 1, bs)
    assert masks == [[True, True], [False, False], [False, False], [True, True]]
    assert np.array_equal(from_planes(decoded, masks, 4, bs), x)


def test_constant_block_wire_and_tails():
    '''Constant blocks store one big-endian int16 value; all-equal partial tails stay constant, a changing tail takes Rice.'''
    zeros = encode_blocks([np.zeros(150, dtype=np.int32)], [np.zeros(150, dtype=np.int16)], Predictor.NONE, 0, 1, 64)
    sevens = encode_blocks([np.full(100, 7, dtype=np.int32)], [np.full(100, 7, dtype=np.int16)], Predictor.NONE, 0, 1, 64)
    assert zeros == bytes([MODE_CONSTANT, 0, 0]) * 3         #zero-padded partial tail: still all-constant
    assert sevens == bytes([MODE_CONSTANT, 0, 7]) * 2        #the constant check sees block[:n]: real samples all 7
    tail = np.concatenate([np.full(99, 7), [5]]).astype(np.int16)
    changed = encode_blocks([tail.astype(np.int32)], [tail], Predictor.NONE, 0, 1, 64)
    assert changed[:3] == bytes([MODE_CONSTANT, 0, 7])       #the full first block stays constant
    assert changed[3] != MODE_CONSTANT                      #the tail changes value: takes the Rice path
    for payload, want in ((zeros, np.zeros(150, dtype=np.int32)),
                           (sevens, np.full(100, 7, dtype=np.int32)),
                           (changed, tail.astype(np.int32))):
        decoded, _ = decode_blocks(payload, 1, len(want), Predictor.NONE, 0, 1, 64)
        assert np.array_equal(decoded[0], want)


def test_mid_side_constant_split():
    '''A constant stereo signal: M = 40000 exceeds int16 (falls back raw), S = 0 stays constant.'''
    x = np.full((128, 2), 20000, dtype=np.int16)
    planes, sources = to_planes(x)
    payload = encode_blocks(planes, sources, Predictor.NONE, 0, 1, 64)
    decoded, masks = decode_blocks(payload, 2, 128, Predictor.NONE, 0, 1, 64)
    assert masks == [[True, True], [False, False]]         #mid raw, side constant
    assert np.array_equal(from_planes(decoded, masks, 2, 64), x)


def test_raw_constant_raw_mix():
    '''raw -> constant -> raw periods keep the masks right and roundtrip exactly.'''
    rng = np.random.default_rng(77)
    bs = 64
    sig = np.concatenate([
        rng.integers(-32768, 32768, bs, dtype=np.int16),
        np.full(bs, 7, dtype=np.int16),
        rng.integers(-32768, 32768, bs, dtype=np.int16),
    ])
    payload = encode_blocks([sig.astype(np.int32)], [sig], Predictor.NONE, 0, 1, bs)
    decoded, masks = decode_blocks(payload, 1, len(sig), Predictor.NONE, 0, 1, bs)
    assert masks == [[True, False, True]]
    assert np.array_equal(decoded[0], sig.astype(np.int32))


def test_partial_last_block_trims_to_frames():
    '''A partial last block is zero-padded for coding and trimmed back to frames on decode.'''
    rng = np.random.default_rng(88)
    x = rng.integers(-32768, 32768, 150, dtype=np.int16)
    payload = encode_blocks([x.astype(np.int32)], [x], Predictor.NONE, 0, 1, 64)
    decoded, _ = decode_blocks(payload, 1, 150, Predictor.NONE, 0, 1, 64)
    assert len(decoded[0]) == 150
    assert np.array_equal(decoded[0], x.astype(np.int32))


def test_closed_loop_error_stays_bounded():
    '''At quant_step > 1 the error stays within step/2 over a long stream: no drift.'''
    rng = np.random.default_rng(99)
    n, step = 5000, 8
    x = np.clip(np.cumsum(rng.integers(-3, 4, n)), -32768, 32767).astype(np.int16)
    payload = encode_blocks([x.astype(np.int32)], [x], Predictor.FIXED, 2, step, 256)
    decoded, _ = decode_blocks(payload, 1, n, Predictor.FIXED, 2, step, 256)
    assert np.abs(decoded[0] - x.astype(np.int32)).max() <= step // 2


def test_encode_pipeline_defaults_lossless():
    '''Missing knobs fall back to DEFAULT_SETTINGS and the defaults roundtrip losslessly.'''
    rng = np.random.default_rng(6000)
    x = rng.integers(-32768, 32768, (300, 2), dtype=np.int16)
    header, payload = encode_pipeline(x, {"sample_rate": 44100})
    for knob, value in DEFAULT_SETTINGS.items():
        assert getattr(header, knob) == value
    assert (header.frames, header.channels, header.sample_rate) == (300, 2, 44100)
    assert np.array_equal(decode_pipeline(header, payload), x)


def test_encode_pipeline_dct_quality_bounds_rejected():
    '''QUANTIZE quality outside [1-100] never encodes: dct_quant_steps rejects it before any wire bytes exist.'''
    x = np.zeros((64, 2), dtype=np.int16)
    for bad_quality in (0, 101):
        with pytest.raises(ValueError, match="quality"):
            encode_pipeline(x, {"sample_rate": 44100, "dct_mode": DCTMode.QUANTIZE,
                                "dct_param": bad_quality, "dct_window_size": 64})


def test_dct_quantize_steps_ladder():
    '''QUANTIZE steps are a JPEG-style quality ladder: all 1 at 100, ramping per bin, coarser as quality drops.'''
    assert np.all(dct_quant_steps(100, 1024) == 1)
    steps50 = dct_quant_steps(50, 1024)
    assert steps50[0] == 1 and steps50[-1] == 1 + 1024 * 1023 // 32   #base ramps 1 (DC) to a window-scaled top
    assert np.all(np.diff(steps50) >= 0)
    assert np.all(dct_quant_steps(5, 1024) >= steps50)
    assert dct_quant_steps(50, 1) == np.array([1])      #single-bin window: the base table stays at 1
    for quality in (0, 101):
        with pytest.raises(ValueError, match="quality"):
            dct_quant_steps(quality, 1024)


def test_encode_pipeline_dct_quantize_is_gradually_lossy():
    '''QUANTIZE degrades gracefully: quality 100 stays near-transparent, lower quality trades error for size.'''
    rng = np.random.default_rng(6003)
    x = rng.integers(-30000, 30001, (512, 2), dtype=np.int16)
    sizes, errs = {}, {}
    for quality in (100, 50, 1):
        header, payload = encode_pipeline(x, {"sample_rate": 44100, "dct_mode": DCTMode.QUANTIZE,
                                             "dct_param": quality, "dct_window_size": 256})
        assert (header.dct_mode, header.dct_param) == (DCTMode.QUANTIZE, quality)
        y = decode_pipeline(header, payload)
        assert y.shape == x.shape and y.dtype == np.int16
        sizes[quality] = len(payload)
        errs[quality] = np.abs(y.astype(np.int64) - x.astype(np.int64)).max()
    assert errs[100] <= 8                                #steps all 1: only transform rounding remains
    assert errs[1] >= errs[50] >= errs[100]
    assert sizes[1] <= sizes[50] <= sizes[100]


def test_end_to_end_lossless(tmp_path):
    '''WAV -> APAC -> WAV through the loader (all file I/O) + pure codec is bit-exact for mono/stereo/3ch.'''
    for channels in (1, 2, 3):
        rng = np.random.default_rng(4000 + channels)
        n, sr = 500, 44100
        x = rng.integers(-32768, 32768, (n, channels), dtype=np.int16)
        wav, apac, back = (tmp_path / f"in{channels}.wav", tmp_path / f"out{channels}.apac",
                           tmp_path / f"back{channels}.wav")
        save_wav(wav, x, sr)                                     #loader owns all file I/O
        metadata, original = load_wav(wav)
        blob = compress((original, metadata["samplerate"]), {})   #codec is a pure bytes transform
        save_apac(apac, blob)
        header, _ = read_apac(load_apac(apac))
        assert (header.frames, header.channels, header.sample_rate, header.quant_step) == (n, channels, sr, 1)
        samples, sample_rate = decompress(load_apac(apac))
        save_wav(back, samples, sample_rate)
        _, decoded = load_wav(back)
        assert np.array_equal(decoded, original)
        assert np.array_equal(decoded, x)


def test_end_to_end_lossy():
    '''quant_step > 1 decodes cleanly with the per-sample error bounded by step/2.'''
    rng = np.random.default_rng(5000)
    n, sr, step = 2000, 44100, 16
    x = np.clip(np.cumsum(rng.integers(-100, 101, (n, 2)), axis=0), -32768, 32767).astype(np.int16)
    decoded, _ = decompress(compress((x, sr), {"quant_step": step}))
    err = np.abs(decoded.astype(np.int64) - x.astype(np.int64))
    assert decoded.shape == x.shape
    assert err.max() <= step // 2


def test_compress_requires_data_tuple():
    '''compress() takes data=(samples, sample_rate) as one tuple; anything else is rejected up front.'''
    x = np.zeros((4, 1), dtype=np.int16)
    for bad in (x, (x,), (x, 44100, "extra"), "not a tuple"):
        with pytest.raises(ValueError, match="samples, sample_rate"):
            compress(bad, {})
    with pytest.raises(ValueError, match="sample_rate"):
        compress((x, 0), {})
    with pytest.raises(ValueError, match="sample_rate"):
        compress((x, 44100.5), {})


def test_decompress_cross_checks_supplied_dict():
    '''decompress() stays header-driven: a supplied settings/metadata dict is only cross-checked against the header.'''
    x = np.zeros((16, 2), dtype=np.int16)
    blob = compress((x, 44100), {"quant_step": 4})
    assert np.array_equal(decompress(blob)[0], x)          #None skips the check entirely
    samples, rate = decompress(blob, {"quant_step": 4, "predictor": Predictor.FIXED})
    assert samples.shape == x.shape and rate == 44100
    samples, rate = decompress(blob, {"samplerate": 44100, "channels": 2, "frames": 16})
    assert samples.shape == x.shape and rate == 44100
    with pytest.raises(ValueError, match="quant_step"):
        decompress(blob, {"quant_step": 8})
    with pytest.raises(ValueError, match="samplerate"):
        decompress(blob, {"samplerate": 48000})


def test_compress_rejects_empty_wav(tmp_path):
    '''A 0-frame wav never reaches the codec: compress_file rejects it with a clean ValueError.'''
    wav = tmp_path / "empty.wav"
    save_wav(wav, np.zeros((0, 1), dtype=np.int16), 44100)
    with pytest.raises(ValueError, match="frames"):
        compress_file(wav, tmp_path / "out.apac", {})


def test_tiny_inputs_roundtrip():
    '''Inputs shorter than a block or the predictor order still roundtrip: exact at step 1, bounded at step 8.'''
    for n, ch in ((1, 1), (2, 1), (3, 1), (2, 2), (3, 2)):
        x = (((np.arange(n * ch) * 37) % 251 - 125).astype(np.int16)).reshape(n, ch)
        back, rate = decompress(compress((x, 44100), {}))
        assert np.array_equal(back, x) and rate == 44100
        back8, _ = decompress(compress((x, 44100), {"quant_step": 8}))
        assert back8.shape == x.shape
        assert np.abs(back8.astype(np.int64) - x.astype(np.int64)).max() <= 4


def test_load_wav_casts_float_wav(tmp_path):
    '''A FLOAT wav takes the float64 path and comes back as the exact round/clip int16 cast (int16 read is zeros there, README NOTES).'''
    values = np.array([0.0, 0.25, -0.25, 0.75, 1.0, -1.0])
    wav = tmp_path / "float.wav"
    sf.write(wav, values, 16000, subtype="FLOAT")
    metadata, data = load_wav(wav)
    expected = np.array([0, 8192, -8192, 24575, 32767, -32767], dtype=np.int16).reshape(-1, 1)
    assert (metadata["frames"], metadata["channels"], metadata["subtype"]) == (6, 1, "FLOAT")
    assert np.array_equal(data, expected)


def test_unseen_file_roundtrip():
    '''One held-out real-world wav (not from the corpus) is bit-exact at step 1 and bounded at step 8.'''
    unseen = sorted((Path(__file__).resolve().parents[1] / "data" / "unseen").glob("*.wav"))
    if not unseen:
        pytest.skip("no held-out file yet: drop a wav (about 1 min, not UrbanSound8K) into data/unseen/")
    metadata, original = load_wav(unseen[0])
    back, rate = decompress(compress((original, metadata["samplerate"]), {}))
    assert np.array_equal(back, original) and rate == metadata["samplerate"]
    back8, _ = decompress(compress((original, metadata["samplerate"]), {"quant_step": 8}))
    assert back8.shape == original.shape
    assert np.abs(back8.astype(np.int64) - original.astype(np.int64)).max() <= 4