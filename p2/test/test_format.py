#Format-layer tests: header bytes, control byte, container framing.

import struct
from dataclasses import replace

import pytest

from audio_codec.codec.format import (APACHeader, BlockError, FormatError, HeaderError, decode_control, decode_header,
                                      encode_control, encode_header, read_apac, write_apac)
from audio_codec.config import (CONTROL_MODE_MASK, DCTMode, EntropyCoder, HEADER_STRUCT, MAX_BLOCK_SIZE_LOG2,
                                MODE_CONSTANT, Predictor, VERSION)

VALID = APACHeader(
    sample_rate=44100, frames=500, channels=2,
    predictor=Predictor.FIXED, predictor_order=2, block_size_log2=10,
    quant_step=1, dct_mode=DCTMode.OFF, dct_window_size=1024, dct_param=0,
    entropy_coder=EntropyCoder.RICE,
)


def patched_header(header: APACHeader, offset: int, fmt: str, value) -> bytes:
    '''encode_header with one field patched at the byte level - reaches decode_header paths encode_header refuses.'''
    buf = bytearray(encode_header(header))
    struct.pack_into(fmt, buf, offset, value)
    return bytes(buf)


def test_header_roundtrip():
    '''decode_header(encode_header(h)) == h.'''
    assert decode_header(encode_header(VALID)) == VALID


def test_encoded_header_is_33_bytes():
    '''The v1 header is exactly HEADER_STRUCT.size bytes on the wire.'''
    assert len(encode_header(VALID)) == HEADER_STRUCT.size == 33


def test_bad_magic_rejected():
    '''Wrong magic bytes raise FormatError before any field is read.'''
    with pytest.raises(FormatError):
        decode_header(patched_header(VALID, 0, ">4s", b"NOPE"))


def test_truncated_container_rejected():
    '''Every short prefix raises FormatError from decode_header and read_apac.'''
    wire = encode_header(VALID)
    for cut in range(HEADER_STRUCT.size):
        with pytest.raises(FormatError):
            decode_header(wire[:cut])
    with pytest.raises(FormatError):
        read_apac(wire[:HEADER_STRUCT.size - 1])


def test_unsupported_version_rejected():
    '''A version other than VERSION raises FormatError.'''
    with pytest.raises(FormatError):
        decode_header(patched_header(VALID, 4, ">B", VERSION + 1))


def test_wrong_declared_length_rejected():
    '''A header_length field other than 33 raises FormatError.'''
    with pytest.raises(FormatError):
        decode_header(patched_header(VALID, 5, ">H", HEADER_STRUCT.size + 1))


def test_out_of_range_fields_rejected():
    '''Samplerate 0, channels 0 and an unknown predictor byte raise HeaderError naming the field.'''
    for offset, fmt, value, field in ((7, ">I", 0, "sample_rate"),
                                      (19, ">B", 0, "channels"),
                                      (20, ">B", 99, "predictor")):
        with pytest.raises(HeaderError) as excinfo:
            decode_header(patched_header(VALID, offset, fmt, value))
        assert any(issue.field == field for issue in excinfo.value.issues)


def test_block_size_log2_capped():
    '''block_size_log2 above the cap is rejected - bounds the decoder's per-block allocation.'''
    with pytest.raises(HeaderError) as excinfo:
        decode_header(patched_header(VALID, 22, ">B", MAX_BLOCK_SIZE_LOG2 + 1))
    assert any(issue.field == "block_size_log2" for issue in excinfo.value.issues)


def test_dct_quality_bounds_rejected():
    '''QUANTIZE quality outside [1-100], a retired mode byte or OFF with a nonzero param raise HeaderError naming the field.'''
    quantized = replace(VALID, dct_mode=DCTMode.QUANTIZE, dct_param=75)
    assert decode_header(encode_header(quantized)) == quantized      #quality 75 roundtrips
    for param in (0, 101, 2**32 - 1):
        with pytest.raises(HeaderError) as excinfo:
            decode_header(patched_header(quantized, 28, ">I", param))
        assert any(issue.field == "dct_param" for issue in excinfo.value.issues)
    with pytest.raises(HeaderError) as excinfo:
        decode_header(patched_header(VALID, 25, ">B", 2))            #retired mode byte (CUTOFF/ATTENUATE range)
    assert any(issue.field == "dct_mode" for issue in excinfo.value.issues)
    with pytest.raises(HeaderError) as excinfo:
        decode_header(patched_header(VALID, 28, ">I", 7))            #OFF must carry param 0
    assert any(issue.field == "dct_param" for issue in excinfo.value.issues)


def test_control_byte_roundtrip():
    '''Modes 0..31 and MODE_CONSTANT roundtrip and never set reserved bits.'''
    for mode in list(range(CONTROL_MODE_MASK + 1)) + [MODE_CONSTANT]:
        assert decode_control(encode_control(mode)) == mode
        assert not encode_control(mode) & ~(CONTROL_MODE_MASK | MODE_CONSTANT)


def test_control_byte_reserved_bits_rejected():
    '''Reserved bits set or the constant bit mixed with mode bits raise BlockError; so do invalid encode modes.'''
    for bad in (0b0100_0000, 0b1000_0000, 0b1100_0000):
        with pytest.raises(BlockError):
            decode_control(bad)
    with pytest.raises(BlockError):
        decode_control(MODE_CONSTANT | 1)                        #constant bit mixed with a mode
    for bad in (-1, CONTROL_MODE_MASK + 2):                      #not MODE_CONSTANT and outside [0-31]
        with pytest.raises(BlockError):
            encode_control(bad)


def test_container_roundtrip_keeps_payload_bytes():
    '''write_apac/read_apac roundtrips the header and keeps the payload byte-exact, empty included.'''
    payload = bytes(range(256)) * 3
    assert read_apac(write_apac(VALID, payload)) == (VALID, payload)
    assert read_apac(write_apac(VALID, b"")) == (VALID, b"")


def test_empty_frames_accepted():
    '''frames == 0 is valid at format level: a header-only container decodes.'''
    empty = replace(VALID, frames=0)
    assert read_apac(write_apac(empty, b"")) == (empty, b"")
