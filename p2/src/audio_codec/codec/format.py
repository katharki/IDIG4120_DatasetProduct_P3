#Contains fileformat specifications and related utilities for the audio codec.

'''
    Format:

    Header | Data 

    #Note: Header optimization is more impactful in our case as most files are relatively short (not very impactful, but still worth considering)

    Header:
        | Magic bytes (4 bytes): "APAC" / 0x41504143 |
        | Version (1 byte) |
        | Header length (2 bytes) |
        | Sample rate (4 bytes) |
        | Sample count/frames (8 bytes) #TODO: consider 4 Byte (about 27 hours at 44.1 kHz) |
        | Channels (1 byte) #TODO: consider sub-byte as 255 channels will likely never be needed and try to align with another field for better packing |
        | Predictor (1 byte) #TODO: as only 3 predictors are planned at most this might be reduced to 2 bits and packed with other fields (6bit channeld + 2bit predictor?) |
        | Predictor Order (1 byte) |
        | Block size (1 byte) #TODO: represents 2**n samples (only 256 possible block sizes, might consider making smaller and packing as anything above 2**16 (65536) might be too many samples) |
        | Quantization step (for residuals) (2 bytes) |
        | DCT mode (1 byte) (off, quantize) |
        | DCT window size (2 bytes) #TODO: examine concequences of larger or smaller window sizes and reconsider |
        | DCT parameter (4 bytes) (spectral quality 1-100, 0 when off) |
        | Entropy coder (1 byte) (Rice coding) |
        #No reserved space as this is handled by version
        # Total header size = 4 + 1 + 2 + 4 + 8 + 1 + 1 + 1 + 1 + 2 + 1 + 2 + 4 + 1 = 33 bytes, Wav header = 44 bytes

    Data:
        Follows format:
        | Block 1 |
        | Block 2 |
        | ...     |

        Block:
            | Control (1 byte) | Predictor parameters (variable) | Block data |

        Control byte:
            | Reserved (2 bits) | Constant (1 bit) | Encoding mode / Rice k (5 bits) |
            -> |xx|c|xxxxx|
            
            Rice k / mode: 0 = raw PCM (int16)
            1-31 = Rice-coded residuals (k = value - 1)
            32 = constant block (bit 5 set), storing single int16 value to represent a constant block.

        Predictor parameters:
            
            Raw PCM: None
            Fixed: None (order stored in global header)
            LPC: Quantized coefficients (per block) (depends on predictor order)
            LMS: None (deterministic initialization/adaptation)

        Block data:
            Raw PCM:
                | Uncompressed int16 samples |

            Rice:
                | Rice-encoded quantized residuals | Byte padding |

'''

from __future__ import annotations

from audio_codec import shared
from dataclasses import dataclass, replace

from audio_codec.shared.validation import ValidationIssue


#Wire-format constants, enums and limits live in audio_codec.config (single project config);
#imported here so every existing `from audio_codec.codec.format import ...` keeps resolving.
from audio_codec.config import (CONTROL_MODE_MASK, DCT_QUALITY_MAX, DCT_QUALITY_MIN, DCTMode, EntropyCoder, HEADER_STRUCT,
                                LPC_PARAM_SIZE, MAGIC, MAX_BLOCK_SIZE_LOG2, MAX_CHANNELS, MAX_DCT_WINDOW_SIZE,
                                MAX_FRAMES, MAX_QUANT_STEP, MAX_RICE_K, MAX_SAMPLE_RATE, MIN_CHANNELS,
                                MIN_DCT_WINDOW_SIZE, MIN_FRAMES, MIN_QUANT_STEP, MIN_SAMPLE_RATE, MODE_CONSTANT,
                                MODE_RAW, PREDICTOR_ORDERS, Predictor, VERSION)


@dataclass(frozen=True)
class APACHeader:
    '''
        APAC audio codec header representation.

        Byte layout:
            |Offset|Size|Field|struct|
            |------|----|-----|------|
            | 0    | 4  | Magic `"APAC"`    | `4s`|
            | 4    | 1  | Version           | `B` |
            | 5    | 2  | Header length     | `H` |
            | 7    | 4  | Sample rate       | `I` |
            | 11   | 8  | Frames            | `Q` |
            | 19   | 1  | Channels          | `B` |
            | 20   | 1  | Predictor         | `B` |
            | 21   | 1  | Predictor order   | `B` |
            | 22   | 1  | Block size (2^n)  | `B` |
            | 23   | 2  | Quantization step | `H` |
            | 25   | 1  | DCT mode          | `B` |
            | 26   | 2  | DCT window size   | `H` |
            | 28   | 4  | DCT parameter     | `I` |
            | 32   | 1  | Entropy coder     | `B` |

    '''
    sample_rate: int
    frames: int
    channels: int
    predictor: Predictor
    predictor_order: int
    block_size_log2: int
    quant_step: int
    dct_mode: DCTMode
    dct_window_size: int
    dct_param: int
    entropy_coder: EntropyCoder


class FormatError(ValueError):
    '''Base class for APAC format errors.'''


class HeaderError(FormatError):
    '''
        Raised when a header is invalid.
        Can hold multiple validation issues.
    '''

    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = list(issues)
        summary = "; ".join(str(issue) for issue in self.issues)
        super().__init__(f"{len(self.issues)} invalid header field(s): {summary}")


class BlockError(FormatError):
    '''
        Raised when a block is malformed (bad mode, reserved bits set, truncation, bad params size).
    '''


def header_issues(header: APACHeader) -> list[ValidationIssue]:
    '''
        Validate APACHeader and return list of issues.

        Parameters:
            - header (APACHeader): The header to validate.

        Returns:
            - list[ValidationIssue]: List of issues found, empty if valid.
    '''
    checks: list[tuple[bool, str, str]] = [
        (
            MIN_SAMPLE_RATE <= header.sample_rate <= MAX_SAMPLE_RATE, 
            "sample_rate", 
            f"must be in the range [{MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE}], "
            f"got {header.sample_rate}"
        ),
        (
            MIN_FRAMES <= header.frames <= MAX_FRAMES,
            "frames",
            f"must be in the range [{MIN_FRAMES}-{MAX_FRAMES}], got {header.frames}"
        ),
        (
            MIN_CHANNELS <= header.channels <= MAX_CHANNELS,
            "channels",
            f"must be in the range [{MIN_CHANNELS}-{MAX_CHANNELS}], got {header.channels}"
        ),
        (
            0 < header.block_size_log2 <= MAX_BLOCK_SIZE_LOG2,
            "block_size_log2",
            f"must be in the range [1-{MAX_BLOCK_SIZE_LOG2}], got {header.block_size_log2}"
        ),
        (
            MIN_QUANT_STEP <= header.quant_step <= MAX_QUANT_STEP,
            "quant_step",
            f"must be in the range [{MIN_QUANT_STEP}-{MAX_QUANT_STEP}], got {header.quant_step}"
        ),
        (
            MIN_DCT_WINDOW_SIZE <= header.dct_window_size <= MAX_DCT_WINDOW_SIZE,
            "dct_window_size",
            f"must be in the range [{MIN_DCT_WINDOW_SIZE}-{MAX_DCT_WINDOW_SIZE}], got {header.dct_window_size}"
        ),
        (
            header.dct_param == 0 if header.dct_mode == DCTMode.OFF
            else DCT_QUALITY_MIN <= header.dct_param <= DCT_QUALITY_MAX,
            "dct_param",
            f"must be 0 for OFF or a spectral quality in [{DCT_QUALITY_MIN}-{DCT_QUALITY_MAX}] for QUANTIZE, got {header.dct_param}"
        ),
    ]
    issues = [ValidationIssue(field, message) for ok, field, message in checks if not ok]

    #Check enums
    if header.predictor not in Predictor.__members__.values():
        issues.append(ValidationIssue("predictor", f"unknown predictor value: {header.predictor}"))
    if header.dct_mode not in DCTMode.__members__.values():
        issues.append(ValidationIssue("dct_mode", f"unknown DCT mode value: {header.dct_mode}"))
    if header.entropy_coder not in EntropyCoder.__members__.values():
        issues.append(ValidationIssue("entropy_coder", f"unknown entropy coder value: {header.entropy_coder}"))

    #Valid order for given predictor
    if header.predictor in PREDICTOR_ORDERS:
        valid_orders = PREDICTOR_ORDERS[header.predictor]
        if header.predictor_order not in valid_orders:
            issues.append(ValidationIssue("predictor_order", f"must be in {valid_orders.start}..{valid_orders.stop - 1} for predictor {Predictor(header.predictor).name}, got {header.predictor_order}"))

    return issues


def validate_header(header: APACHeader) -> None:
    '''
    Validate an APACHeader, raises HeaderError.

    Parameters:
        - header (APACHeader): The header to validate.
    '''
    issues = header_issues(header)
    if issues:
        raise HeaderError(issues)


def encode_header(header: APACHeader) -> bytes:
    '''
        Pack an APACHeader into bytes.

        Parameters:
            - header (APACHeader): The header to encode.

        Returns:
            - bytes: The encoded header bytes.

        Raises:
            - HeaderError: If the header is invalid.
    '''
    
    validate_header(header)

    return HEADER_STRUCT.pack(
        MAGIC, VERSION, HEADER_STRUCT.size,
        header.sample_rate, header.frames, header.channels,
        header.predictor, header.predictor_order, header.block_size_log2,
        header.quant_step, header.dct_mode, header.dct_window_size,
        header.dct_param, header.entropy_coder,
    )


def decode_header(data: bytes) -> APACHeader:
    '''
        Unpack an APACHeader from the start of data.

        Parameters:
            - data (bytes): The byte sequence containing the header at the start.

        Returns:
            - APACHeader: The decoded header object.
            
        Raises:
            - FormatError: If the data is truncated or has bad magic/version/length.
            - HeaderError: If the decoded header is invalid.
    '''
    shared.logger.debug("Decoding APAC header from data of length %d", len(data))

    if len(data) < HEADER_STRUCT.size:
        raise FormatError(f"truncated header: expected {HEADER_STRUCT.size} bytes, got {len(data)}")
    
    magic, version, header_len, sample_rate, frames, channels, predictor, predictor_order, block_size_log2, quant_step, dct_mode, dct_window_size, dct_param, entropy_coder = HEADER_STRUCT.unpack_from(data)

    shared.logger.debug(
        f"Unpacked header fields: magic={magic!r}, version={version!r}, "
        f"header_len={header_len}, sample_rate={sample_rate}, frames={frames}, "
        f"channels={channels}, predictor={predictor}, predictor_order={predictor_order}, "
        f"block_size_log2={block_size_log2}, quant_step={quant_step}, dct_mode={dct_mode}, "
        f"dct_window_size={dct_window_size}, dct_param={dct_param}, entropy_coder={entropy_coder}"
    )
    
    if magic != MAGIC:
        raise FormatError(f"not an APAC file: bad magic bytes {magic!r}")
    
    if version != VERSION:
        raise FormatError(f"unsupported version {version}, expected {VERSION}")
    
    if header_len != HEADER_STRUCT.size:
        raise FormatError(f"bad header length {header_len}, expected {HEADER_STRUCT.size}")
    
    header = APACHeader(
        sample_rate=sample_rate, frames=frames, channels=channels,
        predictor=predictor, predictor_order=predictor_order, block_size_log2=block_size_log2,
        quant_step=quant_step, dct_mode=dct_mode, dct_window_size=dct_window_size,
        dct_param=dct_param, entropy_coder=entropy_coder
    )

    issues = header_issues(header)

    if issues:
        raise HeaderError(issues)

    shared.logger.debug("Header validation passed")
    
    #Ints to enums
    return replace(header, predictor=Predictor(predictor), dct_mode=DCTMode(dct_mode), entropy_coder=EntropyCoder(entropy_coder))


def predictor_params_size(header: APACHeader) -> int:
    '''
        Returns bytesize of predictor coefficients for LPC, else 0.

        Parameters:
            - header (APACHeader): The header describing the blocks.

        Returns:
            - int: LPC_PARAM_SIZE * predictor_order for LPC, 0 for every other predictor.
    '''
    if header.predictor == Predictor.LPC:
        return LPC_PARAM_SIZE * header.predictor_order
    return 0


def encode_control(mode: int) -> int:
    '''
        Builds control block with 2 reserved bits, 1 constant block bit and 5 used for Rice parameter k (0-30 (mode-1)) (raw PCM if mode is 0, constant block if mode is MODE_CONSTANT).

        Parameters:
            - mode: Value to encode into the control byte.
        
        Returns:
            - The encoded control byte as an integer.

        Raises:
            - BlockError: If mode not in [0-31] and not MODE_CONSTANT.
    '''
    if mode != MODE_CONSTANT and not 0 <= mode <= CONTROL_MODE_MASK:
        raise BlockError(f"invalid control mode, got {mode} expected [0-{CONTROL_MODE_MASK}] or {MODE_CONSTANT}")
    
    return mode


def decode_control(control: int) -> int:
    '''
        Decodes control byte to extract the mode/k value.
    
        Parameters:
            - control: The control byte to decode.

        Returns:
            - The extracted mode/k value as an integer.

        Raises:
            - BlockError: If reserved bits are set in the control byte or the constant bit is mixed with mode bits.
    '''
    if control & ~(CONTROL_MODE_MASK | MODE_CONSTANT):
        raise BlockError(f"invalid control byte, reserved bits set: {control:08b}")

    if control & MODE_CONSTANT and control & CONTROL_MODE_MASK:
        raise BlockError(f"invalid control byte, constant bit mixed with mode bits: {control:08b}")
    
    return control


def encode_block(mode: int, predictor_params: bytes, block_data: bytes) -> bytes:
    '''
        Frame one block as control byte + predictor parameters + block data.

        Parameters:
            - mode: The mode/k value for the block.
            - predictor_params: The predictor parameters as bytes.
            - block_data: The actual block data as bytes.

        Returns:
            - The encoded block as bytes.

        Raises:
            - BlockError: If mode not in [0-31] (and not MODE_CONSTANT), raw PCM block data length is not a multiple of 2, or a constant block does not carry exactly 2 data bytes and no predictor parameters.
    '''
    if mode == MODE_RAW and len(block_data) % 2:
        raise BlockError(f"invalid raw PCM block length, got {len(block_data)} bytes, expected multiple of 2")

    if mode == MODE_CONSTANT and (len(block_data) != 2 or predictor_params):
        raise BlockError(f"invalid constant block: expected 2 data bytes and no predictor parameters, got {len(block_data)} data bytes and {len(predictor_params)} parameter bytes")

    #TODO: add predictor parameter validation for non-fixed modes
    return bytes([encode_control(mode)]) + predictor_params + block_data


def decode_block(data: bytes, params_size: int = 0) -> tuple[int, bytes, bytes]:
    '''
        Split one block into (mode, predictor parameters, block data), params_size is the predictor parameter byte length (predictor_params_size(header)). Constant blocks (MODE_CONSTANT) carry no predictor parameters.

        Parameters:
            - data: The encoded block as bytes.
            - params_size: The byte length of the predictor parameters (predictor_params_size(header)).

        Returns:
            - A tuple of (mode, predictor parameters, block data).

        Raises:
            - BlockError: If params_size is negative, the block is truncated, or the control byte is invalid.
    '''
    if params_size < 0:
        raise BlockError(f"invalid predictor parameter size: got {params_size}, must be >= 0")
    
    if len(data) < 1:
        raise BlockError(f"truncated block: expected at least 1 byte, got {len(data)}")
    mode = decode_control(data[0])
    if mode == MODE_CONSTANT:
        return mode, b'', data[1:]
    if len(data) < 1 + params_size:
        raise BlockError(f"truncated block: expected at least {1 + params_size} bytes, got {len(data)}")
    return mode, data[1:1 + params_size], data[1 + params_size:]

def write_apac(header: APACHeader, payload: bytes) -> bytes:
    '''
        Serializes an APAC container: encoded header followed by the payload bytes.

        Pure byte-level function - file I/O lives in the loader (shared.load_file, save_apac).

        Parameters:
            - header: The APAC header to encode.
            - payload: The payload bytes to place after the header.

        Returns:
            - bytes: The full APAC container (encoded header + payload).
    '''

    return encode_header(header) + payload


def read_apac(data: bytes) -> tuple[APACHeader, bytes]:
    '''
        Parses a full APAC container into its header and the payload bytes after it.

        Pure byte-level function - file I/O lives in the loader (shared.load_file, load_apac).

        Parameters:
            - data: The APAC container bytes (encoded header + data section).

        Returns:
            - A tuple of (APACHeader, payload bytes).

        Raises:
            - FormatError: If the container is truncated or the header is invalid.
    '''
    if len(data) < HEADER_STRUCT.size:
        raise FormatError(f"truncated APAC container: expected at least {HEADER_STRUCT.size} bytes, got {len(data)}")
    return decode_header(data), data[HEADER_STRUCT.size:]