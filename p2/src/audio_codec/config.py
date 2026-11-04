'''
Central configuration, holds constants and defaults.

Also holds:
    - Wire-format enums and constants
    - Audio bounds
    - Prediction engine knobs
    - Whole-signal defaults
'''
import struct
from enum import IntEnum


class Predictor(IntEnum):
    NONE = 0
    FIXED = 1
    LPC = 2
    LMS = 3


class DCTMode(IntEnum):
    OFF = 0
    QUANTIZE = 1


class EntropyCoder(IntEnum):
    RICE = 0  #room for later coders NOTE: Currently no plans to add more coders


#--- wire format (format.py) ---
MAGIC = b"APAC"
VERSION = 1
HEADER_STRUCT = struct.Struct(">4sBHIQBBBBHBHIB")  #single source of truth for the 33-byte header layout NOTE: This is v1 and might be changed in other versions
MIN_SAMPLE_RATE = 1
MAX_SAMPLE_RATE = 2**32 - 1
MAX_BLOCK_SIZE_LOG2 = 20  #decoder allocation cap: 1 << this residuals per block (the byte field could hold 255 - DoS)
MIN_FRAMES = 0
MAX_FRAMES = 2**64 - 1
MIN_CHANNELS = 1
MAX_CHANNELS = 2**8 - 1
MIN_QUANT_STEP = 1
MAX_QUANT_STEP = 2**16 - 1
MIN_DCT_WINDOW_SIZE = 1
MAX_DCT_WINDOW_SIZE = 2**16 - 1
CONTROL_MODE_MASK = 0b00011111  #5 mode/k bits (0 = raw, 1-31 = Rice, k = mode - 1)
MODE_RAW = 0  #if k is 0 take raw PCM (avoids Rice coding cases where block size grows rather than shrink)
MODE_CONSTANT = 0b00100000  #control bit 5: constant block
MAX_RICE_K = 30  # k = mode - 1
LPC_PARAM_SIZE = 2  #bytes per LPC coefficient (int16); only used for LPC

#Valid predictor order per predictor (order must also fit its 1-byte header field)
PREDICTOR_ORDERS = {
    Predictor.NONE: range(0, 1),
    Predictor.FIXED: range(0, 5),
    Predictor.LPC: range(1, 256),
    Predictor.LMS: range(0, 256),
}

#--- int16 audio bounds (load_file.py quantization, codec.py clipping) ---
MIN_INT16 = -2**15
MAX_INT16 = 2**15 - 1

#--- prediction engine knobs (codec.py) ---
MAX_COEF = MAX_INT16  #LPC coefficient bounds (int16)
MIN_COEF = MIN_INT16
#Fixed polynomial predictor coefficients per order, max order 4. #TODO: consider expanding to higher orders
FIXED_COEFS = {0: [], 1: [1], 2: [2, -1], 3: [3, -3, 1], 4: [4, -6, 4, -1]}
LMS_MU = 0.1   #LMS adaptation strength, should be small to ensure stable convergence
LMS_EPS = 1.0  #normalizer floor, updates kept finite on quiet history
DCT_QUALITY_MIN = 1   #DCTMode.QUANTIZE quality bounds: 100 keeps every bin step at 1
DCT_QUALITY_MAX = 100
DCT_BASE_RAMP = 32   #QUANTIZE base table: bin step = window_size*k//32, tracks the windowed DCT-II gain

#--- whole-signal defaults(codec.py encode_pipeline) ---
DEFAULT_SETTINGS: dict = {
    "predictor": Predictor.FIXED,
    "predictor_order": 2,
    "block_size_log2": 10,
    "quant_step": 1,
    "dct_mode": DCTMode.OFF,
    "dct_window_size": 2**10, #1024
    "dct_param": 0,
    "entropy_coder": EntropyCoder.RICE,
}
