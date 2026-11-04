#Codec pipeline: decorrelates channels into coding planes (blocking, prediction and quantization layer on next).

'''
    Channel decorrelation into coding planes:

        Pair-wise correlation for every pair of channels.
        Each pair is used to create a mid and side plane. (M = L+R, S = L-R)
        This makes it possible to reconstruct the original by taking the side plane difference and adding it to half of the mid plane.

    Data section (block-major, plane-minor):

        encode_blocks walks every block period for every plane and frames one
        block as control | params | data. Rice blocks carry quantized residual
        indices at the exact-cost k, raw blocks carry the plane's source channel
        as big-endian int16, constant blocks carry one big-endian int16 plane
        value for the whole block.

    Notes:
        Mono-channel and odd leftovers pass with an int32 copy of themselves.
        Stored as int32 as rice coded planes will have the same size when compressed as if they were int16.
'''

import math
import struct
from itertools import zip_longest

import numpy as np

from audio_codec.codec.format import (APACHeader, BlockError, decode_block, encode_block, read_apac, validate_header,
                                      write_apac)
from audio_codec.codec.entropy import best_k, rice_decode, rice_encode, rice_size
from audio_codec.config import (DEFAULT_SETTINGS, DCT_BASE_RAMP, DCT_QUALITY_MAX, DCT_QUALITY_MIN,
                                DCTMode, EntropyCoder, FIXED_COEFS, LMS_EPS, LMS_MU, LPC_PARAM_SIZE, MAX_COEF,
                                MAX_INT16, MIN_COEF, MIN_INT16, MODE_CONSTANT, MODE_RAW, PREDICTOR_ORDERS, Predictor)

def to_planes(samples: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    '''
        Splits int16 samples into int32 planes and their int16 source channels.

        Pairs (0,1), (2,3), ... become two planes each: mid M = L+R and side S = L-R.
        For fallback, each plane keeps the source channel as an int16 array.
        Leftover channels (odd or mono) pass through as int32 copies of themselves.

        Parameters:
            - samples (np.ndarray): (frames, ch) int16 audio

        Returns:
            - tuple[list[np.ndarray], list[np.ndarray]]: (int32 planes, int16 sources) per channel

        Raises:
            - ValueError: if samples is not a 2D int16 array or has no channels
    '''
    src = np.asarray(samples)
    if src.ndim != 2:
        raise ValueError(f"expected 2D (frames, channels), got {src.ndim}D")
    if src.shape[1] == 0:
        raise ValueError("expected at least one channel")
    if src.dtype != np.int16:
        raise ValueError(f"expected int16 samples, got {src.dtype}")

    x = src.astype(np.int32)
    planes, sources = [], []
    for ch in range(0, x.shape[1] - 1, 2):
        left, right = x[:, ch], x[:, ch + 1]
        planes.append(left + right)   #mid plane, M = L+R
        planes.append(left - right)   #side plane, S = L-R
        sources.append(src[:, ch])
        sources.append(src[:, ch + 1])
    if x.shape[1] % 2:                #last channel passes if odd channels
        planes.append(x[:, -1])
        sources.append(src[:, -1])
    return planes, sources


def quantize_residual(res: int, step: int) -> int:
    '''
        Quantizes a prediction residual to a step index.

        q = (2*res + step) // (2*step) picks the nearest multiple of step:
        step = 1 is the exact identity (the lossless anchor), larger steps
        can have a reconstruction error of up to step // 2.

        Parameters:
            - res (int): prediction residual x - pred
            - step (int): quantization step size

        Returns:
            - int: quantized step index q

        Raises:
            - ValueError: if step < 1
    '''
    if step < 1:
        raise ValueError(f"expected step >= 1, got {step}")
    return (2 * res + step) // (2 * step)


def reconstruct_sample(pred: int, q: int, step: int) -> int:
    '''
        Reconstructs a sample from its prediction and quantized step index.

        Inverse of quantize_residual: x_hat = pred + q * step. Encoder and
        decoder push the same x_hat, so their histories stay identical.

        Parameters:
            - pred (int): prediction made from reconstructed history
            - q (int): quantized step index
            - step (int): quantization step size

        Returns:
            - int: reconstructed sample x_hat

        Raises:
            - ValueError: if step < 1
    '''
    if step < 1:
        raise ValueError(f"expected step >= 1, got {step}")
    return pred + q * step


def lpc_analyze(block: np.ndarray, order: int) -> list[int]:
    '''
        Estimates LPC coefficients for one block of plane samples.

        Uses a Hann-windowed biased autocorrelation and Levinson-Durbin
        recursion. Returns clamped Q15 coefficients ordered for newest-first
        history (lag 1 first). Silent or empty blocks, and exhausted error
        energy, produce zero coefficients.

        Parameters:
            - block (np.ndarray): plane samples for this block
            - order (int): LPC order (1..255)

        Returns:
            - list[int]: Q15 coefficients, length `order`

        Raises:
            - ValueError: if order < 1
    '''
    
    if order < 1:
        raise ValueError(f"expected order >= 1, got {order}")

    x = np.asarray(block, dtype=np.float64).ravel()
    n = x.size
    xw = x * np.hanning(n) if n > 2 else x   #hann(2) is all zero, tiny blocks stay raw

    r = [0.0] * (order + 1)   #biased autocorrelation, lag 0 first
    for j in range(min(order, n - 1) + 1):
        r[j] = float(np.dot(xw[:n - j], xw[j:]))

    if r[0] <= 0.0:
        return [0] * order   #silent or empty block: nothing to model

    a = [0.0] * (order + 1)   #a[i] = lag i coefficient, a[0] unused
    err = r[0]
    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc -= a[j] * r[i - j]
        k = acc / err
        prev = list(a)
        a[i] = k
        for j in range(1, i):
            a[j] = prev[j] - k * prev[i - j]
        err *= 1.0 - k * k
        if err <= 0.0:
            return [0] * order   #perfectly predictable: no stable solve

    return [min(32767, max(-32768, round(c * 32768))) for c in a[1:]]   #Q15, clamped into int16


def lpc_params_encode(coefs: list[int]) -> bytes:
    '''
        Packs LPC coefficients for the wire: big-endian int16 per coefficient.

        Parameters:
            - coefs (list[int]): Q15 coefficients from lpc_analyze

        Returns:
            - bytes: packed predictor params, LPC_PARAM_SIZE bytes per coefficient

        Raises:
            - ValueError: if a coefficient is outside int16
    '''
    for c in coefs:
        if not MIN_COEF <= c <= MAX_COEF:
            raise ValueError(f"expected int16 coefficient, got {c}")
    return struct.pack(f'>{len(coefs)}h', *coefs)


def lpc_params_decode(data: bytes, order: int) -> list[int]:
    '''
        Unpacks LPC coefficients from the wire: big-endian int16 per coefficient.

        Parameters:
            - data (bytes): packed predictor params
            - order (int): number of coefficients to unpack

        Returns:
            - list[int]: Q15 coefficients

        Raises:
            - ValueError: if data does not hold exactly `order` coefficients
    '''
    if len(data) != order * LPC_PARAM_SIZE:
        raise ValueError(f"expected {order * LPC_PARAM_SIZE} bytes for {order} coefficients, got {len(data)}")
    return list(struct.unpack(f'>{order}h', data))


def dct_quant_steps(quality: int, window_size: int) -> np.ndarray:
    '''
        Per-bin quantization steps for DCTMode.QUANTIZE: JPEG-style quality.

        One scalar 1-100 scales a frequency-weighted base table (1 at DC,
        window_size/32 per bin index, about 32700 at the top of the default
        1024 window), so lower quality coarsens the spectrum gradually, high
        frequencies always taking the larger steps, and no bin is ever walled
        off. 100 keeps every step at 1, the near-transparent anchor. The table
        is derived, never stored: quality and window size rebuild it
        identically anywhere, so the wire format stays unchanged.

        Parameters:
            - quality (int): spectral quality in [1-100]
            - window_size (int): DCT window length in samples

        Returns:
            - np.ndarray: int64 quantization step per DCT bin

        Raises:
            - ValueError: if window_size < 1 or quality is outside [1-100]
    '''
    if window_size < 1:
        raise ValueError(f"expected window_size >= 1, got {window_size}")
    if not DCT_QUALITY_MIN <= quality <= DCT_QUALITY_MAX:
        raise ValueError(f"expected quality in [{DCT_QUALITY_MIN}-{DCT_QUALITY_MAX}], got {quality}")
    k = np.arange(window_size)
    base = 1 + window_size * k // DCT_BASE_RAMP   #frequency-weighted base table, scales with the window
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality   #libjpeg quality scaling
    return np.maximum(1, (scale * base + 50) // 100)


def dct_shaping(plane: np.ndarray, window_size: int, quality: int) -> np.ndarray:
    '''
        Encoder-side lossy pre-shaping of one coding plane via windowed DCT.

        Shapes each window before encoding; the decoder does not apply this
        operation. Windows are zero-padded when needed. QUANTIZE, the only
        mode beyond OFF, snaps each bin to a multiple of its per-bin step, a
        JPEG-style quality 1-100 in `quality` (steps from dct_quant_steps).
        Even quality 100 runs the transform round trip, so it can differ by
        rounding noise (steps are all 1 there; soft content stays bit-exact).

        Parameters:
            - plane (np.ndarray): int32 coding plane, oldest sample first
            - window_size (int): DCT window length in samples
            - quality (int): spectral quality in [1-100]

        Returns:
            - np.ndarray: shaped int32 plane, `plane` itself for empty planes

        Raises:
            - ValueError: if window_size < 1 or quality is outside [1-100]
    '''
    if window_size < 1:
        raise ValueError(f"expected window_size >= 1, got {window_size}")
    if len(plane) == 0:
        return plane

    steps = dct_quant_steps(quality, window_size)
    n = window_size
    k = np.arange(n)
    twiddle = np.exp(-1j * np.pi * k / (2 * n))    #forward DCT-II: even-extension FFT + phase
    weights = np.where(k == 0, 0.5, 1.0)           #inverse DCT-III: DC carries half weight
    back = np.exp(1j * np.pi * k / (2 * n))
    out = np.zeros(len(plane), dtype=np.int32)
    for start in range(0, len(plane), n):
        x = plane[start:start + n].astype(np.float64)
        real = x.size
        if real < n:
            x = np.concatenate([x, np.zeros(n - real)])   #zero-pad the last window
        coefs = (np.fft.fft(np.concatenate([x, x[::-1]]))[:n] * twiddle).real
        coefs = np.rint(coefs / steps) * steps
        z = np.fft.ifft(np.concatenate([coefs * weights * back, np.zeros(n)]))[:n]
        y = 2.0 * z.real[:real]
        out[start:start + real] = np.clip(np.round(y), -2**31, 2**31 - 1).astype(np.int32)
    return out


class PlaneEngine:
    '''
        Closed-loop predictor for one coding plane.

        Both sides run the identical per-sample sequence predict -> quantize ->
        reconstruct -> push over identical inputs, so their reconstructed history
        stays bit-identical and quantized indices decode back exactly.
        History is per plane: newest sample first, empty history predicts 0,
        and it carries across blocks within the stream.

        Parameters:
            - predictor (Predictor): predictor mode from the header
            - order (int): predictor order

        Raises:
            - ValueError: if the order is outside PREDICTOR_ORDERS for the mode
    '''

    def __init__(self, predictor: Predictor, order: int):
        valid = PREDICTOR_ORDERS[predictor]

        if order not in valid:
            raise ValueError(f"expected order {valid.start}..{valid.stop - 1} for predictor {predictor.name}, got {order}")
        
        self.predictor = predictor
        self.order = order
        self._coefs = list(FIXED_COEFS[order]) if predictor == Predictor.FIXED else []
        self._lms_w: list[float] = [0.0] * order if predictor == Predictor.LMS else []   #NLMS weights
        self._history: list[int] = []   #newest sample first

    def _predict(self) -> int:
        '''
            Predicts the next sample from reconstructed history.

            NONE predicts 0.
            FIXED correlates its integer coefficients with the newest `order`
            history samples (set at construction).
            LPC correlates its Q15 coefficients the same way, then rescales the
            MAC sum with the arithmetic shift down 15 (the Q15 fraction needs
            it; both sides apply the identical shift, so predictions stay
            bit-identical).
            LMS correlates its NLMS weights with the same window and floors the result.

            Returns:
                - int: prediction pred
        '''
        if self.predictor == Predictor.NONE:
            return 0
        
        if self.predictor == Predictor.FIXED:
            return sum(c * h for c, h in zip(self._coefs, self._history))

        if self.predictor == Predictor.LPC:
            return (sum(c * h for c, h in zip(self._coefs, self._history))) >> 15   #Q15 rescale, floor shift
        
        if self.predictor == Predictor.LMS:
            return math.floor(sum(w * h for w, h in zip(self._lms_w, self._history)))

    def push(self, x_hat: int) -> None:
        '''
            Appends a reconstructed sample to the history, newest first.

            Keeps only the newest `order` samples (order 0 keeps none).
            In LMS mode this adapts the weights first.

            Parameters:
                - x_hat (int): reconstructed sample to append
        '''

        if self.predictor == Predictor.LMS and self.order > 0:
            pred = sum(w * x for w, x in zip(self._lms_w, self._history))
            err = x_hat - pred
            norm = LMS_EPS + sum(x * x for x in self._history)
            self._lms_w = [w + (0.0 if h is None else LMS_MU * err * h / norm)
                           for w, h in zip_longest(self._lms_w, self._history)]

        self._history.insert(0, x_hat)
        del self._history[self.order:]

    def push_raw(self, values: np.ndarray) -> None:
        '''
            Appends a raw or constant block's samples to the history, newest first.

            A raw block redefines the plane to carry its source channel and a
            constant block replays one plane value; either way both sides push
            the same values outside the prediction loop. Should not adapt predictor state
            such as LMS weights; adaptation resumes on the next
            predicted block.

            Parameters:
                - values (np.ndarray): raw source samples for this block, oldest first
        '''
        if len(values) == 0:
            return
        keep = values[-self.order:] if self.order else []
        self._history = [int(v) for v in reversed(keep)]   #newest first, capped at order

    def set_coeffs(self, coefs: list[int]) -> None:
        '''
            Adopts per-block LPC coefficients for prediction.

            Both sides call this with identical wire values before coding a
            block, so their predictions stay bit-identical. FIXED coefficients
            are set once at construction and cannot be replaced.

            Parameters:
                - coefs (list[int]): Q15 coefficients, length `order`

            Raises:
                - ValueError: if not in LPC mode, wrong coefficient count, or a coefficient outside int16
        '''
        if self.predictor != Predictor.LPC:
            raise ValueError(f"expected predictor {Predictor.LPC.name}, got {self.predictor.name}")
        if len(coefs) != self.order:
            raise ValueError(f"expected {self.order} coefficients, got {len(coefs)}")
        for c in coefs:
            if not MIN_COEF <= c <= MAX_COEF:
                raise ValueError(f"expected int16 coefficient, got {c}")
        self._coefs = list(coefs)

    def checkpoint(self) -> tuple[list[int], list[int], list[float]]:
        '''
            Captures the mutable engine state for a later rollback.

            The encoder checkpoints before trialing a block it may store raw.
            If the raw block is kept, the engine rolls back to checkpoint and replays the raw samples.
            LMS weights are part of the snapshot: the trial adapts them, push_raw never does.

            Returns:
                - tuple[list[int], list[int], list[float]]: (history, coefficients, LMS weights) snapshot
        '''
        return (list(self._history), list(self._coefs), list(self._lms_w))

    def rollback(self, state: tuple[list[int], list[int], list[float]]) -> None:
        '''
            Restores an engine state captured by checkpoint.

            Parameters:
                - state (tuple[list[int], list[int], list[float]]): snapshot from checkpoint()
        '''
        self._history, self._coefs, self._lms_w = list(state[0]), list(state[1]), list(state[2])

    def code_block(self, block: np.ndarray, step: int) -> list[int]:
        '''
            Encodes one block of plane samples into quantized indices.

            Per sample: pred from reconstructed history, res = x - pred,
            q = quantize_residual(res, step), x_hat = reconstruct_sample(pred, q, step),
            push(x_hat). Step 1 encodes losslessly; larger steps bound the error.

            Parameters:
                - block (np.ndarray): plane samples for this block
                - step (int): quantization step size

            Returns:
                - np.ndarray: quantized indices (int64), ready for entropy coding
        '''
        #vectorized fast path: the FIXED filters are exactly (1 - z^-1)^order, so at
        #step 1 the residuals are the order-fold difference of the history-extended
        #block and x_hat == x sample for sample (integer-exact, no per-sample loop)
        if step == 1 and self.predictor in (Predictor.NONE, Predictor.FIXED):
            return self._code_block_diff(block)

        indices = []
        
        for x in block:
            pred = self._predict()
            q = quantize_residual(int(x) - pred, step)
            indices.append(q)
            self.push(reconstruct_sample(pred, q, step))
        
        return np.asarray(indices, dtype=np.int64)

    def _code_block_diff(self, block: np.ndarray) -> np.ndarray:
        '''
            Vectorized code_block for the NONE/FIXED predictors at step 1.

            Prepends the engine history (missing entries read as 0, matching the
            zip-truncated MAC in _predict) and takes the order-fold difference:
            residual = (1 - z^-1)^order x. x_hat equals x, so the newest-first
            history is just the block's last `order` samples.

            Parameters:
                - block (np.ndarray): plane samples for this block

            Returns:
                - np.ndarray: residual indices (int64), identical to the closed loop
        '''
        x = np.asarray(block, dtype=np.int64)
        n = self.order
        if n:
            prefix = [0] * (n - len(self._history)) + [int(v) for v in reversed(self._history)]
            a = np.concatenate([np.asarray(prefix, dtype=np.int64), x])
            indices = np.diff(a, n=n)
            if x.size:
                self._history = [int(v) for v in x[-1:-n - 1:-1]]
        else:
            indices = x
        return indices

    def decode_block(self, indices: list[int], step: int) -> list[int]:
        '''
            Decodes quantized indices back into samples, mirroring code_block.

            Per sample: pred, x_hat = reconstruct_sample(pred, q, step), push(x_hat).
            The same engine class runs on both sides, so identical inputs keep
            encoder and decoder histories bit-identical.

            Parameters:
                - indices (list[int]): quantized indices for this block
                - step (int): quantization step size

            Returns:
                - np.ndarray: reconstructed plane samples (int64)
        '''
        #vectorized fast path: u = q*step is known and the FIXED filters equal
        #(1 - z^-1)^order, so the closed loop inverts to an order-fold cumulative
        #sum of u seeded from the history's successive differences (any step)
        if self.predictor in (Predictor.NONE, Predictor.FIXED):
            return self._decode_block_cumsum(indices, step)

        samples = []

        for q in indices:
            pred = self._predict()
            x_hat = reconstruct_sample(pred, q, step)
            samples.append(x_hat)
            self.push(x_hat)

        return np.asarray(samples, dtype=np.int64)

    def _decode_block_cumsum(self, indices: np.ndarray, step: int) -> np.ndarray:
        '''
            Vectorized decode_block for the NONE/FIXED predictors.

            With A_order = q*step and A_t[i] = A_t[i-1] + A_(t+1)[i], integrating
            down to A_0 reconstructs x_hat: each level is one np.cumsum seeded with
            the value of (1 - z^-1)^t x_hat just before the block, a binomial mix
            of the history entries (missing entries read as 0). All int64
            arithmetic, so the result matches the per-sample closed loop bit for bit.

            Parameters:
                - indices (np.ndarray): quantized indices for this block
                - step (int): quantization step size

            Returns:
                - np.ndarray: reconstructed plane samples (int64)
        '''
        acc = np.asarray(indices, dtype=np.int64) * step
        n = self.order
        for t in range(n - 1, -1, -1):
            seed = sum((-1) ** j * math.comb(t, j) * (self._history[j] if j < len(self._history) else 0)
                       for j in range(t + 1))
            acc = np.cumsum(acc) + seed
        if acc.size and n:
            self._history = [int(v) for v in acc[-1:-n - 1:-1]]
        return acc


#Block walker: assembles/parses the data section, one Rice-or-raw block per plane per period.
def encode_blocks(planes: list[np.ndarray], sources: list[np.ndarray],
                  predictor: Predictor, order: int, step: int, block_size: int) -> bytes:
    '''
        Assembles the data section from coding planes (block-major, plane-minor).

        Per block period, per plane: the last block is zero-padded, LPC analyzes
        the block (params are emitted for every raw/Rice block, keeping the stride
        uniform), then the engine codes the block in a closed loop. A block with all
        values being the same are stored as constant blocks (single int16 value represents the entire block)
        This means a homogenous blockcosts 3 bytes instead of Rice's
        1-bit-per-sample floor. Rice coding at the exact-cost k is kept only when
        strictly smaller than the raw fallback (the plane's source channel as
        big-endian int16, 2 bytes per sample); otherwise the engine rolls back
        and replays the source samples, keeping encoder and decoder histories
        identical.

        Parameters:
            - planes (list[np.ndarray]): int32 coding planes, all the same length
            - sources (list[np.ndarray]): int16 source channels, one per plane
            - predictor (Predictor): predictor mode from the header
            - order (int): predictor order
            - step (int): quantization step size
            - block_size (int): samples per block, usually 1 << block_size_log2

        Returns:
            - bytes: the data section, every block period holding one block per plane

        Raises:
            - ValueError: if no planes, plane/source counts differ, a plane or
              source length differs, block_size < 1, or step < 1
    '''
    if not planes:
        raise ValueError("expected at least one plane")
    if len(planes) != len(sources):
        raise ValueError(f"expected {len(planes)} source channels, got {len(sources)}")
    frames = len(planes[0])
    for plane, source in zip(planes, sources):
        if len(plane) != frames or len(source) != frames:
            raise ValueError(f"expected planes and sources of {frames} samples, got {len(plane)} and {len(source)}")
    if block_size < 1:
        raise ValueError(f"expected block_size >= 1, got {block_size}")
    if step < 1:
        raise ValueError(f"expected step >= 1, got {step}")

    engines = [PlaneEngine(predictor, order) for _ in planes]
    payload = bytearray()

    for start in range(0, frames, block_size):
        n = min(block_size, frames - start)
        for plane, source, engine in zip(planes, sources, engines):
            block = np.zeros(block_size, dtype=np.int32)
            block[:n] = plane[start:start + n]                     #last block zero-padded
            src_block = np.zeros(block_size, dtype=np.int16)
            src_block[:n] = source[start:start + n]

            state = engine.checkpoint()                            #the trial may still go raw
            params = b''
            if predictor == Predictor.LPC:
                coefs = lpc_analyze(block, order)
                params = lpc_params_encode(coefs)
                engine.set_coeffs(coefs)

            indices = engine.code_block(block, step)
            k = best_k(indices)
            rice = rice_encode(indices, k)
            value = int(block[0])
            constant = n and bool(np.all(block[:n] == value)) and -32768 <= value <= 32767
            if constant and 3 <= 1 + len(params) + len(rice):      #constant block: control + int16 value
                engine.rollback(state)                             #history replays the constant
                engine.push_raw(np.full(block_size, value, dtype=np.int32))
                payload += encode_block(MODE_CONSTANT, b'', struct.pack('>h', value))
            elif len(rice) < 2 * block_size:                       #raw costs 2 bytes per sample
                payload += encode_block(k + 1, params, rice)
            else:
                engine.rollback(state)                             #history replays the source instead
                engine.push_raw(src_block)
                payload += encode_block(MODE_RAW, params, src_block.astype('>i2').tobytes())

    return bytes(payload)


def decode_blocks(payload: bytes, planes_count: int, frames: int,
                  predictor: Predictor, order: int, step: int, block_size: int) -> tuple[list[np.ndarray], list[list[bool]]]:
    '''
        Parses the data section back into coding planes (block-major, plane-minor).

        Mirrors encode_blocks: every block period walks one block per plane as
        control | params | data. Rice blocks (mode 1-31, k = mode - 1) decode
        block_size residuals and reconstruct in the same closed loop as the
        encoder; raw blocks (mode 0) carry the plane's source channel as
        big-endian int16, redefining the plane for that period; constant blocks
        (mode 32) fill the whole block with one int16 plane value and stay in
        plane domain. Rice blocks end
        where the decoder stops, so the walk advances by the exact Rice byte
        count. The raw mask per plane records the raw periods: a raw segment is
        in source-channel domain, which the mid-side inverse needs to know.
        Planes are trimmed to frames.

        Parameters:
            - payload (bytes): the data section from read_apac
            - planes_count (int): number of coding planes
            - frames (int): sample frames from the header
            - predictor (Predictor): predictor mode from the header
            - order (int): predictor order
            - step (int): quantization step size
            - block_size (int): samples per block, usually 1 << block_size_log2

        Returns:
            - tuple[list[np.ndarray], list[list[bool]]]: (int32 planes trimmed to
              frames, raw mask per plane per block period)

        Raises:
            - ValueError: if planes_count < 1, frames < 0, block_size < 1, or step < 1
            - BlockError: if the data section is truncated or has trailing bytes
    '''
    if planes_count < 1:
        raise ValueError(f"expected at least one plane, got {planes_count}")
    if frames < 0:
        raise ValueError(f"expected frames >= 0, got {frames}")
    if block_size < 1:
        raise ValueError(f"expected block_size >= 1, got {block_size}")
    if step < 1:
        raise ValueError(f"expected step >= 1, got {step}")

    engines = [PlaneEngine(predictor, order) for _ in range(planes_count)]
    params_size = LPC_PARAM_SIZE * order if predictor == Predictor.LPC else 0
    periods = (frames + block_size - 1) // block_size
    values = [[] for _ in range(planes_count)]
    raw_masks = [[] for _ in range(planes_count)]
    data = memoryview(payload)

    for _ in range(periods):
        for p in range(planes_count):
            mode, params, data = decode_block(data, params_size)
            engine = engines[p]
            if mode == MODE_RAW:
                if len(data) < 2 * block_size:
                    raise BlockError(f"truncated raw block: expected {2 * block_size} bytes, got {len(data)}")
                block = np.frombuffer(data[:2 * block_size], dtype='>i2').astype(np.int32)
                data = data[2 * block_size:]
                engine.push_raw(block)
                values[p].extend(block.tolist())
                raw_masks[p].append(True)
            elif mode == MODE_CONSTANT:
                if len(data) < 2:
                    raise BlockError(f"truncated constant block: expected 2 bytes, got {len(data)}")
                block = np.full(block_size, struct.unpack('>h', data[:2])[0], dtype=np.int32)
                data = data[2:]
                engine.push_raw(block)
                values[p].extend(block.tolist())
                raw_masks[p].append(False)
            else:
                k = mode - 1
                indices = rice_decode(data, block_size, k)
                data = data[(rice_size(indices, k) + 7) // 8:]      #blocks end where the decoder stops
                if predictor == Predictor.LPC:
                    engine.set_coeffs(lpc_params_decode(params, order))
                values[p].extend(engine.decode_block(indices, step).tolist())
                raw_masks[p].append(False)

    if len(data):
        raise BlockError(f"trailing bytes in data section: {len(data)}")

    planes = [np.array(v[:frames], dtype=np.int32) for v in values]
    return planes, raw_masks


#Whole-signal pipeline: header-driven encode/decode plus the file adapters around them. Knob defaults: audio_codec.config.DEFAULT_SETTINGS.


def from_planes(planes: list[np.ndarray], raw_masks: list[list[bool]], channels: int, block_size: int) -> np.ndarray:
    '''
        Merges coding planes back into (frames, channels) int16 samples. Mirrors to_planes.

        Pairs of planes reconstruct their source channels per block period,
        following the raw masks:
            Rice+Rice -> L = (M+S)>>1, R = (M-S)>>1 (M+S and M-S are even)
            raw mid + Rice side -> L = M, R = M-S (the raw mid plane carries L)
            Rice mid + raw side -> L = M-S, R = S (the raw side plane carries R)
            raw mid + raw side -> L = M, R = S (the raw planes carry L and R)
        An unpaired leftover plane is its own channel. Reconstructed values are
        clipped into int16: lossless files are untouched, lossy quantization can
        otherwise push a boundary sample one step past the range.

        Parameters:
            - planes (list[np.ndarray]): int32 coding planes, trimmed to frames
            - raw_masks (list[list[bool]]): per plane per block period, True where the segment is raw
            - channels (int): channel count of the original signal
            - block_size (int): samples per block period

        Returns:
            - np.ndarray: (frames, channels) int16 samples

        Raises:
            - ValueError: if channels < 1, plane or mask counts mismatch the channels,
              plane lengths differ, or a mask does not cover every block period
    '''
    if channels < 1:
        raise ValueError(f"expected at least one channel, got {channels}")
    if len(planes) != channels:
        raise ValueError(f"expected {channels} planes for {channels} channels, got {len(planes)}")
    if len(raw_masks) != channels:
        raise ValueError(f"expected {channels} raw masks, got {len(raw_masks)}")

    frames = len(planes[0])
    periods = (frames + block_size - 1) // block_size
    for plane, mask in zip(planes, raw_masks):
        if len(plane) != frames:
            raise ValueError(f"expected planes of {frames} samples, got {len(plane)}")
        if len(mask) != periods:
            raise ValueError(f"expected {periods} mask entries per plane, got {len(mask)}")

    out = np.empty((frames, channels), dtype=np.int16)
    for ch in range(0, channels - 1, 2):   #pairs: planes sit at their channel indices
        mid, side = planes[ch], planes[ch + 1]
        mid_raw, side_raw = raw_masks[ch], raw_masks[ch + 1]
        left = np.empty(frames, dtype=np.int32)
        right = np.empty(frames, dtype=np.int32)
        for period in range(periods):
            seg = slice(period * block_size, min((period + 1) * block_size, frames))
            m, s = mid[seg], side[seg]
            if mid_raw[period] and side_raw[period]: #raw+raw: the planes carry L and R
                left[seg], right[seg] = m, s
            elif mid_raw[period]: #raw mid carries L; R = L - S
                left[seg], right[seg] = m, m - s
            elif side_raw[period]: #raw side carries R; L = M - R
                left[seg], right[seg] = m - s, s
            else: #Rice+Rice: exact, M+S and M-S are even
                left[seg], right[seg] = (m + s) >> 1, (m - s) >> 1
        
        out[:, ch] = np.clip(left, MIN_INT16, MAX_INT16)
        out[:, ch + 1] = np.clip(right, MIN_INT16, MAX_INT16)
    
    if channels % 2:   #leftover plane is its own channel
        out[:, -1] = np.clip(planes[-1], MIN_INT16, MAX_INT16)
    
    return out


def encode_pipeline(samples: np.ndarray, settings: dict) -> tuple[APACHeader, bytes]:
    '''
        Encodes (frames, channels) int16 samples into an APAC header and data section.

        Settings carry the sample rate (from load_wav metadata) plus codec knobs.
        Fallback to default. Optional DCT pre-shaping (QUANTIZE, the JPEG-style
        quality ladder) shapes the planes on the encoder side only, so
        DCTMode.OFF (default) keeps the whole path lossless
        (disregarding initial quantization when converting to int16).

        Parameters:
            - samples (np.ndarray): (frames, channels) int16 audio
            - settings (dict): sample_rate (required) plus codec settings

        Returns:
            - tuple[APACHeader, bytes]: validated header and the encoded data section

        Raises:
            - ValueError: if settings lacks sample_rate or samples is not 2D int16
            - HeaderError: if the combined settings produce an invalid header
    '''

    if "sample_rate" not in settings:
        raise ValueError("settings must carry 'sample_rate' (from load_wav metadata)")
    
    cfg = {**DEFAULT_SETTINGS, **settings}

    x = np.asarray(samples)
    planes, sources = to_planes(x) #validates 2D int16
    
    if cfg["dct_mode"] != DCTMode.OFF:
        planes = [dct_shaping(p, cfg["dct_window_size"], cfg["dct_param"]) for p in planes]

    block_size = 1 << cfg["block_size_log2"]
    payload = encode_blocks(planes, sources, cfg["predictor"], cfg["predictor_order"], cfg["quant_step"], block_size)

    header = APACHeader(
        sample_rate=cfg["sample_rate"],
        frames=x.shape[0],
        channels=x.shape[1],
        predictor=cfg["predictor"],
        predictor_order=cfg["predictor_order"],
        block_size_log2=cfg["block_size_log2"],
        quant_step=cfg["quant_step"],
        dct_mode=cfg["dct_mode"],
        dct_window_size=cfg["dct_window_size"],
        dct_param=cfg["dct_param"],
        entropy_coder=cfg["entropy_coder"],
    )
    validate_header(header)#NOTE: Shared validation of header to match with decode validation. Important that validation is thorough
    return header, payload


def decode_pipeline(header: APACHeader, payload: bytes) -> np.ndarray:
    '''
        Decodes an APAC header and data section back into (frames, channels) int16 samples.

        Mirrors encode_pipeline: the data section parses block-major per plane
        (planes_count == channels: one plane/channel), then the planes merge back.

        Parameters:
            - header (APACHeader): validated header
            - payload (bytes): the data section after the header

        Returns:
            - np.ndarray: (frames, channels) int16 samples

        Raises:
            - BlockError: if the data section is truncated or malformed
            - ValueError: if planes and raw masks disagree with the header
    '''
    block_size = 1 << header.block_size_log2
    planes, raw_masks = decode_blocks(payload, header.channels, header.frames,
                                      header.predictor, header.predictor_order, header.quant_step, block_size)
    return from_planes(planes, raw_masks, header.channels, block_size)


_HEADER_FIELDS = {   #settings/metadata keys verified by verify_against_header, mapped onto APAC header fields
    "sample_rate": "sample_rate",   #settings dicts: compress() merges the sample rate in
    "samplerate": "sample_rate",    #load_wav metadata dicts
    "channels": "channels",
    "frames": "frames",
    "predictor": "predictor",
    "predictor_order": "predictor_order",
    "block_size_log2": "block_size_log2",
    "quant_step": "quant_step",
    "dct_mode": "dct_mode",
    "dct_window_size": "dct_window_size",
    "dct_param": "dct_param",
    "entropy_coder": "entropy_coder",
}


def verify_against_header(header: APACHeader, settings_or_metadata: dict | None) -> None:
    '''
        Cross-checks a settings or metadata dict against a parsed APAC header.

        Every key that maps onto a header field (_HEADER_FIELDS) must agree with the
        container - wire enums compare by value, so IntEnum settings and raw ints are
        interchangeable. Loader metadata extras (name, subtype, duration, ...) map
        onto nothing and get ignored, so a whole load_wav metadata dict can be passed
        as-is.

        Parameters:
            - header (APACHeader): the parsed container header
            - settings_or_metadata (dict | None): the dict to check (None = no-op)

        Raises:
            - ValueError: if the dict is neither dict nor None, or a supplied key contradicts the header
    '''
    if settings_or_metadata is None:
        return
    if not isinstance(settings_or_metadata, dict):
        raise ValueError(f"settings_or_metadata must be a dict or None, got {type(settings_or_metadata).__name__}")
    for key, value in settings_or_metadata.items():
        field = _HEADER_FIELDS.get(key)
        if field is None:
            continue
        stored = getattr(header, field)
        if value != stored:
            raise ValueError(f"supplied {key}={value!r} does not match the APAC header ({field}={stored!r})")


def compress(data: tuple, settings: dict) -> bytes:
    '''
        Compresses (samples, sample_rate) into a full APAC container (header + data section).

        Compresses incoming wav samples into an APAC container. data bundles what the
        loader hands over: the (frames, channels) int16 samples plus the sample rate
        stored in the APAC header - decompress() returns the same pair, so decoded
        output can round-trip straight back into compress().

        Parameters:
            - data (tuple): (samples, sample_rate) - (frames, channels) int16 samples
              as handed over by the loader, and the sample rate in Hz
            - settings (dict): codec settings

        Returns:
            - bytes: the APAC container (encoded header + data section)

        Raises:
            - ValueError: if the data tuple or settings are invalid
            - HeaderError: if the combined settings produce an invalid header
    '''
    if not isinstance(data, (tuple, list)) or len(data) != 2:
        raise ValueError("data must be a (samples, sample_rate) tuple")
    samples, sample_rate = data
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be a positive integer, got {sample_rate!r}")
    header, payload = encode_pipeline(samples, {**settings, "sample_rate": int(sample_rate)})
    return write_apac(header, payload)


def decompress(data: bytes, settings_or_metadata: dict | None = None) -> tuple[np.ndarray, int]:
    '''
        Decodes a full APAC container back into samples plus the stored sample rate.

        Parses the APAC container and decodes it back into raw samples and the stored
        sample rate. The decode stays header-driven; settings_or_metadata never
        configures it - when supplied (the encoder-side settings dict or the original
        file's load_wav metadata), verify_against_header cross-checks every known key
        against the container first, catching wrong-file mixups. Pass None to skip.

        Parameters:
            - data (bytes): the APAC container (header + data section)
            - settings_or_metadata (dict | None): optional settings or metadata dict
              to cross-check against the header (None = no check)

        Returns:
            - tuple[np.ndarray, int]: ((frames, channels) int16 samples, sample rate in Hz)

        Raises:
            - FormatError: if the container is truncated or the header is invalid
            - BlockError: if the data section is truncated or malformed
            - ValueError: if a supplied settings/metadata key contradicts the header
    '''
    header, payload = read_apac(data)
    verify_against_header(header, settings_or_metadata)
    return decode_pipeline(header, payload), header.sample_rate
