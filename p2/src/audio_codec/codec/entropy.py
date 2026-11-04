#Rice entropy coder, converts residuals from predictor into codes and back.

'''
    Entropy coding (Rice):

    Residual to code:
        zigzag (deal with signed numbers) -> split (unary & remainder) -> emit (convert to rice code)

    Block data:
        | Rice codewords (MSB) | zero padding the byte |

    Notes:
        k is stored in the control byte of the block 5 least significant bits.
        if k is 0 then Rice is not used and the residuals are stored as raw PCM, else k = mode - 1
        keeps control of block boundaries through sample count
'''


import numpy as np

from audio_codec.config import MAX_RICE_K
from audio_codec.codec.format import BlockError


def zigzag(x: int) -> int:
    '''
        Maps a signed integer to non-negative zigzag value.
        Inspired by FLAC.
        Essentially maps smaller number to minimal bit representation.

        Mapping: 0 -> 0, -1 -> 1, +1 -> 2, -2 -> 3 etc.

        Parameters:
            - x (int): signed value

        Returns:
            - int: zigzag value
    '''
    
    return 2 * x * (1 - (res := (x < 0)) * 2) - res




def unzigzag(n: int) -> int:
    '''
        Inverts the zigzag mapping. Takes half of n and flips if odd.


        Parameters:
            - n (int): zigzag value.

        Returns:
            - int: the original signed value.

        Raises:
            - ValueError: if n is negative.
    '''
    if n < 0:
        raise ValueError(f"invalid zigzag value: {n} must be non-negative")
    
    return (n >> 1) ^ -(n & 1)


def zigzag_np(residuals: np.ndarray) -> np.ndarray:
    '''
        Zigzag on array of residuals.
        Same mapping as zigzag().

        Parameters:
            - residuals (np.ndarray): signed values

        Returns:
            - np.ndarray: non-negative zigzag values
    '''
    residuals = np.asarray(residuals, dtype=np.int64)
    return 2 * np.abs(residuals) - (residuals < 0)


def rice_encode(residuals: np.ndarray, k: int) -> bytes:
    '''
        Encode signed residuals with Rice coding into byte stream.

        Residual code: q zero bits, one 1 bit, then k remainder.
        Stops when all residuals have been encoded (samples/residuals per block).

        Parameters:
            - residuals (np.ndarray): signed residuals
            - k (int): Rice parameter

        Returns:
            - bytes: packed bit stream, zero-padded

        Raises:
            - ValueError: if k is outside [0-MAX_RICE_K]
    '''
    if not 0 <= k <= MAX_RICE_K:
        raise ValueError(f"k must be in the range [0-{MAX_RICE_K}], got {k}")

    #vectorized: lay every codeword out as bits (q zeros | 1 terminator | k remainder bits, MSB first), then pack
    z = zigzag_np(residuals)
    quotient = z >> k
    remainder = z & ((1 << k) - 1)
    lengths = quotient + (1 + k)                       #bits per codeword
    total_bits = int(lengths.sum())
    if total_bits == 0:
        return b''                                     #empty input: nothing on the wire

    bits = np.zeros(total_bits, dtype=np.uint8)
    starts = np.zeros(lengths.size, dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])            #first codeword starts at bit 0
    bits[starts + quotient] = 1                        #terminator bit after the unary zeros
    if k:
        slots = starts[:, None] + quotient[:, None] + 1 + np.arange(k)[None, :]   #remainder slots, MSB first
        bits[slots] = (remainder[:, None] >> np.arange(k - 1, -1, -1)[None, :]) & 1

    return np.packbits(bits).tobytes()                 #MSB-first packing, last byte zero-padded


def rice_decode(data: bytes, count: int, k: int) -> np.ndarray:
    '''
        Decodes [count] residuals from [data].

        Vectorized: terminator indices chain as J[i] = C[ones[J[i-1]] + k + 1]
        (C[p] = number of 1 bits before bit p), walked with binary lifting
        (O(log count) passes, no per-residual Python loop).

        Parameters:
            - data (bytes): Rice encoded bytes stream
            - count (int): number of residuals to decode (sample count)
            - k (int): Rice parameter

        Returns:
            - np.ndarray: decoded residuals as int64

        Raises:
            - ValueError: if k is outside [0-MAX_RICE_K] or count is negative
            - BlockError: if data ends before expected number of residuals are decoded
    '''
    if not 0 <= k <= MAX_RICE_K:
        raise ValueError(f"k must be in the range [0-{MAX_RICE_K}], got {k}")
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")

    if count == 0:
        return np.empty(0, dtype=np.int64)
    if len(data) == 0:
        raise BlockError(f"missing data: expected more residuals (residual 0 of {count})")

    #2*count + 16 bytes covers every block the encoder keeps; growth handles the rest
    limit = min(len(data), 2 * count + 16)
    while True:
        bits = np.unpackbits(np.frombuffer(data[:limit], dtype=np.uint8))    #one bit per element, MSB first per byte
        total_bits = bits.size
        ones = np.flatnonzero(bits)                                          #ascending positions of the 1 bits
        m = ones.size
        exhausted = limit >= len(data)

        if m == 0:                                                           #no terminator anywhere in the table
            if not exhausted:
                limit = min(len(data), limit * 4)
                continue
            raise BlockError(f"missing data: expected more residuals (residual 0 of {count})")

        if k == 0:
            #no remainder bits: every 1 bit terminates one codeword
            if m < count:
                if not exhausted:
                    limit = min(len(data), limit * 4)
                    continue
                raise BlockError(f"missing data: expected more residuals (residual {m} of {count})")
            terms = ones[:count]
        else:
            #J[i] = ones-index of codeword i's terminator (chain in the docstring);
            #sentinel m (jump[m] = m) absorbs missing terminators and straddling remainders
            C = np.concatenate(([0], np.cumsum(bits, dtype=np.int64)))
            jump = np.append(C[np.minimum(ones + (k + 1), total_bits)], m)

            #binary lifting: square jump to double the advance, fill J in doubling phases
            J = np.empty(count, dtype=np.int64)
            J[0] = 0
            filled = 1
            while filled < count:
                take = min(filled, count - filled)
                J[filled:filled + take] = jump[J[:take]]
                filled += take
                if filled < count:
                    jump = jump[jump]

            #sentinel = missing terminator; fewer than k+1 bits after a terminator = straddle
            sent = J == m
            terms = ones[np.minimum(J, m - 1)]               #safe gather; rows behind sentinels are unused
            straddle = (~sent) & (terms + (k + 1) > total_bits)
            if sent.any() or straddle.any():
                if not exhausted:
                    limit = min(len(data), limit * 4)
                    continue
                if straddle.any():                           #straddle at i precedes the sentinel at i + 1
                    raise BlockError(f"missing data: expected more residuals while reading remainder (residual {int(np.flatnonzero(straddle)[0])} of {count})")
                raise BlockError(f"missing data: expected more residuals (residual {int(np.flatnonzero(sent)[0])} of {count})")

        #quotients from the terminator gaps, remainders straight out of [bits]
        starts = np.empty(count, dtype=np.int64)
        starts[0] = 0
        starts[1:] = terms[:-1] + k + 1
        z = (terms - starts) << k
        for j in range(k):                                  #k remainder bits, MSB first
            z |= bits[terms + 1 + j].astype(np.int64) << (k - 1 - j)
        return (z >> 1) ^ -(z & 1)                          #unzigzag


def rice_size(residuals: np.ndarray, k: int) -> int:
    '''
        Calculates the bit cost of Rice-encoding residuals at the given k.
        Each residual costs q + 1 + k bits.

        Parameters:
            - residuals (np.ndarray): signed residuals
            - k (int): Rice parameter, [0-MAX_RICE_K]

        Returns:
            - int: total bits

        Raises:
            - ValueError: if k is outside [0-MAX_RICE_K]
    '''
    if not 0 <= k <= MAX_RICE_K:
        raise ValueError(f"k must be in the range [0-{MAX_RICE_K}], got {k}")

    zigzagged = zigzag_np(residuals)
    return int(np.sum((zigzagged >> k) + 1 + k))


def best_k(residuals: np.ndarray) -> int:
    '''
        Calculates the best Rice parameter (k) for the given residuals.
        If perf(k_a)==perf(k_b), k_a < k_b -> k_a is chosen.

        Parameters:
            - residuals (np.ndarray): signed residuals

        Returns:
            - int: best Rice parameter, [0-MAX_RICE_K]
    '''

    zigzagged = zigzag_np(residuals)

    #one shifted matrix instead of MAX_RICE_K + 1 separate sums; argmin ties keep the lowest k
    ks = np.arange(MAX_RICE_K + 1, dtype=np.int64)
    costs = (zigzagged[None, :] >> ks[:, None]).sum(axis=1, dtype=np.int64) + zigzagged.size * (1 + ks)

    return int(np.argmin(costs))
