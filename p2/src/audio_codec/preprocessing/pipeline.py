#Contains data normalization and validation
from __future__ import annotations

import numpy as np

from audio_codec.shared.validation import ValidationIssue


def wav_issues(metadata: dict, data: np.ndarray) -> list[ValidationIssue]:
    '''
    Detect every problem with a loaded WAV file. Pure: never raises, never logs.

    Parameters:
        - metadata (dict): The metadata of the WAV file (as returned by load_wav()).
        - data (np.ndarray): The audio data of the WAV file (int16, shape (frames, channels)).

    Returns:
        - list[ValidationIssue]: One issue per problem found; empty if the file is valid.
    '''
    #Metadata presence and ranges, one line per field.
    checks: list[tuple[bool, str, str]] = [
        (bool(metadata), "metadata", "metadata is missing"),
        (metadata.get("samplerate", 0) > 0, "samplerate", f"invalid samplerate: {metadata.get('samplerate')}"),
        (metadata.get("channels", 0) > 0, "channels", f"invalid channels: {metadata.get('channels')}"),
        (metadata.get("frames", 0) > 0, "frames", f"invalid frames: {metadata.get('frames')}"),
        (data.ndim == 2, "data", f"expected 2D data (frames, channels), got {data.ndim}D"),
    ]
    issues = [ValidationIssue(field, message) for ok, field, message in checks if not ok]

    #Data/metadata consistency (only meaningful once metadata declares a shape and data is 2D).
    if metadata.get("frames") and metadata.get("channels") and data.ndim == 2:
        expected = (metadata["frames"], metadata["channels"])
        if data.shape != expected:
            issues.append(ValidationIssue("data", f"data shape {tuple(data.shape)} does not match metadata shape {expected}"))

    return issues


def validate_wav(metadata: dict, data: np.ndarray) -> bool:
    '''
    Check whether a loaded WAV file is valid for further processing.

    Parameters:
        - metadata (dict): The metadata of the WAV file.
        - data (np.ndarray): The audio data of the WAV file.

    Returns:
        - bool: True if and only if wav_issues() finds no problems. Log/report at the call site.
    '''
    return not wav_issues(metadata, data)
