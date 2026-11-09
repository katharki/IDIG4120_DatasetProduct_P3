from __future__ import annotations
import os
import numpy as np
from pathlib import Path
import soundfile as sf
from audio_codec.config import MAX_INT16, MIN_INT16
from audio_codec.shared import logger


def load_folder(folder: Path) -> list[tuple[dict, np.ndarray]]:
    '''
    Load all files in a folder and return their contents.

    Parameters:
        - folder (Path): The path to the folder containing the files to be loaded.

    Returns:
        - list[tuple[dict, np.ndarray]]: List of files.

    Raises:
        - FileNotFoundError: If the folder does not exist or is not a folder.
    
    Note:
        - Might OOM if the folder contains a large number of or very large audio files.
    '''
    if not folder.exists():
        raise FileNotFoundError(f"No such folder: '{folder}'")
    
    if not folder.is_dir():
        raise FileNotFoundError(f"No such folder: '{folder}'")

    files = []
    for file in folder.iterdir():
        if file.is_file():
            try:
                files.append(load_file(file))
            except Exception as e:
                print(f"Failed to load file: '{file}' with error: {e}")
    return files

def load_file(infile: Path) -> tuple[dict, np.ndarray]:
    '''
    Load a file and return its contents.

    Parameters:
        - infile (Path): The path to the file to be loaded.

    Returns:
        - tuple[dict, np.ndarray]: A tuple containing the metadata and audio data of the file.

    Raises:
        - FileNotFoundError: If the file does not exist or is not a file.
        - ValueError: If the file has no suffix.
    '''
    if not infile.exists():
        raise FileNotFoundError(f"No such file: '{infile}'")
    
    if not infile.is_file():
        raise FileNotFoundError(f"No such file: '{infile}'")

    if not infile.suffix:
        raise ValueError(f"File has no suffix: '{infile}'")

    # match case is for eventual support of more file types in the future (sort of already supported through soundfile, but metadata might differ (not in scope of assignment))
    match infile.suffix:
        case ".wav":
            try:
                return load_wav(infile)
            except Exception as e:
                raise e
        case _:
            raise ValueError(f"Unsupported file type: '{infile.suffix}'")



def load_wav(infile: Path) -> tuple[dict, np.ndarray]:
    '''
    Load a WAV file and return its metadata and audio data converted to [np.int16].

    Parameters:
        - infile (Path): The path to the WAV file to be loaded.

    Returns:
        - tuple[dict, np.ndarray]: A tuple containing the metadata and audio data of the WAV file.

    Raises:
        - ValueError: If the WAV file cannot be loaded.

    Notes:
        - Audio data is always a 2d array
        - Subtypes wider than 16 bits (PCM_24/PCM_32) and float subtypes (FLOAT/DOUBLE) are read at native
          precision and quantized to int16 with round + clip (max error 1/2 LSB per sample). PCM_U8/PCM_16,
          u-law, a-law and ADPCM files use libsndfile's int16 read directly, which is measured correct there.
        - Metadata has the format:
            {
                "name": name,
                "samplerate": samplerate,
                "channels": channels,
                "frames": frames,
                "duration": frames/samplerate,
                "format": format,
                "subtype": subtype,
                "endian": endian,
                "format_info": format_info,
                "subtype_info": subtype_info,
                "sections": sections,
                "extra_info": extra_info
            }

    '''
    try:
        info = sf.info(infile)
        metadata = {
            "name": info.name,
            "samplerate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration": info.duration,
            "format": info.format,
            "subtype": info.subtype,
            "endian": info.endian,
            "format_info": info.format_info,
            "subtype_info": info.subtype_info,
            "sections": info.sections,
            "extra_info": info.extra_info,
        }
        if info.subtype in ('PCM_U8', 'PCM_16', 'ULAW', 'ALAW', 'IMA_ADPCM', 'MS_ADPCM'):
            np_data, _ = sf.read(infile, dtype=np.int16, always_2d=True)
        else:
            xf, _ = sf.read(infile, dtype='float64', always_2d=True)
            np_data = np.clip(np.round(xf * float(MAX_INT16)), MIN_INT16, MAX_INT16).astype(np.int16)
        return metadata, np_data

    except Exception as e:
        raise ValueError(f"Failed to load WAV file: '{infile}' with error: {e}") from e


def save_wav(outfile: Path, data: np.ndarray, samplerate: int) -> None:
    '''
    Save audio data as a 16-bit PCM WAV file.

    Parameters:
        - outfile (Path): The path of the WAV file to write.
        - data (np.ndarray): (frames, channels) int16 audio.
        - samplerate (int): Sample rate in Hz.

    Raises:
        - IOError: If writing the file fails.
    '''
    try:
        sf.write(Path(outfile), data, samplerate, subtype="PCM_16")
    except Exception as e:
        raise IOError(f"Failed to write WAV file '{outfile}': {e}") from e


def load_apac(infile: Path) -> bytes:
    '''
    Load an APAC file as raw container bytes (header + data section).

    Parameters:
        - infile (Path): The path of the APAC file to load.

    Returns:
        - bytes: The raw APAC container bytes.

    Raises:
        - IOError: If reading the file fails.
    '''
    try:
        return Path(infile).read_bytes()
    except Exception as e:
        raise IOError(f"Failed to read APAC file '{infile}': {e}") from e


def save_apac(outfile: Path, data: bytes) -> None:
    '''
    Save APAC container bytes to a file.

    Parameters:
        - outfile (Path): The path of the APAC file to write.
        - data (bytes): The APAC container bytes.

    Raises:
        - IOError: If writing the file fails.
    '''
    try:
        Path(outfile).write_bytes(data)
    except Exception as e:
        raise IOError(f"Failed to write APAC file '{outfile}': {e}") from e


