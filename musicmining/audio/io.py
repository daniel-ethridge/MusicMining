"""I/O for audio files"""

import librosa
import numpy as np
from scipy.io.wavfile import write as wavwrite


def load(path: str, *, to_mono_float32: bool=True, **kwargs):
    """
    Load an audio file.

    Parameters
    ----------
    path : str
        Path to audio file.
    to_mono_float32 : bool, optional
        Convert to float32 format and to mono. Defaults to True.
    **kwargs
        Additional arguments passed to librosa.load.

    Returns
    -------
    tuple
        (audio data, sample rate)
    """
    # Initial load
    data, sr = librosa.load(path, **kwargs)

    if not to_mono_float32:
        return data, sr

    # ensure mono
    data = librosa.to_mono(data)

    # convert
    if (data.dtype == np.uint8):
        data -= (1 << 7)
        data /= (1 << 7)
    elif (data.dtype == np.int16):
        data /= (1 << 15)
    elif (data.dtype == np.int32):
        data /= (1 << 31)

    # return
    data = np.float32(data)
    return data, sr


def write_wav(filename: str, sr: int, data: np.ndarray):
    """
    Call scipy.io.wavfile.write.

    Parameters
    ----------
    filename : str
        Output wav file.
    sr : int
        The sample rate (in samples/sec).
    data : numpy ndarray
        A 1-D or 2-D NumPy array of either integer or float data-type.
    """
    wavwrite(filename, sr, data)
