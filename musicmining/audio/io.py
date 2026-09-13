"""I/O for audio files"""

from typing import Optional

import librosa
import numpy as np
import scipy.io.wavfile as wav_file


def load(path: str, sr: Optional[int] = 22050, **kwargs):
    """
    Load an audio file. This is a wrapper around librosa.load.

    Parameters
    ----------
    path : str
        Path to audio file.
    sr : int, optional
        Target sample rate. If None, uses the native sampling rate. Defaults to 22050
    **kwargs
        Additional arguments passed to librosa.load.

    Returns
    -------
    tuple
        (audio data, sample rate)
    """
    # Initial load
    return librosa.load(path, sr=sr, **kwargs)


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
    wav_file.write(filename, sr, data)
