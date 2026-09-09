import librosa
import numpy as np
from scipy.io.wavfile import write as wavwrite


def load(path, *, to_mono_float32=True, **kwargs):
    """
    Load an audio file.
    
    Args:
        path: Path to audio file.
        to_float32: Convert to float32 format. Defaults to True.
        **kwargs: Additional arguments passed to librosa.load.
    
    Returns:
        tuple: (audio_data, sample_rate)
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


def write_wav(filename, rate: int, data: np.ndarray):
    """
    Wrapper around scipy.io.wavfile.write.

    Parameters:
        filename: Output wav file.
        rate: The sample rate (in samples/sec).
        data: A 1-D or 2-D NumPy array of either integer or float data-type.
    """
    wavwrite(filename, rate, data)
