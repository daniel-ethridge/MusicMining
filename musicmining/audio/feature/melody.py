"""Extract melody features from essentia."""

import essentia.standard as es
import numpy as np
import pandas as pd

from musicmining.audio import io


def melody_from_file(file_path, sr, *, hop_size=1024, block_size=4096):
    """
    Extract melody features from a file.

    Parameters
    ----------
    file_path : str
        Path to an audio file.
    sr : int
        Target sample rate of audio
    hop_size : int, optional
        STFT hop length for librosa and essentia methods.
    block_size : int, optional
        FFT window size for librosa and essentia methods.

    Returns
    -------
    pandas.DataFrame
        Melody features out.
    """
    data, rate = io.load(file_path, sr)
    return melody_from_audio(data, rate, hop_size=hop_size, block_size=block_size)


def melody_from_audio(data, sr, *, hop_size=1024, block_size=4096):
    """
    Extract melody features from audio.

    Parameters
    ----------
    data : np.ndarray
        Audio data
    sr : int
        Sample rate of the audio
    hop_size : int, optional
        STFT hop length for librosa and essentia methods. Defaults to 1024.
    block_size : int, optional
        FFT window size for librosa and essentia methods. Defaults to 4096.

    Returns
    -------
    pandas.DataFrame
        Melody features out.
    """
    # Create feature dictionary
    feature_data = {}

    loudness = es.EqualLoudness(sampleRate=sr)
    data = loudness(data)

    # pitch extraction
    pitch_extractor = es.PredominantPitchMelodia(frameSize=block_size, hopSize=hop_size)
    pitch_values, pitch_confidence = pitch_extractor(data)
    pitch_filter = es.PitchFilter()
    melody = pitch_filter(pitch_values, pitch_confidence)
    melody_clean = [val for val in melody if val > 0]
    if len(melody_clean) > 0:
        melody_cents = [1200 * np.log2(val / 261.63) for val in melody_clean]
        feature_data["melody_avg_f0_cents_midCref"] = [np.mean(melody_cents)]
        feature_data["melody_std_f0_cents_midCref"] = [np.std(melody_cents)]
    else:
        feature_data["melody_avg_f0_cents_midCref"] = [np.nan]
        feature_data["melody_std_f0_cents_midCref"] = [np.nan]

    return pd.DataFrame(feature_data)
