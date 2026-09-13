"""Extract rhythm features from essentia and librosa."""

import essentia.standard as es
import numpy as np
import pandas as pd
from librosa.feature import tempo
from librosa.onset import onset_strength

from musicmining.audio import io


def rhythm_from_file(file_path, sr):
    """
    Extract rhythm features from a file.

    Parameters
    ----------
    file_path : str
        Path to an audio file.
    sr : int
        Target sample rate of audio

    Returns
    -------
    pandas.DataFrame
        Rhythm features out.
    """
    data, rate = io.load(file_path, sr=sr)
    return rhythm_from_audio(data, rate)


def rhythm_from_audio(data, sr):
    """
    Extract rhythm features from audio.

    Parameters
    ----------
    data : np.ndarray
        Audio data
    sr : int
        Sample rate of the audio

    Returns
    -------
    pandas.DataFrame
        Rhythm features out.
    """
    feature_data = {}

    # First prediction
    rhythm_ex = es.RhythmExtractor2013(maxTempo=240)
    bpm, _, _, estimates, _ = rhythm_ex(data)
    std_estimates = np.std(estimates)

    # Refined prediction
    onset_env = onset_strength(y=data, sr=sr)
    pred_tempo = tempo(onset_envelope=onset_env, start_bpm=bpm, std_bpm=std_estimates)
    feature_data["rhythm_tempo_librosa"] = pred_tempo
    feature_data["rhythm_tempo_essentia"] = bpm
    feature_data["rhythm_std_tempo_estimates"] = std_estimates

    return pd.DataFrame(feature_data)
