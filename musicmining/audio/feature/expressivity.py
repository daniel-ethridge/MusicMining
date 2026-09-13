"""
Extract expressivity features from essentia.

Currently, this module extracts aspects of vibrato from the audio. Average and spread measurements of both vibrato
speed and depth are calculated.
"""

import essentia.standard as es
import numpy as np
import pandas as pd

from musicmining.audio import io


def expressivity_from_file(file_path, sr, *, hop_size=1024, block_size=4096):
    """
    Extract expressivity features from a file.

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
        Expressivity features out.
    """
    data, sr = io.load(file_path, sr=sr)
    return expressivity_from_audio(data, sr, hop_size=hop_size, block_size=block_size)


def expressivity_from_audio(data, sr, *, hop_size=1024, block_size=4096):
    """
    Extract expressivity features from audio.

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
        Expressivity features out.
    """
    # Create feature dictionary
    feature_data = {}

    # Read file
    loudness = es.EqualLoudness(sampleRate=sr)
    data = loudness(data)

    # pitch extraction
    pitch_extractor = es.PredominantPitchMelodia(frameSize=block_size, hopSize=hop_size)
    pitch_values, pitch_confidence = pitch_extractor(data)
    pitch_filter = es.PitchFilter()
    melody = pitch_filter(pitch_values, pitch_confidence)

    # Vibrato
    num_frames = len(melody)
    duration_of_audio = len(data) / sr
    p_fs = num_frames/duration_of_audio
    vibrato = es.Vibrato(sampleRate=p_fs)
    vib_frq, vib_extend = vibrato(melody)
    vib_frq = np.array(vib_frq)
    vib_extend = np.array(vib_extend)

    feature_data["expressivity_avg_vibrato_speed"] = [np.mean(vib_frq)]
    feature_data["expressivity_std_vibrato_speed"] = [np.std(vib_frq)]
    feature_data["expressivity_avg_vibrato_depth"] = [np.mean(vib_extend)]
    feature_data["expressivity_std_vibrato_depth"] = [np.std(vib_extend)]

    return pd.DataFrame(feature_data)
