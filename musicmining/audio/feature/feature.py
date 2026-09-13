"""Extract audio features from a file."""

from enum import Flag
from typing import Optional

import numpy as np
import pandas as pd

from musicmining.audio import feature as feat
from musicmining.util.exceptions import ParameterError


class Feat(Flag):
    """Feature bitmask."""
    NONE = 0
    DYNAMICS = 1 << 0
    EXPRESSIVITY = 1 << 1
    HARMONY = 1 << 2
    MELODY = 1 << 3
    RHYTHM = 1 << 4
    TIMBRE = 1 << 5
    VOCAL = 1 << 6


def extract_feats(
        audio: np.ndarray,
        sr: int,
        *,
        file_path: Optional[str] = None,
        hop_size: int=1024,
        block_size: int=4096,
        feat_to_exclude: Feat = Feat.NONE):
    """
    Extract features from audio.

    This is a convenience function to extract dynamics, expressivity, harmony, melody, rhythm, timbre, and vocal
    features with a single function call.

    Parameters
    ----------
    audio : numpy.ndarray
        Audio data
    sr : int
        The sample rate of the audio
    file_path : str, optional
        The path to the audio file. Because of how the opensmile feature extraction works, vocal feature extraction
        requires the path to the audio file. If `feat_to_exclude` does not include `Feat.VOCAL`, `file_path` must be
        passed.
    hop_size : int, optional
        STFT hop length for librosa and essentia methods. Default to 1024.
    block_size : int, optional
        FFT window size for librosa and essentia methods. Default to 4096.
    feat_to_exclude : Feat
        A bitmask of `Feat`. If `file_path` is omitted or set to None, `feat_to_exclude` must include `Feat.VOCAL`.

    Returns
    -------
    pandas.DataFrame
        The extracted features.
    """
    # Check for vocals path if necessary
    if feat_to_exclude & Feat.VOCAL == Feat.NONE and file_path is None:
        raise ParameterError("If vocals are included, a file path for the audio must be included as well.")

    # Add features
    concat_list = []
    if feat_to_exclude & Feat.DYNAMICS == Feat.NONE:
        concat_list.append(feat.dynamics_from_audio(audio, sr, hop_size=hop_size, block_size=block_size))
    if feat_to_exclude & Feat.EXPRESSIVITY == Feat.NONE:
        concat_list.append(feat.expressivity_from_audio(audio, sr, hop_size=hop_size, block_size=block_size))
    if feat_to_exclude & Feat.HARMONY == Feat.NONE:
        concat_list.append(feat.harmony_from_audio(audio, sr, block_size=block_size))
    if feat_to_exclude & Feat.MELODY == Feat.NONE:
        concat_list.append(feat.melody_from_audio(audio, sr, hop_size=hop_size, block_size=block_size))
    if feat_to_exclude & Feat.RHYTHM == Feat.NONE:
        concat_list.append(feat.rhythm_from_audio(audio, sr))
    if feat_to_exclude & Feat.TIMBRE == Feat.NONE:
        concat_list.append(feat.timbre_from_audio(audio, sr, hop_size=hop_size, block_size=block_size))
    if feat_to_exclude & Feat.VOCAL == Feat.NONE:
        concat_list.append(feat.vocal_from_file(file_path, sr, hop_size=hop_size, block_size=block_size))

    return pd.concat(concat_list, axis=1)


