"""Audio effects"""


import numpy as np
from librosa.effects import pitch_shift as lib_pitch_shift


def pitch_shift(y: np.ndarray, *, sr: float, n_steps: float, **kwargs) -> np.ndarray:
    """
    Shift the pitch of a file by `n_steps` steps.

    This is a wrapper around librosa.effects.pitchshift.

    Parameters
    ----------
    y : numpy.ndarray
        The input audio file
    sr : float
        The sample rate of the audio file
    n_steps : float
        How many (fractional) steps to shift `y`.
    **kwargs : Additional keyword arguments
        See librosa.effects.pitch_shift for details

    Returns
    -------
    numpy.ndarray
        The pitch-shifted audio time-series. See librosa.effects.pitch_shift for details.
    """
    return lib_pitch_shift(y, sr=sr, n_steps=n_steps, **kwargs)


def pitch_shift_group(y: np.ndarray, *, sr: float, n_steps: np.ndarray, return_original: bool=False, **kwargs) -> dict:
    """
    Shift the pitch of a waveform multiple times.

    Parameters
    ----------
    y : np.ndarray
        The input audio file
    sr : float
        The sampling rate of of `y`
    n_steps : np.ndarray
        A one-dimensional numpy array containing all the desired pitch shift amounts
    return_original : bool
        If True, the original audio is included in the returned dictionary and can be accessed using at the key '0'. If
        False, the original audio is omitted from the returned dictionary. Defaults to False.
    **kwargs : additional keyword arguments
        See librosa.effects.pitch_shift for details

    Returns
    -------
    dict
        A dictionary containing the shifted audio files. The keys are the elements of `n_steps`, and the values are
        the shifted audio files.
    """
    shift_dict = {}
    for step in n_steps:
        shift_dict[step] = pitch_shift(y, sr=sr, n_steps=step, **kwargs)

    if return_original:
        shift_dict[0] = y

    return shift_dict
