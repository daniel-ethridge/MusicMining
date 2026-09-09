from librosa.effects import pitch_shift as lib_pitch_shift
from typing import List
import numpy as np



def pitch_shift(y: np.ndarray, *, sr: float, n_steps: float, **kwargs) -> np.ndarray:
    return lib_pitch_shift(y, sr=sr, n_steps=n_steps, **kwargs)


def pitch_shift_group(y: np.ndarray, *, sr: float, n_steps: np.array, return_original=False, **kwargs) -> dict:
    shift_dict = {}
    for step in n_steps:
        shift_dict[step] = pitch_shift(y, sr=sr, n_steps=step, **kwargs)

    if return_original:
        shift_dict[0] = y
        
    return shift_dict