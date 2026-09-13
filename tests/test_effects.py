import pytest
from .utility_functions import generate_tone
from musicmining.audio.effects import pitch_shift, pitch_shift_group
import numpy as np


def test_pitch_shift():
    """The pitch_shift is a pure wrapper around librosa.effects.pitch_shift. We trust librosa"""
    pass


def test_pitch_shift_group():
    y = generate_tone(22050, 32)

    steps = np.array([4, 3, 8, -2, 7, -4, 8])
    group = pitch_shift_group(y, sr=22050, n_steps=steps)

    assert len(group.keys()) == np.unique(steps).shape[0]
    for k in steps:
        assert k in group.keys()

    group = pitch_shift_group(y, sr=22050, n_steps=steps, return_original=True)
    assert len(group.keys()) == np.unique(steps).shape[0] + 1
    for k in steps:
        assert k in group.keys()

    assert 0 in group.keys()
