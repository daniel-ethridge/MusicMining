
import pytest
from musicmining.audio import io
import numpy as np


@pytest.mark.parametrize('shift, type_func', (
    [8, np.uint8], 
    [16, np.int16],
    [32, np.int32],
    [32, np.float32]
))
@pytest.mark.parametrize('sr', ([22050, 44100, 11025]))
def test_load(shift, type_func, sr):
    data = np.zeros((sr, 2))

    if type_func != np.float32:
        data[:, 0] = np.random.random(sr) * (1 << shift)
        data[:, 1] = np.random.random(sr) * (1 << shift)
    else:
        data[:, 0] = np.random.random(sr)
        data[:, 1] = np.random.random(sr)

    for n in range(data.shape[1]):
        for i, val in enumerate(data[:, n]):
            if val > (1 << (shift -1)):
                data[i][n] /= 2
        
    data = type_func(data)
    io.write_wav("./tests/test.wav", sr, data)

    audio, fs = io.load("./tests/test.wav", to_mono_float32=True, sr=sr)

    assert fs == sr
    assert audio.dtype == np.float32
    assert audio.shape == (sr,)
    assert np.min(audio) >= -1
    assert np.max(audio) <= 1
