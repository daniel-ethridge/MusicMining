import numpy as np
import pytest

from musicmining.audio import io
from .utility_functions import generate_tone

rates = [11025, 22050, 44100, 48000]
depths = [8, 16, 32, 64]


@pytest.fixture(params=rates)
def sample_rate(request):
    return request.param


@pytest.fixture(params=depths)
def bit_depth(request):
    return request.param


def test_write(tmp_path, sample_rate, bit_depth):
    y = generate_tone(sample_rate, bit_depth)
    io.write_wav(f"{tmp_path}/temp_audio.wav", sample_rate, y)


def test_load(tmp_path, sample_rate, bit_depth):
    audio_path = f"{tmp_path}/temp_audio.wav"

    y = generate_tone(sample_rate, bit_depth)
    io.write_wav(audio_path, sample_rate, y)

    _, sr = io.load(audio_path)
    assert sr == 22050

    _, sr = io.load(audio_path, sr=None)
    assert sr == sample_rate

    _, sr = io.load(audio_path, sr=44100)
    assert sr == 44100
