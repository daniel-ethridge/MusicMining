import numpy as np
import pytest

from musicmining.audio import io

rates = [11025, 22050, 44100, 48000]
depths = [8, 16, 32, 64]


@pytest.fixture(params=rates)
def sample_rate(request):
    return request.param


@pytest.fixture(params=depths)
def bit_depth(request):
    return request.param


def generate_sine_wav(rate, depth):
    length = 5
    t = np.linspace(0, length, length * rate, endpoint=False)
    y = np.sin(2 * np.pi * t) + 0.7 * np.sin(3 * np.pi * t)
    y /= np.max(np.abs(y))

    assert len(y) == length * rate

    if depth == 8:
        y += 1
        y *= 1 << 7
        y = np.uint8(y)
        assert 255 == pytest.approx(np.max(y))
        assert 0 == pytest.approx(np.min(y))
        assert y.dtype == np.uint8

    elif depth == 16:
        y *= 1 << 15
        y = np.int16(y)
        assert 32767 == pytest.approx(np.max(y))
        assert -32768 == pytest.approx(np.min(y))
        assert y.dtype == np.int16

    elif depth == 32:
        y = np.float32(y)
    elif depth == 64:
        y = np.float64(y)

    return y


def test_write(tmp_path, sample_rate, bit_depth):
    y = generate_sine_wav(sample_rate, bit_depth)
    io.write_wav(f"{tmp_path}/temp_audio.wav", sample_rate, y)


def test_load(tmp_path, sample_rate, bit_depth):
    audio_path = f"{tmp_path}/temp_audio.wav"

    y = generate_sine_wav(sample_rate, bit_depth)
    io.write_wav(audio_path, sample_rate, y)

    _, sr = io.load(audio_path)
    assert sr == 22050

    _, sr = io.load(audio_path, sr=None)
    assert sr == sample_rate

    _, sr = io.load(audio_path, sr=44100)
    assert sr == 44100
