import pytest
import numpy as np


def generate_tone(rate, depth):
    """Generate a 5 second tone at a specific bit depth and sample rate"""
    length = 5
    t = np.linspace(0, length, length * rate, endpoint=False)
    y = np.sin(440 * 2 * np.pi * t)
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
