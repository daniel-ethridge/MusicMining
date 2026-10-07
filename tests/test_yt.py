import os
import pytest
from musicmining.audio import download_audio_mp3


def test_download(tmp_path):
    # Gangnam Style
    url = "https://www.youtube.com/watch?v=9bZkp7q19f0"
    success, _ = download_audio_mp3(url, tmp_path)
    assert(success)


def test_download_age_restricted(tmp_path):
    # a random age restricted video
    url = "https://www.youtube.com/watch?v=GG0LY8OLBG8"
    success, err = download_audio_mp3(url, tmp_path)
    assert(not success)
    assert(err is not None)


def test_download_bad_url(tmp_path):
    # invalid url
    url = "https://www.youtube.com/watch?v=bad"
    success, err = download_audio_mp3(url, tmp_path)
    assert(not success)
    assert(err is not None)