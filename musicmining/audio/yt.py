"""YouTube downloading functionality using the package yt_dlp"""

from typing import Optional

import yt_dlp


def download_audio_mp3(url: str, output_directory: str, filename: Optional[str] = None):
    """
    Download mp3 file from YouTube URL.

    Parameters
    ----------
    url : str
        The URL of a YouTube video
    output_directory : str
        The directory in which to save the file
    filename : str, optional
        If provided, the name of the file (including extension) containing the downloaded audio. If None, the file name
        will be <YOUTUBE_ID>_<VIDEO_TITLE>.mp3. Defaults to None.
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_directory}/%(id)s_%(title)s.%(ext)s",
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True, None

    except yt_dlp.utils.DownloadError as e:
        return False, e


if __name__ == "__main__":
    # Download Gangnam style
    print(download_audio_mp3("https://www.youtube.com/watch?v=9bZkp7q19f0", "./"))
