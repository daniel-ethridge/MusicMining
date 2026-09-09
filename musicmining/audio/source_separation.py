import os
from pathlib import Path

import torch
from torchaudio.transforms import Fade
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import numpy as np
from enum import Flag
import librosa
from typing import List, Optional


class Stem(Flag):
    VOCALS = 1 << 0  # 0001 (1)
    DRUMS  = 1 << 1   # 0010 (2)
    BASS   = 1 << 2    # 0100 (4)
    OTHER  = 1 << 3   # 1000 (8)

    
    def english_map(combo: int):
        word_out = []

        if combo & Stem.VOCALS:
            word_out.append('V')
        if combo & Stem.DRUMS:
            word_out.append('D')
        if combo & Stem.BASS: 
            word_out.append('B')
        if combo & Stem.OTHER:
            word_out.append('O')

        return ''.join(word_out), len(word_out)


"""
Functions '_source_separation()' and 'separate_sources() adapted from PyTorch documentation.'
https://docs.pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html
"""
def _source_separation(
    model,
    audio,
    sr: float,
    segment: float=10.0,
    overlap: float=0.1,
    device: Optional[str]=None,
):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        segment (int): segment length in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    if device is None:
        device = audio.device
    else:
        device = torch.device(device)

    batch, channels, length = audio.shape

    chunk_len = int(sr * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sr
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = audio[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final


def separate_sources(
        *,
        audio: np.ndarray,
        sr: float,
        segment=10,
        overlap=0.1,
        stem_combinations: List[Stem]=[Stem.VOCALS, Stem.DRUMS, Stem.OTHER, Stem.BASS],
        device: Optional[torch.device] | Optional[str] = None) -> dict:
    """
    Separate a file into constituent parts. 
    Default is vocals and accompaniment with num_outputs=2. For vocals, drums, bass, and other, set num_outputs=4.
    Other values for num_outputs will throw an error.

    Note that output audio files are preprocessed after this function call according to audio.preprocess_audio()
    with default arguments.
    
    Args:
        audio (numpy ndarray): Audio data
        sr (float): Sample rate of the audio
        segment (int): segment length in seconds. See pytorch documentation for music source separation
        overlap (float): Overlap length in seconds. See pytorch documentation for music source separation
        stem_combinations (List[Stem]): Stems to combine. Stems can be combined with &. To combine vocals with drums and other with bass, write [Stem.VOCALS & Stem.DRUMS, Stem.OTHER & Stem.BASS]. To create vocals and accompaniment mixes, write [Stem.VOCALS, Stem.DRUMS & Stem.OTHER & Stem.BASS]. Default is to extract and return all four stems separately.
        device: Optional torch device. If None, will auto-detect.

    To combine vocals with drums
    """
    for combo in stem_combinations:
        if combo == 0:
            raise RuntimeError('All stem combinations must contain at least one stem.')

    # setup
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data = torch.from_numpy(audio.T).float()
    
    # HDemucs expects stereo audio (2 channels)
    # If mono, duplicate the channel to create stereo
    if data.dim() == 1:
        data = data.repeat(2, 1)
    
    data = data.to(device)

    ref = data.mean(0)
    data = (data - ref.mean()) / ref.std()  # normalization

    # separate audio
    sources = _source_separation(
        model,
        data[None],
        device=device,
        sr=sr,
        segment=segment,
        overlap=overlap,
    )[0]

    # Process sources and save to audio dictionary
    sources = sources * ref.std() + ref.mean()
    sources_list = model.sources
    sources = list(sources)
    audios = dict(zip(sources_list, sources))

    # Split stems apart
    stems = {
        'drums': audios["drums"].cpu().numpy(),
        'bass': audios["bass"].cpu().numpy(),
        'other': audios["other"].cpu().numpy(),
        'vocals': audios["vocals"].cpu().numpy()
    }

    # Save split audio
    aud_dict = {}
    for combo in stem_combinations:
        aud_combo = np.zeros(stems['bass'].shape)
        if combo & Stem.VOCALS:
            aud_combo += stems['vocals']
        if combo & Stem.BASS:
            aud_combo += stems['bass']
        if combo & Stem.OTHER:
            aud_combo += stems['other']
        if combo & Stem.DRUMS:
            aud_combo += stems['drums']

        aud_combo = np.float32(aud_combo)
        aud_combo = librosa.to_mono(aud_combo)
        aud_combo -= np.mean(aud_combo)

        aud_name, aud_count = Stem.english_map(combo)
        aud_combo /= aud_count
        aud_dict[aud_name] = aud_combo

    return aud_dict
        

        # if vocals.ndim == 2:
        #     vocals = vocals.T
        # vocals = np.float32(vocals) 
        # for channel in range(vocals.shape[1]):
        #     vocals[channel] -= np.mean(vocals[channel])

        # accompaniment = (drums + bass + other) / 3
        # if accompaniment.ndim == 2:
        #     accompaniment = accompaniment.T
        # accompaniment = np.float32(accompaniment)
        # for channel in range(accompaniment.shape[1]):
        #     accompaniment[channel] -= np.mean(accompaniment[channel])
