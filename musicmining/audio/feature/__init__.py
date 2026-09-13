"""
musicmining.audio.feature

musicmining.audio.feature contains convenience methods for quickly extracting common features from audio. Methods
from librosa and essentia are utilized and organized in order to provide easy access to these features.
"""

from .dynamics import dynamics_from_audio, dynamics_from_file
from .expressivity import expressivity_from_audio, expressivity_from_file
from .feature import Feat, extract_feats
from .harmony import harmony_from_audio, harmony_from_file
from .melody import melody_from_audio, melody_from_file
from .rhythm import rhythm_from_audio, rhythm_from_file
from .timbre import timbre_from_audio, timbre_from_file
from .vocal import vocal_from_file

__all__ = [
    dynamics_from_file,
    dynamics_from_audio,
    expressivity_from_file,
    expressivity_from_audio,
    Feat,
    extract_feats,
    harmony_from_file,
    harmony_from_audio,
    melody_from_file,
    melody_from_audio,
    rhythm_from_file,
    rhythm_from_audio,
    timbre_from_file,
    timbre_from_audio,
    vocal_from_file
]
