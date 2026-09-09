from musicmining.audio import feature as feat
from enum import Flag
from typing import Optional
import pandas as pd


class Feat(Flag):
    NONE = 0
    DYNAMICS = 1 << 0
    EXPRESSIVITY = 1 << 1
    HARMONY = 1 << 2
    MELODY = 1 << 3
    RHYTHM = 1 << 4
    TIMBRE = 1 << 5
    VOCAL = 1 << 6


def extract_feats(
        audio, 
        sr, 
        *, 
        file_path: Optional[str] = None, 
        hop_size=1024, 
        block_size=4096, 
        feat_to_exclude: Feat = Feat.NONE):

    # Check for vocals path if necessary
    if feat_to_exclude & Feat.VOCAL == Feat.NONE:
        assert file_path is not None, "If vocals are included, a file path for the audio must be included as well."

    # Add features
    concat_list = []
    if feat_to_exclude & Feat.DYNAMICS == Feat.NONE:
        concat_list.append(feat.dynamics_from_audio(audio, sr, hop_size=hop_size, block_size=block_size))
    if feat_to_exclude & Feat.EXPRESSIVITY == Feat.NONE:
        concat_list.append(feat.expressivity_from_audio(audio, sr, hop_size=hop_size, block_size=block_size))
    if feat_to_exclude & Feat.HARMONY == Feat.NONE:
        concat_list.append(feat.harmony_from_audio(audio, sr, block_size=block_size))
    if feat_to_exclude & Feat.MELODY == Feat.NONE:
        concat_list.append(feat.melody_from_audio(audio, sr, hop_size=hop_size, block_size=block_size))
    if feat_to_exclude & Feat.RHYTHM == Feat.NONE:
        concat_list.append(feat.rhythm_from_audio(audio, sr))
    if feat_to_exclude & Feat.TIMBRE == Feat.NONE:
        concat_list.append(feat.timbre_from_audio(audio, sr, hop_size=hop_size, block_size=block_size))
    if feat_to_exclude & Feat.VOCAL == Feat.NONE:
        concat_list.append(feat.vocal_from_file(file_path, sr, hop_size=hop_size, block_size=block_size))

    return pd.concat(concat_list, axis=1)

    
