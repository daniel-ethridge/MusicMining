from musicmining.audio import io
import librosa.feature as feat
import numpy as np
import pandas as pd
import essentia.standard as es



def dynamics_from_file(file_path, sr, *, hop_size=1024, block_size=4096):
        data, sr = io.load(file_path, sr=sr)
        return dynamics_from_audio(data, sr, hop_size=hop_size, block_size=block_size)


def dynamics_from_audio(data, sr, *, hop_size=1024, block_size=4096):

        feature_data = {}
        feat_rms = feat.rms(y=data, frame_length=block_size, hop_length=hop_size).flatten()
        max_rms = np.max(feat_rms)
        if (max_rms == 0):
            rms_on_mean = np.nan
        else:
            rms_on = np.array([0 if val/max_rms < 0.1 else 1 for val in feat_rms])
            rms_on_mean = np.mean(rms_on)
        feature_data["dynamics_avg_rms"] = [np.mean(feat_rms)]
        feature_data["dynamics_std_rms"] = [np.std(feat_rms)]
        feature_data["dynamics_percent_active"] = [rms_on_mean]

        # Extract loudness (essentia)
        loudness = es.Loudness()
        loudness_values = []

        for frame in es.FrameGenerator(data, block_size, hop_size):
            loudness_values.append(loudness(frame))

        loudness_values = np.array(loudness_values)
        feature_data["dynamics_frames_avg_loudness"] = [np.mean(loudness_values)]
        feature_data["dynamics_frames_std_loudness"] = [np.std(loudness_values)]
        feature_data["dynamics_full_track_loudness"] = [loudness(data)]

        # Extract dynamic complexity (essentia)
        dynamic_complexity = es.DynamicComplexity(frameSize=block_size/sr, sampleRate=sr)
        dynamic_comp_out, loudness_out = dynamic_complexity(data)
        feature_data["dynamics_dynamic_complexity"] = [dynamic_comp_out]
        feature_data["dynamics_dynamic_complexity_loudness_db"] = [loudness_out]

        return pd.DataFrame(feature_data)


def dynamics_feats():
    pass