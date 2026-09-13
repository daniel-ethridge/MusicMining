"""Extract timbre features from essentia and librosa."""

import essentia.standard as es
import librosa.feature as feat
import numpy as np
import pandas as pd

from musicmining.audio import io


def timbre_from_file(file_path: str, sr: int, hop_size=1024, block_size=4096):
    """
    Extract timbred features from a file.

    Parameters
    ----------
    file_path : str
        Path to an audio file.
    sr : int
        Target sample rate of audio
    hop_size : int, optional
        STFT hop length for librosa and essentia methods.
    block_size : int, optional
        FFT window size for librosa and essentia methods.

    Returns
    -------
    pandas.DataFrame
        Timbre features out.
    """
    audio, rate = io.load(file_path, sr=sr)
    return timbre_from_audio(audio, rate, hop_size=hop_size, block_size=block_size)


def timbre_from_audio(data: np.ndarray, sr: int, *, hop_size: int = 1024, block_size: int = 4096):
    """
    Extract timbre features from audio.

    Parameters
    ----------
    data : np.ndarray
        Audio data
    sr : int
        Sample rate of the audio
    hop_size : int, optional
        STFT hop length for librosa and essentia methods. Defaults to 1024.
    block_size : int, optional
        FFT window size for librosa and essentia methods. Defaults to 4096.

    Returns
    -------
    pandas.DataFrame
        Timbre features out.
    """
    ## Extract features ##
    # Create dictionary
    feature_data = {}

    # spectral centroid
    spec_cent = feat.spectral_centroid(y=data, sr=sr, hop_length=hop_size, n_fft=block_size).flatten()
    feature_data["timbre_avg_spec_cent"] = [np.mean(spec_cent)]
    feature_data["timbre_std_spec_cent"] = [np.std(spec_cent)]

    # spectral bandwidth
    spec_band = feat.spectral_bandwidth(y=data, sr=sr, hop_length=hop_size, n_fft=block_size).flatten()
    feature_data["timbre_avg_spec_band"] = [np.mean(spec_band)]
    feature_data["timbre_std_spec_band"] = [np.std(spec_band)]

    # spectral contrast
    spec_cont = feat.spectral_contrast(y=data, sr=sr, hop_length=hop_size, n_fft=block_size)
    for i in range(spec_cont.shape[0]):
        extracted_feature = spec_cont[i, :].flatten()
        feature_data[f"timbre_avg_spec_cont_{i + 1}"] = [np.mean(extracted_feature)]
        feature_data[f"timbre_std_spec_cont_{i + 1}"] = [np.std(extracted_feature)]

    # spectral flatness
    spec_flat = feat.spectral_flatness(y=data, hop_length=hop_size, n_fft=block_size).flatten()
    feature_data["timbre_avg_spec_flat"] = [np.mean(spec_flat)]
    feature_data["timbre_std_spec_flat"] = [np.std(spec_flat)]

    # spectral rolloff
    spec_flat = feat.spectral_rolloff(y=data, sr=sr, hop_length=hop_size, n_fft=block_size).flatten()
    feature_data["timbre_avg_spec_rolloff"] = [np.mean(spec_flat)]
    feature_data["timbre_std_spec_rolloff"] = [np.std(spec_flat)]

    # mfccs
    mfccs = feat.mfcc(y=data, sr=sr)
    for i in range(mfccs.shape[0]):
        extracted_feature = mfccs[i, :].flatten()
        feature_data[f"timbre_avg_mfcc_{i + 1}"] = [np.mean(extracted_feature)]
        feature_data[f"timbre_std_mfcc_{i + 1}"] = [np.std(extracted_feature)]

    # Zero crossings
    zero_cross = feat.zero_crossing_rate(y=data, hop_length=hop_size, frame_length=block_size).flatten()
    feature_data["timbre_avg_zero_cross"] = [np.mean(zero_cross)]
    feature_data["timbre_std_zero_cross"] = [np.std(zero_cross)]

    # Odd-even Harmonic preprocessing
    eq_loud = es.EqualLoudness(sampleRate=sr)
    pitch_melodia = es.PitchMelodia(sampleRate=sr, hopSize=hop_size, frameSize=block_size)
    pitch_filter = es.PitchFilter()
    eq_data = eq_loud(data)
    pitches, pitch_confidence = pitch_melodia(eq_data)
    corrected_pitches = pitch_filter(pitches, pitch_confidence)

    # Odd-Even Harmonic Energy -- Create objects
    spectrum = es.Spectrum(size=block_size)
    spectral_peaks = es.SpectralPeaks(sampleRate=sr, minFrequency=20)
    harmonic_peaks = es.HarmonicPeaks()
    odd_even = es.OddToEvenHarmonicEnergyRatio()

    # Attack time -- create objects
    onset_d1 = es.OnsetDetection(sampleRate=sr, method="hfc")
    onset_d2 = es.OnsetDetection(sampleRate=sr, method="flux")
    onset_d3 = es.OnsetDetection(sampleRate=sr, method="rms")

    # get frames
    odd_to_even_ratios = []
    log_attack_times = []
    frames = []
    for frame in es.FrameGenerator(data, block_size, hop_size):
        frames.append(frame)

    detections = np.zeros((3, len(frames)))
    # clipped = False
    for i, frame in enumerate(frames):
        # even odd
        spectrum_out = spectrum(frame)
        frequencies, magnitudes = spectral_peaks(spectrum_out)
        harm_frqs, harm_mags = harmonic_peaks(frequencies, magnitudes, corrected_pitches[i])
        odd_to_even_ratios.append(odd_even(harm_frqs, harm_mags))
        # if odd_to_even_ratios[-1] == 1000:
        #     clipped = True

        # onset detection
        detections[0][i] = onset_d1(spectrum_out, spectrum_out)
        detections[1][i] = onset_d2(spectrum_out, spectrum_out)
        detections[2][i] = onset_d3(spectrum_out, spectrum_out)

    onsets = es.Onsets()
    envelope = es.Envelope()
    log_attack = es.LogAttackTime(sampleRate=sr)
    weights = np.ones(3) * 1 / 3
    weights = weights.astype(np.float32)
    detections = detections.astype(np.float32)
    onsets_out = onsets(detections, weights)

    for onset_time in onsets_out:
        onset_sample = onset_time * sr
        frame = data[int(onset_sample) : int(onset_sample + block_size)]
        data_envel = envelope(frame)
        log_attack_times.append(log_attack(data_envel)[0])

    # Save odd to even ratios
    odd_to_even_ratios = np.array(odd_to_even_ratios)
    feature_data["timbre_min_odd_even_ratio"] = [np.min(odd_to_even_ratios)]
    feature_data["timbre_max_odd_even_ratio"] = [np.max(odd_to_even_ratios)]
    feature_data["timbre_avg_odd_even_ratio"] = [np.mean(odd_to_even_ratios)]
    feature_data["timbre_med_odd_even_ratio"] = [np.median(odd_to_even_ratios)]
    feature_data["timbre_std_odd_even_ratio"] = [np.std(odd_to_even_ratios)]
    # feature_data["timbre_odd_even_ratio_clipped"] = [clipped]

    # Save attack times
    attack_times = [10**val for val in log_attack_times]
    log_attack_times = np.array(log_attack_times)
    attack_times = np.array(attack_times)

    feature_data["timbre_avg_attack_times"] = [np.mean(attack_times)]
    feature_data["timbre_std_attack_times"] = [np.std(attack_times)]
    feature_data["timbre_avg_log_attack_times"] = [np.mean(log_attack_times)]
    feature_data["timbre_std_log_attack_times"] = [np.std(log_attack_times)]

    return pd.DataFrame(feature_data)
