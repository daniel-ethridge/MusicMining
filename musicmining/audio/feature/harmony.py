import numpy as np
from musicmining.audio import io
import essentia.standard as es
import pandas as pd


def harmony_from_file(file_path, sr, *, block_size=4096):
    data, sr = io.load(file_path, sr=sr)
    return harmony_from_audio(data, sr, block_size=block_size)
    

def harmony_from_audio(data, sr, *, block_size=4096):
            
    if len(data) % 2 != 0:
        data = np.pad(data, (0, 1))
    # data = preprocess_audio(data)

    ## Extract features ##
    # Create dictionary
    feature_data = {}

    # Odd-even Harmonic preprocessing
    # eq_loud = es.EqualLoudness(sampleRate=sr)
    # pitch_melodia = es.PitchMelodia(sampleRate=sr, hopSize=hop_size, frameSize=block_size)
    # pitch_filter = es.PitchFilter()
    # eq_data = eq_loud(data)
    # pitches, pitch_confidence = pitch_melodia(eq_data)
    # corrected_pitches = pitch_filter(pitches, pitch_confidence)

    # Harmonicities Energy -- Create objects
    spectrum = es.Spectrum(size=block_size)
    spectral_peaks = es.SpectralPeaks(sampleRate=sr)
    # harmonic_peaks = es.HarmonicPeaks()
    # inharmonicity = es.Inharmonicity()

    # Key -- create objects
    hpcp = es.HPCP()
    key_extractor = es.Key()

    # # itesr over frames
    # inharmonicities = []
    # for i, frame in enumesr(es.FrameGenerator(data, block_size, hop_size)):
    #     # inharmonicities
    #     spectrum_out = spectrum(frame)
    #     frequencies, magnitudes = spectral_peaks(spectrum_out)
    #     harm_frqs, harm_mags = harmonic_peaks(frequencies, magnitudes, corrected_pitches[i])
    #     inharmonicities.append(inharmonicity(harm_frqs, harm_mags))

    # print(inharmonicities)

    # # Save inharmonicities
    # inharmonicities = np.array(inharmonicities)     
    # feature_data["harmony_avg_inharmonicity"] = [np.mean(inharmonicities)]
    # feature_data["harmony_std_inharmonicity"] = [np.std(inharmonicities)]

    # Get key info

    spectrum_out = spectrum(data)
    frequencies, magnitudes = spectral_peaks(spectrum_out)
    pcp = hpcp(frequencies, magnitudes)
    key_attributes = key_extractor(pcp)
    feature_data["harmony_key"] = [key_attributes[0]]
    feature_data["harmony_modality"] = [key_attributes[1]]
    feature_data["harmony_key_confidence"] = [key_attributes[2]]

    return pd.DataFrame(feature_data)




# def harmony(
#     file_path,
#     *,
#     hop_size,
#     block_size
# ) -> pd.DataFrame:
    
#     # Get audio
#     sr, data = wavread(filename=file_path)
#     if len(data) % 2 != 0:
#         data = np.pad(data, (0, 1))
#     # data = preprocess_audio(data)

#     ## Extract features ##
#     # Create dictionary
#     feature_data = {}

#     # Odd-even Harmonic preprocessing
#     # eq_loud = es.EqualLoudness(sampleRate=sr)
#     # pitch_melodia = es.PitchMelodia(sampleRate=sr, hopSize=hop_size, frameSize=block_size)
#     # pitch_filter = es.PitchFilter()
#     # eq_data = eq_loud(data)
#     # pitches, pitch_confidence = pitch_melodia(eq_data)
#     # corrected_pitches = pitch_filter(pitches, pitch_confidence)

#     # Harmonicities Energy -- Create objects
#     spectrum = es.Spectrum(size=block_size)
#     spectral_peaks = es.SpectralPeaks(sampleRate=sr)
#     # harmonic_peaks = es.HarmonicPeaks()
#     # inharmonicity = es.Inharmonicity()

#     # Key -- create objects
#     hpcp = es.HPCP()
#     key_extractor = es.Key()

#     # # itesr over frames
#     # inharmonicities = []
#     # for i, frame in enumesr(es.FrameGenerator(data, block_size, hop_size)):
#     #     # inharmonicities
#     #     spectrum_out = spectrum(frame)
#     #     frequencies, magnitudes = spectral_peaks(spectrum_out)
#     #     harm_frqs, harm_mags = harmonic_peaks(frequencies, magnitudes, corrected_pitches[i])
#     #     inharmonicities.append(inharmonicity(harm_frqs, harm_mags))

#     # print(inharmonicities)

#     # # Save inharmonicities
#     # inharmonicities = np.array(inharmonicities)     
#     # feature_data["harmony_avg_inharmonicity"] = [np.mean(inharmonicities)]
#     # feature_data["harmony_std_inharmonicity"] = [np.std(inharmonicities)]

#     # Get key info

#     spectrum_out = spectrum(data)
#     frequencies, magnitudes = spectral_peaks(spectrum_out)
#     pcp = hpcp(frequencies, magnitudes)
#     key_attributes = key_extractor(pcp)
#     feature_data["harmony_key"] = [key_attributes[0]]
#     feature_data["harmony_modality"] = [key_attributes[1]]
#     feature_data["harmony_key_confidence"] = [key_attributes[2]]

#     return pd.DataFrame(feature_data)