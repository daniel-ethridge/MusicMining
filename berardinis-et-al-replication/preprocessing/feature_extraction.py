import os

import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt


def main():

    # Allocate memory for final dataframe
    log_mels_df = None
    
    # Read in audio metadata. Iterate over each file in the chorus directory
    df = pd.read_csv("track_metadata.csv")
    df = df.set_index("track_id")
    for root, _, files in os.walk("datasets/PMEmo2019/PMEmo2019/chorus"):
        for count, file in enumerate(files):
            try:
                print(f"file {count} of {len(files)}")
                mp3_file = os.path.join(root, file)
                file_id = int(file.split(".")[0])
                offset = df.loc[file_id]['offset']
            except KeyError:
                continue

            # Read in audio file at 44100. trim to 20 seconds
            audio, _ = librosa.load(mp3_file, sr=44100, mono=True)
            start = int(offset * 44100)
            end = int((offset + 20) * 44100)

            if end <= len(audio):
                audio = audio[start:end]
            else:
                audio = audio[start:]
                audio = librosa.util.fix_length(audio, size=20*44100)

            mel_spec = librosa.feature.melspectrogram(y=audio, sr=44100, n_fft=512, hop_length=256, n_mels=96)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            # fig, ax = plt.subplots()
            # img = librosa.display.specshow(
            #     log_mel_spec, x_axis='time', y_axis='mel', sr=44100, n_fft=512, hop_length=256, fmax=8000, ax=ax)
            # fig.colorbar(img, ax=ax, format='%+2.0f dB')
            # ax.set(title='Log-Mel Spectrogram')
            # plt.show()

            if log_mels_df is None:
                cols = [f"mel{i}" for i in np.arange(0, log_mel_spec.shape[1])]
                log_mels_df = pd.DataFrame(log_mel_spec, columns=cols)
                log_mels_df.insert(0, 'track_id', file_id)
            else:
                cols = [f"mel{i}" for i in np.arange(0, log_mel_spec.shape[1])]
                temp_df = pd.DataFrame(log_mel_spec, columns=cols)
                temp_df.insert(0, 'track_id', file_id)
                log_mels_df = pd.concat([log_mels_df, temp_df])

        break  # Break out of os.walk entirely

    log_mels_df.to_csv("log_mels.csv", index=False)
    print("df written")


if __name__ == "__main__":
    main()