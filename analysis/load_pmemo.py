import argparse
import pandas as pd
from analysis.pmemo_constants import Paths
from musicmining.sql import save as sql_save, load as sql_load, delete_table as sql_delete_table
from musicmining.audio import io
from musicmining.audio.effects import pitch_shift_group
from musicmining.audio import feature as feat
from musicmining.audio.source_separation import separate_sources, Stem
from librosa.effects import pitch_shift
import os 
from multiprocessing import Pool, Manager
from typing import List, Optional
from pathlib import Path
import datetime
import numpy as np
import shutil


def generate_file_key(
        file_stem, 
        stems_in_file: Optional[Stem] | Optional[str] = None,
        pitch_shift: int = 0,
    ):

    if stems_in_file is not None:
        if type(stems_in_file) == str:
            word_out = stems_in_file
        else:
            word_out, word_length = Stem.english_map(stems_in_file)

            assert word_length != 4, "If all stems are in the file (i.e. no source separation used), pass an empty list to 'stems_in_file'"

            "".join(word_out)

    else:
        word_out = 'ALL'

    if pitch_shift > 0:
        shift = f"U{pitch_shift}"
    elif pitch_shift < 0:
        shift = f"D{-1*pitch_shift}"
    else:
        shift = "0"

    return f"{file_stem}_{word_out}_{shift}"


def parse_file_key(file_key: str):
    file_key_sep = file_key.split("_")

    if file_key_sep[1] == "ALL":
        full_mix = True
        vocals_only = False
    elif file_key_sep[1] == "V":
        full_mix = False
        vocals_only = True
    else:
        full_mix = False
        vocals_only = False

    if file_key_sep[2] == "0":
        pitch_shifted = False
    else:
        pitch_shifted = True

    return {
        "full_mix": full_mix,
        "vocals_only": vocals_only,
        "pitch_shifted": pitch_shifted,
        "parsed_key": file_key_sep
    }
    

def logging(results_iter):
    complete = 0
    for result in results_iter:
        yield result
        complete += 1

        if complete % 20 == 0:
            print(f"{complete} complete")


def worker_write_separated_audio(args):
    file, lock, ss_audio_dir = args

    audio, sr = io.load(file, sr=44100) 
    group = pitch_shift_group(audio, sr=sr, n_steps=np.array([-2, -1, 1, 2]), return_original=True)

    try: 
        for key, value in group.items():
            sources = separate_sources(
                audio=value,
                sr=sr,
                stem_combinations = [Stem.VOCALS, Stem.OTHER | Stem.BASS | Stem.DRUMS]
            )

            for s_key, s_value in sources.items():
                file_key = generate_file_key(Path(file).stem, s_key, key)

                io.write_wav(os.path.join(ss_audio_dir, f"{file_key}.wav"), sr, s_value)
        return 0
    
    except BaseException as e:
        if lock is not None:
            with lock:
                with open("error-logs.txt", 'a') as f:
                    f.write(f"{datetime.datetime.now()} :: {file_key} :: {type(e)} :: {str(e)}\n")
            return 1
        else:
            with open("error-logs.txt", 'a') as f:
                f.write(f"{datetime.datetime.now()} :: {file_key} :: {type(e)} :: {str(e)}\n")
            return 1


def worker_save_orig_audio(args):
    file, lock, database_path, audio_table_name = args
    file_stem = Path(file).stem

    feature_key_column = 'feature_key'
    try:
        # features
        audio, sr = io.load(file, sr=44100)

        file_key = generate_file_key(file_stem)
        orig_df = feat.extract_feats(audio, sr=sr, feat_to_exclude = feat.Feat.VOCAL)
        orig_df.insert(0, feature_key_column, file_key)

        group = pitch_shift_group(audio, sr=sr, n_steps=np.array([-2, -1, 1, 2]))
        file_key = generate_file_key(file_stem, pitch_shift=-2)
        down_2_df = feat.extract_feats(group[-2], sr=sr, feat_to_exclude = feat.Feat.VOCAL)
        down_2_df.insert(0, feature_key_column, file_key)

        file_key = generate_file_key(file_stem, pitch_shift=-1)
        down_1_df = feat.extract_feats(group[-1], sr=sr, feat_to_exclude = feat.Feat.VOCAL)
        down_1_df.insert(0, feature_key_column, file_key)

        file_key = generate_file_key(file_stem, pitch_shift=1)
        up_1_df = feat.extract_feats(group[1], sr=sr, feat_to_exclude = feat.Feat.VOCAL)
        up_1_df.insert(0, feature_key_column, file_key)

        file_key = generate_file_key(file_stem, pitch_shift=2)
        up_2_df = feat.extract_feats(group[2], sr=sr, feat_to_exclude = feat.Feat.VOCAL)
        up_2_df.insert(0, feature_key_column, file_key)

        full_df = pd.concat([orig_df, down_2_df, down_1_df, up_1_df, up_2_df], axis=0)
        # write to db
        if lock is not None:
            with lock:
                sql_save(
                    path=database_path, 
                    table=audio_table_name,
                    data=full_df,
                    if_exists='append', 
                    index=False)

        else:
            sql_save(
                path=database_path, 
                table=audio_table_name,
                data=full_df,
                if_exists='append', 
                index=False)
        return 0
    
    except BaseException as e:
        with lock:
            with open("error-logs.txt", 'a') as f:
                f.write(f"{datetime.datetime.now()} :: {file_key} :: {type(e)} :: {str(e)}\n")
        return 1


def worker_save_sep_audio(args):
    file, lock, database_path, vocal_table_name, acc_table_name = args

    feature_key_column = 'feature_key'
    try:
        # features
        audio, sr = io.load(file, sr=44100)

        file_key = Path(file).stem
        parsed_key = parse_file_key(file_key)

        if parsed_key['vocals_only']:
            df = feat.extract_feats(audio, sr=sr, file_path=file)
        else:
            df = feat.extract_feats(audio, sr=sr, feat_to_exclude = feat.Feat.VOCAL)
        df.insert(0, feature_key_column, file_key)

        # write to db
        if lock is not None:
            if parsed_key['vocals_only']:
                with lock:
                    sql_save(
                        path=database_path, 
                        table=vocal_table_name,
                        data=df,
                        if_exists='append', 
                        index=False)
            else:
                with lock:
                    sql_save(
                        path=database_path, 
                        table=acc_table_name,
                        data=df,
                        if_exists='append', 
                        index=False)

        else:
            if parsed_key['vocals_only']:
                sql_save(
                    path=database_path, 
                    table=vocal_table_name,
                    data=df,
                    if_exists='append', 
                    index=False)
            else:
                sql_save(
                    path=database_path, 
                    table=acc_table_name,
                    data=df,
                    if_exists='append', 
                    index=False)
        return 0
    
    except BaseException as e:
        with lock:
            with open("error-logs.txt", 'a') as f:
                f.write(f"{datetime.datetime.now()} :: {file_key} :: {type(e)} :: {str(e)}\n")
        return 1


def pmemo_csv_to_sqlite(
        *, 
        database_path: str = Paths.pmemo_database,
        original_audio_table_name: str = "original_audio_features",
        vocals_audio_table_name: str = "vocal_audio_features",
        accompaniment_audio_table_name: str = "accompaniment_audio_features",
        annotations_table_name: str = "annotations",
        overwrite_annotations: bool = False,
        overwrite_original_audio_feats: bool = False,
        overwrite_separated_audio_feats: bool = False,
        n_jobs: int=12,
        n_gpu_jobs: int=1):
    """
    1. Write the static annotations csv file to database file
    2. Extract features from original audio
    3. Save to audio_features_table
    4. Separate audio sources
    5. Extract features from source separated audio
    6. Save to audio_features_table
    """

    # ## write annotations to sqlite ##
    if overwrite_annotations:  # overwriting annotations for sure
        df = pd.read_csv(Paths.pmemo_static_annots_file)
        df = df.rename(columns={
            "musicId": "music_id",
            "Arousal(mean)": "arousal",
            "Valence(mean)": "valence"
        }, inplace=False)

        sql_save(
            path=database_path, 
            table=annotations_table_name,
            data=df,
            if_exists='replace',
            index=False)

    # Need to check if exists
    else:
        try:
            _ = sql_load(path=database_path, table=annotations_table_name)
        except pd.errors.DatabaseError:
            df = pd.read_csv(Paths.pmemo_static_annots_file)
            df = df.rename(columns={
                "musicId": "music_id",
                "Arousal(mean)": "arousal",
                "Valence(mean)": "valence"
            }, inplace=False)

            sql_save(
                path=database_path, 
                table=annotations_table_name,
                data=df,
                index=False)


    if overwrite_original_audio_feats:
        sql_delete_table(path=database_path, table=original_audio_table_name)

        # ## extract audio features ##
        files = [os.path.join(Paths.pmemo_audio_dir, file) for file in os.listdir(Paths.pmemo_audio_dir)]

        if n_jobs > 1:  # Multiprocessing
            with Manager() as manager:
                lock = manager.Lock()
                worker_args = [(file, lock, database_path, original_audio_table_name) for file in files]
                with Pool(n_jobs) as p:
                    results_iterator = p.imap(worker_save_orig_audio, worker_args)
                    _ = list(logging(results_iterator))

        else:  # Single process
            complete = 0
            for file in files:
                worker_args = (file, None, database_path, original_audio_table_name)
                worker_save_orig_audio(worker_args)
                complete += 1
                if complete % 20 == 0:
                    print(f"{complete} complete")

    if overwrite_separated_audio_feats:
        # Need a temp directory for source separated files
        ss_audio_dir = os.path.join(Paths.pmemo, "temp_split_audio")
        Path(ss_audio_dir).mkdir(exist_ok=True)  # start directory lifespan
        files = [os.path.join(Paths.pmemo_audio_dir, file) for file in os.listdir(Paths.pmemo_audio_dir)]

        # Save all separated files to a directory
        # if n_gpu_jobs > 1:
        #     with Manager() as manager:
        #         lock = manager.Lock()
        #         with Pool(n_gpu_jobs) as p:
        #             worker_args = [(file, lock, ss_audio_dir) for file in files]
        #             results_iterator = p.imap(worker_write_separated_audio, worker_args)
        #             _ = list(logging(results_iterator))

        # else:
        #     complete = 0
        #     print("1 process")
        #     for file in files:
        #         worker_args = (file, None, ss_audio_dir)
        #         worker_write_separated_audio(worker_args)
        #         complete += 1
        #         if complete % 20 == 0:
        #             print(f"{complete} complete")

        files = [os.path.join(ss_audio_dir, file) for file in os.listdir(ss_audio_dir)]
        if n_jobs > 1:  # Multiprocessing
            with Manager() as manager:
                lock = manager.Lock()
                worker_args = [
                    (file, lock, database_path, vocals_audio_table_name, accompaniment_audio_table_name) 
                    for file in files]
                with Pool(n_jobs) as p:
                    results_iterator = p.imap(worker_save_sep_audio, worker_args)
                    _ = list(logging(results_iterator))

        else:  # Single process
            complete = 0
            for file in files:
                worker_args = (file, None, database_path, vocals_audio_table_name, accompaniment_audio_table_name)
                worker_save_sep_audio(worker_args)
                complete += 1
                if complete % 20 == 0:
                    print(f"{complete} complete")

                # audio, sr = io.load(file, sr=44100) 
                # group = pitch_shift_group(audio, sr=sr, n_steps=np.array([-2, -1, 1, 2]), return_original=True)

                # for key, value in group.items():
                #     sources = separate_sources(
                #         audio=value,
                #         sr=sr,
                #         stem_combinations = [Stem.VOCALS, Stem.OTHER | Stem.BASS | Stem.DRUMS]
                #     )

                #     for s_key, s_value in sources.items():
                #         file_key = generate_file_key(Path(file).stem, s_key, key)

                #         io.write_wav(os.path.join(ss_audio_dir, f"{file_key}.wav"), sr, s_value)

        # shutil.rmtree(ss_audio_dir)  # end directory lifespan

    # for file in files:
    #     # load and preprocess audio
    #     audio, sr = io.load(file, sr=44100)

    #     # Extract features
    #     # feature_dict = {}
        
    #     """
    #         - Make sure sample rate, block size, and hop size are all included where necessary
    #         - 
    #     """
    #     full_df = feat.extract_feats(audio, sr, file_path=file)

    #     # Store in sqlite
    #     sql_save(
    #         path=database_path, 
    #         table=audio_table_name,
    #         data=full_df,
    #         if_exists='append')

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", default="./pmemo.db")
    # parser.add_argument("-a")
    
    pmemo_csv_to_sqlite(overwrite_separated_audio_feats=True, n_jobs=12, n_gpu_jobs=2)


#     # Read in data
#     responses = pd.read_csv("./datasets/PMEmo2019/PMEmo2019/features/static_features.csv")

#     audio_folder = "./datasets/PMEmo2019/PMEmo2019/chorus"
#     files = os.listdir(audio_folder)

#     for f in files:
#         file = os.path.join(audio_folder, f)

#         # load and preprocess audio
#         audio = io.load(file, sr=44100)

#         # Separate sources
#         sources = ss.separate_sources(
#             audio=audio,
#             stem_combinations = [ss.Stem.VOCALS, ss.Stem.OTHER & ss.Stem.BASS & ss.Stem.DRUMS]
#         )

#         # Extract features
#         feature_dict = {}
#         for key, value, in sources.items():

#             """
#             TODO: 
#                 - Make sure sample rate, block size, and hop size are all included where necessary
#                 - 
#             """
#             dynamics = feat.dynamics_feats()
#             expressivity = feat.expressivity_feats()
#             harmony = feat.harmony_feats()
#             melody = feat.melody_feats()
#             rhythm = feat.rhythm_feats()
#             timbre = feat.timbre_feats()
#             vocal = feat.vocal_feats() 

#             # We assume that all dataframes have the same number of rows
#             full_df = pd.concat([dynamics, expressivity, harmony, melody, rhythm, timbre, vocal], axis=1)

#             # Normalize features
#             scaler = StandardScaler()
#             full_df = scaler.fit_transform(full_df)

#             # Store in sqlite
#             mmSqlt.save(database="", table="", data=full_df)


# if __name__ == "__main__":
#     main()
