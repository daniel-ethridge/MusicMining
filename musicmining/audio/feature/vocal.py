import opensmile
from musicmining.audio import io


def vocal_from_file(file_path, sr, *, hop_size=None, block_size=None):
        

        smile = opensmile.Smile(
            feature_set = opensmile.FeatureSet.eGeMAPSv02,
            feature_level = opensmile.FeatureLevel.Functionals,
            sampling_rate=sr
        )

        processed = smile.process_file(file_path)
        processed = processed.reset_index()
        processed = processed.drop("file", axis=1)

        replace_cols = {}
        for col in processed.columns:
            replace_cols[col] = f"vocal_{col.replace('.', '_')}"

        processed = processed.rename(columns=replace_cols)
        processed = processed.drop(columns=['vocal_start', 'vocal_end'])

        return processed
    