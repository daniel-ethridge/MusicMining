from typing import Optional, Generator, Literal
from sklearn.model_selection import KFold, RepeatedKFold
import pandas as pd
import numpy as np


def split_data_and_labels(
        pmemo_data: pd.DataFrame, 
        annotations: pd.DataFrame,
        label: Literal["valence", "arousal"]="valence"):
    
    if label != "arousal" and label != "valence":
        raise RuntimeError("Bad label")

    pmemo_data['music_id'] = pmemo_data['feature_key'].apply(lambda x: int(x.split("_")[0]))
    df_data = pd.merge(left=pmemo_data, right = annotations, how='inner', on='music_id').drop(['music_id'], axis=1)
    labels = df_data.pop(label)

    return df_data, labels


class AugmentedPMemoSplitter:
    def __init__(
            self, 
            df_keys_idxs: pd.DataFrame,
            n_splits: int=5,
            *,
            shuffle=False,
            random_state=None,
            n_repeats: Optional[int]=None):

        self.df_keys_idxs = df_keys_idxs
        self.n_splits = n_splits

        if n_repeats is None or n_repeats == 1:
            self.kfold = KFold(n_splits, shuffle=shuffle, random_state=random_state)
        elif n_repeats == 1:
            self.kfold = KFold(n_splits, shuffle=shuffle, random_state=random_state)
        else:
            self.kfold = RepeatedKFold(n_splits=n_splits, random_state=random_state)


    def split(self, X, y=None, groups=None) -> Generator[np.ndarray, np.ndarray]:

        x_copy = X.reset_index()
        for split in self.kfold.split(x_copy):

            # train split -- easy part
            train_split = split[0]

            # test split -- need to remove 
            test_split = split[1]
            test_set = x_copy.iloc[test_split]
            reduced_test_set = test_set[test_set['feature_key'].apply(lambda x: x.split("_")[2] == "0")]
            reduced_test_split = np.array(reduced_test_set.index)
            yield train_split, reduced_test_split


    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# if __name__ == "__main__": 

#     from musicmining.sql import load as sql_load
#     from analysis.pmemo_constants import Paths, SqlNames
    
#     df = sql_load(
#         path=Paths.pmemo_database,
#         table=SqlNames.original_audio_table
#     )

#     annots = sql_load(
#         path=Paths.pmemo_database,
#         table=SqlNames.annotations_table
#     )

#     data, labels = split_data_and_labels(df, annots, 'valence')
    