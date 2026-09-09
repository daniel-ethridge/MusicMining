import pandas as pd
import numpy as np
from musicmining.sql import load as sql_load
from analysis.pmemo_constants import Paths
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split, KFold, RepeatedKFold
from sklearn.metrics import accuracy_score
from typing import Generator, Optional

from analysis.pmemo_splitter import AugmentedPMemoSplitter


def main():
    df = sql_load(path="pmemo.db", table="original_audio_features")

    splitter = AugmentedPMemoSplitter(df_keys_idxs=df.copy()['feature_key'], n_splits=10, n_repeats=30)

    labels = sql_load(path="pmemo.db", table="annotations").drop('arousal', axis=1)
    df['music_id'] = df['feature_key'].apply(lambda x: int(x.split("_")[0]))
    df_data = pd.merge(left=df, right = labels, how='inner', on='music_id').drop(['music_id'], axis=1)

    cols = [col for col in df.columns if "harmony" in col]
    df_data = df_data.drop(cols, axis=1)
    labels = df_data.pop('valence')

    parameters = {
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': list(np.arange(5, 20)),
        'n_estimators': list(np.arange(10, 20)),
        'max_features': ["sqrt", "log2"]
    }

    forest = RandomForestRegressor()
    clf = GridSearchCV(
        estimator=forest, 
        param_grid=parameters,
        cv=splitter, 
        verbose=1,
        n_jobs=12)

    df_data = df_data.set_index(['feature_key'])
    output = cross_validate(
        clf, 
        df_data,
        labels,
        cv=splitter, 
        return_estimator=True, 
        return_train_score=True,
        n_jobs=12
    )

    breakpoint()


if __name__ == "__main__":
    main()



# def generate_split_idxs(df, train_pct, k=5):
#     for _ in range(k):
#         df_orig = df[df['feature_key'].apply(lambda x: '0' in x.split('_')[2])]
#         df_shifted = df[df['feature_key'].apply(lambda x: '0' not in x.split('_')[2])]

#         # get keys, create splits
#         keys = df_orig['feature_key']
#         keys = list(keys.sample(frac=1))
#         threshold = int(np.floor(len(keys) * train_pct))

#         test_keys = keys[threshold:]
#         test_set = df_orig[df_orig['feature_key'].apply(lambda x: x in test_keys)]

#         train_keys = keys[:threshold]
#         train_set = df_orig[df_orig['feature_key'].apply(lambda x: x in train_keys)]

#         df_shifted['train_key'] = df_shifted['feature_key'].apply(
#             lambda x: True if x.split("_")[0] in [id.split("_")[0] for id in train_keys] else False
#         )

#         train_set = pd.concat((train_set, df_shifted[df_shifted['train_key'] == True])).drop('train_key', axis=1)

#         # sanity check
#         train_set_ids = list(train_set["feature_key"])
#         train_set_ids = list(set([id.split('_')[0] for id in train_set_ids]))
#         test_set_ids = list(test_set["feature_key"])
#         test_set_ids = list(set([id.split('_')[0] for id in test_set_ids]))
#         both_lists = train_set_ids + test_set_ids
#         both_sets = set(both_lists)
#         assert len(both_lists) == len(both_sets), "Something went wrong"

#         # sanity check 2
#         train_set_idxs = np.array(train_set['array_idx'])
#         assert len(train_set_idxs) == len(set(train_set_idxs))
#         test_set_idxs = np.array(test_set['array_idx'])
#         assert len(test_set_idxs) == len(set(test_set_idxs))
#         all_idxs = list(train_set_idxs) + list(test_set_idxs)
#         assert len(all_idxs) == len(set(all_idxs))

#         # # add labels
#         # train_set["music_id"] = train_set['feature_key'].apply(lambda x: int(x.split("_")[0]))
#         # test_set["music_id"] = test_set['feature_key'].apply(lambda x: int(x.split("_")[0]))
#         # train_set = pd.merge(left=train_set, right=y, how='inner', on='music_id')
#         # test_set = pd.merge(left=test_set, right=y, how='inner', on='music_id')

#         yield (train_set_idxs, test_set_idxs)








