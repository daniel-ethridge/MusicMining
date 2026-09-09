import pytest

import pandas as pd
from typing import Optional
from musicmining.sql import load as sql_load
from analysis.pmemo_constants import Paths, SqlNames
from analysis.pmemo_splitter import split_data_and_labels


@pytest.mark.split
@pytest.mark.parametrize("label", ["valence", "arousal"])
@pytest.mark.parametrize(
    "table", [SqlNames.original_audio_table, SqlNames.accompaniment_audio_table, SqlNames.vocals_audio_table])
def test_split_data(table, label):
    df = sql_load(
        path=Paths.pmemo_database,
        table=table
    )

    annots = sql_load(
        path=Paths.pmemo_database,
        table=SqlNames.annotations_table
    )

    data, labels = split_data_and_labels(df, annots, label)

    assert data.shape[0] == labels.shape[0]
    assert data.shape[0] != 0
    assert labels.shape == (data.shape[0], )
    assert labels.name == label


@pytest.mark.split
@pytest.mark.xfail(raises=RuntimeError)
def test_split_data_bad_label():
    df = sql_load(
        path=Paths.pmemo_database,
        table=SqlNames.original_audio_table
    )

    annots = sql_load(
        path=Paths.pmemo_database,
        table=SqlNames.annotations_table
    )

    _ = split_data_and_labels(df, annots, ' ')

