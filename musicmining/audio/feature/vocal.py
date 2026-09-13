"""Extract vocal features from the opensmile package."""

from typing import Optional

import opensmile
import pandas as pd


def vocal_from_file(
    file_path: str,
    sr : int,
    *,
    hop_size: Optional[int]=None,
    block_size: Optional[int]=None
) -> pd.DataFrame:
    """
    Extract vocal features from a file.

    This is a wrapper around opensmile.Smile.process_file

    Parameters
    ----------
    file_path : str
        Path to an audio file.
    sr : int
        Target sample rate in hertz.
    hop_size : int, optional
        Currently has no effect.
    block_size : int, optional
        Currently has no effect.

    Returns
    -------
    pandas.DataFrame
        Vocal features out.
    """
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
    processed = processed.drop(columns=["vocal_start", "vocal_end"])

    return processed
