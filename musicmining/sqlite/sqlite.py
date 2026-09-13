"""musicmining.sql provide functionality to easily save, load, and delete tables from a sqlite database."""

from typing import Literal

import pandas as pd
from sqlalchemy import create_engine, text


def save(
    *,
    path: str,
    table: str,
    data: pd.DataFrame,
    if_exists: Literal["fail", "replace", "append", "delete_rows"] = "fail",
    **kwargs,
):
    """
    Save a table to a sqlite database.

    Creates the database if it does not exist, then saves the data to the file in the appropriate `table`.

    Parameters
    ----------
    path : str
        Path to the database, or the path where the database should be created. Note that the directory in which to
        place the sqlite file must already exist.
    table : str
        Name of the table.
    data : pandas.DataFrame
        The dataframe to save to a table.
    if_exists : {'fail', 'replace', 'append', 'delete_rows'}, optional
        How to behave if the table already exists. See pandas.DataFrame.to_sql for details. By default "fail".
    **kwargs : Additional keyword arguments
         See pandas.DataFrame.to_sql for details.
    """
    engine = create_engine(f"sqlite:///{path}")

    with engine.connect() as conn:
        data.to_sql(table, conn, if_exists=if_exists, **kwargs)


def load(
    *, path: str, table: str, if_exists: Literal["fail", "replace", "append", "delete_rows"] = "fail", **kwargs
) -> pd.DataFrame:
    """
    Load a pandas.DataFrame from a sqlite table.

    Parameters
    ----------
    path : str
        _description_
    table : str
        _description_
    if_exists : Literal['fail', 'replace', 'append', 'delete_rows'], optional
        How to behave if the table already exists. See pandas.DataFrame.to_sql for details. By default "fail".
    **kwargs : additional keyword arguments
        See pandas.read_sql_table for details

    Returns
    -------
    pandas.DataFrame
        Data from the table
    """
    engine = create_engine(f"sqlite:///{path}")
    with engine.connect() as conn:
        data = pd.read_sql_table(table, conn, if_exists=if_exists, **kwargs)

    return data


def delete_table(*, path: str, table: str):
    """
    Delete a table from an existing database.

    Parameters
    ----------
    path : str
        The path to the existing database file
    table : str
        The table to drop if it exists.
    """
    engine = create_engine(f"sqlite:///{path}")
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table};"))
        conn.commit()
