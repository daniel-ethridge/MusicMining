"""musicmining.sql provide functionality to easily save, load, and delete tables from a sqlite database."""

from pathlib import Path
from typing import Literal

import pandas as pd
from sqlalchemy import create_engine, text

from musicmining.util.exceptions import ParameterError


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
    # Check if path exists
    dir = Path(path).parent
    if not Path(dir).exists():
        raise ParameterError("Parent directory of file does not exist")

    if path == ":memory:":
        engine = create_engine("sqlite:///:memory:")
    else:
        engine = create_engine(f"sqlite:///{path}")

    with engine.connect() as conn:
        data.to_sql(table, conn, if_exists=if_exists, **kwargs)


def load(*, path: str, table: str, **kwargs) -> pd.DataFrame:
    """
    Load a pandas.DataFrame from a sqlite table.

    Parameters
    ----------
    path : str
        _description_
    table : str
        _description_
    **kwargs : additional keyword arguments
        See pandas.read_sql_table for details

    Returns
    -------
    pandas.DataFrame
        Data from the table
    """
    file_exists = Path(path).exists()
    if not file_exists:
        raise ParameterError("Database not found.")

    engine = create_engine(f"sqlite:///{path}")
    with engine.connect() as conn:
        data = pd.read_sql_table(table, conn, **kwargs)

    return data


def delete_table(*, path: str, table: str, if_not_exists: Literal["fail", "succeed"] = "fail"):
    """
    Delete a table from an existing database.

    Parameters
    ----------
    path : str
        The path to the existing database file
    table : str
        The table to drop if it exists.
    not_exists : {'fail', 'succeed' }, optional
        If the table does not exist, determines success and failure. Defaults to "fail".
    """
    file_exists = Path(path).exists()
    if not file_exists:
        raise ParameterError("Database not found.")

    engine = create_engine(f"sqlite:///{path}")
    with engine.connect() as conn:
        if if_not_exists == "succeed":
            conn.execute(text(f"DROP TABLE IF EXISTS {table};"))
        else:
            conn.execute(text(f"DROP TABLE {table};"))
        conn.commit()
