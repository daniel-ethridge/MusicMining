import pytest

import pandas as pd
import numpy as np

from musicmining.sqlite import save as sql_save, load as sql_load, delete_table as sql_delete_table
from musicmining.util.exceptions import ParameterError
from sqlalchemy.exc import OperationalError


@pytest.fixture
def test_table() -> pd.DataFrame:
    return pd.DataFrame({
        "A": np.random.random(20),
        "B": np.random.random(20),
        "C": np.random.random(20)
    })


@pytest.mark.parametrize("dir_exists", [True, False])
def test_save(dir_exists, test_table, tmp_path):
    if dir_exists:
        sql_save(path=":memory:", table="test", data=test_table)
        sql_save(path=f"{tmp_path}/test.db", table="test", data=test_table)
        with pytest.raises(ValueError):
            sql_save(path=f"{tmp_path}/test.db", table="test", data=test_table)
        sql_save(path=f"{tmp_path}/test.db", table="test", if_exists='replace', data=test_table)

    else:
        with pytest.raises(ParameterError):
            sql_save(path=f"{tmp_path}/does/not/exist.db", table="test", data=test_table)


def test_load(tmp_path, test_table):
    db_path = f"{tmp_path}/test.db"
    test_table_name = "test"

    sql_save(path=db_path, table=test_table_name, data=test_table, if_exists="replace")
    _ = sql_load(path=db_path, table=test_table_name)

    with pytest.raises(ParameterError):
        _ = sql_load(path=f"{tmp_path}/does/not/exist.db", table=test_table_name)

    with pytest.raises(ValueError):
        _ = sql_load(path=db_path, table="does_not_exist")


def test_delete_table(tmp_path, test_table):
    db_path = f"{tmp_path}/test.db"
    test_table_name = "test"
    def set_up_test():
        sql_save(path=db_path, table=test_table_name, data=test_table, if_exists="replace")

    set_up_test()
    sql_delete_table(path=db_path, table=test_table_name)

    set_up_test()
    with pytest.raises(ParameterError):
        sql_delete_table(path=f"{tmp_path}/does/not/exist.db", table=test_table_name)

    set_up_test()
    sql_delete_table(path=db_path, table="does_not_exist", if_not_exists='succeed')

    set_up_test()
    with pytest.raises(OperationalError):
        sql_delete_table(path=db_path, table="does_not_exist", if_not_exists='fail')
