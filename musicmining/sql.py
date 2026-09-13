import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np


def save(
        *, 
        path: str, 
        table: str,
        data: pd.DataFrame, 
        **kwargs):
    engine = create_engine(f'sqlite:///{path}')

    with engine.connect() as conn:
        data.to_sql(table, conn, **kwargs)


def load(*, path: str, table: str, **kwargs) -> pd.DataFrame:
    engine = create_engine(f'sqlite:///{path}')
    with engine.connect() as conn:
        data = pd.read_sql_table(table, conn, **kwargs)

    return data


def delete_table(*, path: str, table: str):
    engine = create_engine(f'sqlite:///{path}')

    with engine.connect() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS {table};'))
        conn.commit()
