### {{{                          --     imports     --
import pandas as pd
from typing import List, Tuple

from biocomptools.toollib import common as cm
import logging
import psycopg2
from psycopg2.extras import execute_values
from typing import Optional, Union
from enum import Enum


##────────────────────────────────────────────────────────────────────────────}}}

dblog = logging.getLogger('biocomptools.database')
dblog.setLevel(logging.DEBUG)
config = cm.load_config()
DBType = Enum('DBType', 'sqlite postgres')
ListOrTuple = Union[List, Tuple]

### {{{                     --     db utils     --


def connect_postgres():
    try:
        conn = psycopg2.connect(
            dbname=config.db.name,
            user=config.db.user,
            password=config.db.password,
            host=config.db.host,
            port=config.db.port,
        )
    except Exception as e:
        dblog.error(f'Error connecting to database: {e}')
        raise e
    return conn



def connect_to_db(which_db: DBType = DBType.sqlite):
    # list or tuple type:
    if which_db == DBType.sqlite:
        engine = cm.get_biocompdb_sqlite_engine(config.db.sqlite.path)
        return engine
    elif which_db == DBType.postgres:
        return connect_postgres()
    else:
        raise ValueError(f'Invalid db type: {which_db}')



def execute_query(
    query: str,
    params: Optional[ListOrTuple] = None,
    conn: Optional[psycopg2.extensions.connection] = None,
    close_after=True,
):
    conn = conn or connect_to_db()
    cur = conn.cursor()
    try:
        cur.execute(query, params)
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        if close_after:
            conn.commit()
            cur.close()
            conn.close()

    return cur, conn


def execute_many(
    query: str,
    params: ListOrTuple,
    conn: Optional[psycopg2.extensions.connection] = None,
    dry_run: bool = False,
    close_after: bool = True,
):
    if dry_run:
        dblog.info(f'Dry run: {query} with {len(params)} values:')
        dblog.info(params)
        return
    conn = conn or connect_to_db()
    cur = conn.cursor()
    try:
        execute_values(cur, query, params)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        if close_after:
            cur.close()
            conn.close()
    return cur, conn


def query_to_df(
    query: str,
    params: Optional[ListOrTuple] = None,
    conn: Optional[psycopg2.extensions.connection] = None,
):
    conn = conn or connect_to_db()
    cur, _ = execute_query(query, params, conn, close_after=False)
    if cur.description is None:
        return pd.DataFrame()
    df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    cur.close()
    conn.close()
    return df


def get_row(
    query: str,
    params: Optional[ListOrTuple] = None,
    conn: Optional[psycopg2.extensions.connection] = None,
):
    df = query_to_df(query, params, conn)
    if len(df) == 0:
        return None
    return df.iloc[0].to_dict()


def get_row_by_id(table_name, key_column, id, conn=None):
    return get_row(f'SELECT * FROM {table_name} WHERE {key_column} = %s', (id,), conn)


def table_to_df(table_name, **kwargs):
    return query_to_df(f'SELECT * FROM {table_name}', **kwargs)


def convert_types_to_sql(df):
    # convert types to sql types
    df = df.copy()
    for col in df.columns:
        # posixpath to string
        if df[col].dtype == 'O':
            df[col] = df[col].astype(str)
        # datetime to string
        if df[col].dtype == 'datetime64[ns]':
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df


def update_table(
    df, table_name, key_column, conn=None, dry_run=False, columns=None, update_only=False
):

    df = convert_types_to_sql(df)
    assert key_column in df.columns

    if columns is not None:
        # check that all columns are present in the dataframe
        for col in columns:
            if col not in df.columns:
                raise ValueError(f'Column {col} not found in dataframe')
    else:
        columns = df.columns.to_list()

    columns = list(set(columns + [key_column]))

    if update_only:
        query = f"""
                UPDATE {table_name} SET {', '.join([f'{col} = v.{col}' for col in columns])}
                FROM (VALUES %s) v({', '.join(columns)})
                WHERE {table_name}.{key_column} = v.{key_column}
                """
    else:
        query = f"""
            INSERT INTO {table_name} ({', '.join(columns)})
            VALUES %s ON CONFLICT ({key_column})
            DO UPDATE SET ({', '.join(columns)}) = ({', '.join(['EXCLUDED.' + col for col in columns])})
            """

    execute_many(query, df[columns].values, conn, dry_run=dry_run)

    dblog.info(f'Updated {len(df)} rows in table {table_name}')


def insert_row_if_not_exists(table_name, row, key_column, conn=None):
    existing_row = get_row_by_id(table_name, key_column, row[key_column], conn)
    if existing_row is None:
        insert_row(table_name, row, conn)
    else:
        update_row(table_name, row, key_column, conn)


def insert_if_not_exists(table_name, df, key_column, conn=None):
    for _, row in df.iterrows():
        insert_row_if_not_exists(table_name, row, key_column, conn)


def insert_row(table_name, row, conn=None):
    columns = ', '.join(row.keys())
    values = ', '.join(['%s'] * len(row))
    query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
    # convert values types to sql types
    row = convert_types_to_sql(pd.DataFrame([row])).iloc[0].to_dict()
    execute_query(query, list(row.values()), conn)


def update_row(table_name, row, key_column, conn=None):
    columns = ', '.join(row.keys())
    values = ', '.join(['%s'] * len(row))
    query = f"UPDATE {table_name} SET ({columns}) = ({values}) WHERE {key_column} = %s"
    # convert values types to sql types
    row = convert_types_to_sql(pd.DataFrame([row])).iloc[0].to_dict()
    execute_query(query, list(row.values()) + [row[key_column]], conn)


##────────────────────────────────────────────────────────────────────────────}}}
