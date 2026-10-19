import psycopg
import os
from dotenv import load_dotenv


load_dotenv()

DB_NAME: str = os.environ.get("DB_NAME", None)
DB_USER: str = os.environ.get("DB_USER", None)
DB_PASS: str = os.environ.get("DB_PASS", None)
DB_HOST: str = os.environ.get("DB_HOST", None)


def db_client(sql_query: str = None, params: tuple = ()) -> None:
    with psycopg.connect(user=DB_USER, password=DB_PASS, host=DB_HOST, dbname=DB_NAME) as conn:
        cur = conn.cursor()
        cur.execute(sql_query, params=params)
        conn.commit()


def init() -> None:
    sql_query: str = """
        CREATE TABLE IF NOT EXISTS games (
            id        INT GENERATED ALWAYS AS IDENTITY,
            gameId    VARCHAR(255) NOT NULL,
            botColor  VARCHAR(255) NOT NULL DEFAULT 'white' CHECK (botColor IN ('white', 'black')),
            winner    VARCHAR(255) NOT NULL DEFAULT 'none' CHECK (winner IN ('white', 'black', 'none')),
            fen       TEXT NOT NULL,
            status    VARCHAR(255) NOT NULL
        );
    """
    db_client(sql_query)

init()
