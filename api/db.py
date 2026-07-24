from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterable

import psycopg2
import psycopg2.extras


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "replidub"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}


def get_connection():
    return psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


@contextmanager
def get_cursor(commit: bool = False):
    conn = get_connection()
    cur = conn.cursor()
    try:
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def fetch_all(query_text: str, params: Iterable[Any] | None = None) -> list[dict]:
    with get_cursor(commit=False) as cur:
        cur.execute(query_text, params)
        return list(cur.fetchall())


def fetch_one(query_text: str, params: Iterable[Any] | None = None) -> dict | None:
    with get_cursor(commit=False) as cur:
        cur.execute(query_text, params)
        row = cur.fetchone()
        return dict(row) if row else None


def execute(query_text: str, params: Iterable[Any] | None = None) -> int:
    """
    For INSERT / UPDATE / DELETE when you do not need returned row data.
    Returns affected row count.
    """
    with get_cursor(commit=True) as cur:
        cur.execute(query_text, params)
        return cur.rowcount


def execute_returning(query_text: str, params: Iterable[Any] | None = None) -> dict | None:
    """
    For INSERT/UPDATE ... RETURNING ...
    Returns first returned row or None.
    """
    with get_cursor(commit=True) as cur:
        cur.execute(query_text, params)
        row = cur.fetchone()
        return dict(row) if row else None