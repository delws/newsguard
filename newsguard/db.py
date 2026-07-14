"""Робота з БД: підключення, застосування схеми, upsert-хелпери.

Використання:
    python -m newsguard.db   # застосувати schema.sql до БД з DATABASE_URL
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"


def get_conn() -> psycopg.Connection:
    """Підключення до PostgreSQL. Рядки повертаються як dict."""
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL не задано — скопіюйте .env.example у .env")
    conn = psycopg.connect(dsn, row_factory=dict_row)
    _try_register_vector(conn)
    return conn


def _try_register_vector(conn: psycopg.Connection) -> None:
    """Реєструє тип vector для psycopg. До apply_schema() розширення ще нема — це не помилка."""
    try:
        from pgvector.psycopg import register_vector

        register_vector(conn)
    except psycopg.ProgrammingError:
        conn.rollback()
        log.info("Розширення vector ще не встановлено; спершу виконайте apply_schema()")


def apply_schema(conn: psycopg.Connection) -> None:
    """Ідемпотентно застосовує schema.sql (усі оператори — IF NOT EXISTS)."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.execute(sql)
    conn.commit()
    _try_register_vector(conn)
    log.info("Схему застосовано: %s", SCHEMA_PATH)


# ---------------------------------------------------------------------------
# Upsert-хелпери
# ---------------------------------------------------------------------------

def upsert_source(
    conn: psycopg.Connection,
    *,
    name: str,
    kind: str,
    identifier: str,
    role: str,
    topic: list[str] | None = None,
    trust_note: str | None = None,
    added_by: str | None = None,
) -> int:
    """Створює або оновлює джерело за ключем (kind, identifier). Повертає id."""
    row = conn.execute(
        """
        INSERT INTO sources (name, kind, identifier, role, topic, trust_note, added_by)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (kind, identifier) DO UPDATE
            SET name = EXCLUDED.name,
                role = EXCLUDED.role,
                topic = EXCLUDED.topic,
                trust_note = EXCLUDED.trust_note
        RETURNING id
        """,
        (name, kind, identifier, role, topic or [], trust_note, added_by),
    ).fetchone()
    conn.commit()
    return row["id"]


def upsert_post(
    conn: psycopg.Connection,
    *,
    source_id: int,
    external_id: str,
    published_at: Any,
    text: str,
    url: str | None = None,
    topic: list[str] | None = None,
) -> int | None:
    """Вставляє пост; дублікат за (source_id, external_id) пропускається.

    Повертає id нового поста або None, якщо пост уже був (тоді його НЕ треба
    повторно чанкувати/ембедити).
    """
    row = conn.execute(
        """
        INSERT INTO posts (source_id, external_id, published_at, text, url, topic)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (source_id, external_id) DO NOTHING
        RETURNING id
        """,
        (source_id, external_id, published_at, text, url, topic or []),
    ).fetchone()
    conn.commit()
    return row["id"] if row else None


def insert_chunks(conn: psycopg.Connection, chunks: list[dict]) -> None:
    """Пакетна вставка чанків: [{post_id, chunk_text, published_at, embedding}, ...]."""
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (post_id, chunk_text, published_at, embedding)
            VALUES (%(post_id)s, %(chunk_text)s, %(published_at)s, %(embedding)s)
            """,
            chunks,
        )
    conn.commit()


def insert_claim(conn: psycopg.Connection, *, post_id: int, claim_text: str) -> int:
    row = conn.execute(
        "INSERT INTO claims (post_id, claim_text) VALUES (%s, %s) RETURNING id",
        (post_id, claim_text),
    ).fetchone()
    conn.commit()
    return row["id"]


def insert_verdict(
    conn: psycopg.Connection,
    *,
    claim_id: int,
    verdict: str,
    confidence: float,
    evidence: dict,
) -> int:
    """evidence: {chunk_ids: [...], reasoning: "...", sources: [{name, url}, ...]}."""
    row = conn.execute(
        """
        INSERT INTO verdicts (claim_id, verdict, confidence, evidence)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (claim_id, verdict, confidence, Jsonb(evidence)),
    ).fetchone()
    conn.commit()
    return row["id"]


def upsert_channel_score(
    conn: psycopg.Connection,
    *,
    source_id: int,
    window_start: Any,
    window_end: Any,
    supported_cnt: int,
    contradicted_cnt: int,
    no_data_cnt: int,
    score: float | None,
) -> int:
    """Перерахунок рейтингу за вікно перезаписує попередній результат того ж вікна."""
    row = conn.execute(
        """
        INSERT INTO channel_scores
            (source_id, window_start, window_end,
             supported_cnt, contradicted_cnt, no_data_cnt, score)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (source_id, window_start, window_end) DO UPDATE
            SET supported_cnt = EXCLUDED.supported_cnt,
                contradicted_cnt = EXCLUDED.contradicted_cnt,
                no_data_cnt = EXCLUDED.no_data_cnt,
                score = EXCLUDED.score,
                computed_at = now()
        RETURNING id
        """,
        (source_id, window_start, window_end,
         supported_cnt, contradicted_cnt, no_data_cnt, score),
    ).fetchone()
    conn.commit()
    return row["id"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with get_conn() as _conn:
        apply_schema(_conn)
    print("OK: схему застосовано")
