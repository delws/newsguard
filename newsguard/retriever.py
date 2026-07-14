"""Пошук релевантних фрагментів еталонного корпусу для claim.

КРИТИЧНО: новини живуть годинами, тому пошук ЗАВЖДИ обмежений часовим
вікном ±window_hours (типово 48) від дати публікації поста, що перевіряється.
Без цього фільтра торішня схожа новина "підтвердить" будь-що.

Запит вектора йде з префіксом "query: " (вимога e5), корпус збережено
з префіксом "passage: ".
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from . import embedder

log = logging.getLogger(__name__)


def retrieve(
    conn,
    claim_text: str,
    post_published_at: datetime,
    cfg: dict,
) -> list[dict]:
    """Повертає top-k чанків корпусу в часовому вікні, за спаданням схожості.

    Кожен елемент: {chunk_id, chunk_text, published_at, url, source_name, similarity}.
    Порожній список — легітимний результат: суддя зобов'язаний дати no_data.
    """
    r_cfg = cfg["retrieval"]
    window = timedelta(hours=int(r_cfg["window_hours"]))
    top_k = int(r_cfg.get("top_k", 8))

    model = embedder.get_model(cfg)
    query_vec = model.encode(
        cfg["embedding"]["prefix_query"] + claim_text,
        normalize_embeddings=True,
    )

    # більше probes -> кращий recall ivfflat (типово 1); корпус малий, це дешево
    conn.execute("SET ivfflat.probes = 16")
    rows = conn.execute(
        """
        SELECT c.id            AS chunk_id,
               c.chunk_text,
               c.published_at,
               p.url,
               s.name          AS source_name,
               1 - (c.embedding <=> %(vec)s::vector) AS similarity
        FROM chunks c
        JOIN posts   p ON p.id = c.post_id
        JOIN sources s ON s.id = p.source_id AND s.role = 'reference'
        WHERE c.published_at BETWEEN %(t_from)s AND %(t_to)s   -- обов'язковий часовий фільтр
        ORDER BY c.embedding <=> %(vec)s::vector
        LIMIT %(k)s
        """,
        {
            "vec": list(map(float, query_vec)),
            "t_from": post_published_at - window,
            "t_to": post_published_at + window,
            "k": top_k,
        },
    ).fetchall()

    log.debug("claim %r: знайдено %d чанків у вікні ±%s год",
              claim_text[:60], len(rows), r_cfg["window_hours"])
    return rows
