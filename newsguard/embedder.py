"""Чанкування та векторизація еталонного корпусу.

Модель: intfloat/multilingual-e5-large (MIT), ЛОКАЛЬНО на CPU.
ОБОВ'ЯЗКОВІ префікси e5 (без них якість пошуку падає):
    "passage: " — для чанків корпусу (тут),
    "query: "   — для claim при пошуку (retriever.py).

Чанкування: за абзацами, жадібне пакування до ~chunk_target_tokens (типово 500);
занадто довгий абзац ріжеться навпіл рекурсивно. Ліміт e5 — 512 токенів,
надлишок модель мовчки обрізає, тому тримаємось нижче.

Векторизуються ЛИШЕ пости еталонних джерел (role='reference'):
користувацькі пости не є доказовою базою — з них витягуються claims.

CLI:
    uv run python -m newsguard.embedder            # довекторизувати нове + статистика
    uv run python -m newsguard.embedder --stats    # лише статистика корпусу
"""
from __future__ import annotations

import argparse
import logging

from . import db
from .config import load_config

log = logging.getLogger(__name__)

_model = None  # ліниве завантаження: ~2.2 ГБ ваг, вантажимо один раз на процес


def get_model(cfg: dict):
    """Повертає SentenceTransformer з config.embedding.model (нічого не хардкодимо)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # важкий імпорт — усередині

        name = cfg["embedding"]["model"]
        log.info("Завантажую модель %s (перший запуск качає ваги з HuggingFace)…", name)
        _model = SentenceTransformer(name, device="cpu")
    return _model


def chunk_text(text: str, tokenizer, target_tokens: int = 500) -> list[str]:
    """Ділить текст на чанки ~target_tokens, не розриваючи абзаци без потреби."""
    def tok_len(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    def split_long(par: str) -> list[str]:
        # абзац, що сам більший за ліміт, ріжемо навпіл за словами рекурсивно
        if tok_len(par) <= target_tokens:
            return [par]
        words = par.split()
        mid = len(words) // 2
        return split_long(" ".join(words[:mid])) + split_long(" ".join(words[mid:]))

    paragraphs: list[str] = []
    for par in (p.strip() for p in text.split("\n\n")):
        if par:
            paragraphs.extend(split_long(par))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for par in paragraphs:
        plen = tok_len(par)
        if current and current_len + plen > target_tokens:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(par)
        current_len += plen
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def embed_new_posts(conn, cfg: dict, batch_size: int = 16) -> int:
    """Чанкує та векторизує еталонні пости, які ще не мають чанків. Повертає к-ть чанків."""
    rows = conn.execute(
        """
        SELECT p.id, p.text, p.published_at
        FROM posts p
        JOIN sources s ON s.id = p.source_id AND s.role = 'reference'
        LEFT JOIN chunks c ON c.post_id = p.id
        WHERE c.id IS NULL
        ORDER BY p.id
        """
    ).fetchall()
    if not rows:
        log.info("Нових еталонних постів немає")
        return 0

    emb_cfg = cfg["embedding"]
    model = get_model(cfg)
    tokenizer = model.tokenizer
    target = int(emb_cfg.get("chunk_target_tokens", 500))
    prefix = emb_cfg["prefix_passage"]

    # 1) чанкуємо все
    pending: list[dict] = []
    for row in rows:
        for chunk in chunk_text(row["text"], tokenizer, target):
            pending.append({
                "post_id": row["id"],
                "chunk_text": chunk,
                "published_at": row["published_at"],
            })
    log.info("Постів: %d, чанків до векторизації: %d", len(rows), len(pending))

    # 2) векторизуємо батчами; normalize -> косинусна відстань = внутрішній добуток
    done = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        vectors = model.encode(
            [prefix + c["chunk_text"] for c in batch],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for c, v in zip(batch, vectors):
            c["embedding"] = v
        db.insert_chunks(conn, batch)  # комміт після кожного батча: збій не втрачає прогрес
        done += len(batch)
        if done % (batch_size * 10) == 0 or done == len(pending):
            log.info("Векторизовано %d/%d чанків", done, len(pending))
    return done


def print_stats(conn) -> None:
    """Статистика корпусу: джерела/пости/чанки та покриття за датами."""
    rows = conn.execute(
        """
        SELECT s.name, s.kind,
               COUNT(DISTINCT p.id)                  AS posts,
               COUNT(c.id)                           AS chunks,
               MIN(p.published_at)::date             AS date_from,
               MAX(p.published_at)::date             AS date_to
        FROM sources s
        LEFT JOIN posts p  ON p.source_id = s.id
        LEFT JOIN chunks c ON c.post_id = p.id
        WHERE s.role = 'reference'
        GROUP BY s.id, s.name, s.kind
        ORDER BY posts DESC
        """
    ).fetchall()
    total = conn.execute(
        """
        SELECT COUNT(DISTINCT s.id) AS sources,
               COUNT(DISTINCT p.id) AS posts,
               COUNT(c.id)          AS chunks
        FROM sources s
        LEFT JOIN posts p  ON p.source_id = s.id
        LEFT JOIN chunks c ON c.post_id = p.id
        WHERE s.role = 'reference'
        """
    ).fetchone()

    print(f"\n{'Джерело':<24} {'тип':<9} {'постів':>7} {'чанків':>7}  покриття дат")
    print("-" * 78)
    for r in rows:
        dates = f"{r['date_from']} … {r['date_to']}" if r["date_from"] else "—"
        print(f"{r['name']:<24} {r['kind']:<9} {r['posts']:>7} {r['chunks']:>7}  {dates}")
    print("-" * 78)
    print(f"Разом: {total['sources']} джерел, {total['posts']} постів, {total['chunks']} чанків")


def main() -> None:
    parser = argparse.ArgumentParser(description="Векторизація еталонного корпусу")
    parser.add_argument("--stats", action="store_true", help="лише статистика, без векторизації")
    args = parser.parse_args()

    cfg = load_config()
    with db.get_conn() as conn:
        if not args.stats:
            embed_new_posts(conn, cfg)
        print_stats(conn)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
