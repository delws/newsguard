"""Агрегація вердиктів у рейтинг джерела (channel_scores).

ПРИНЦИП: рейтингується не "канал загалом" одним промптом, а СТАТИСТИКА
вердиктів по атомарних твердженнях за ковзне вікно. no_data ≠ фейк, тому
непідтверджені твердження НЕ прирівнюються до спростованих — але канал,
який пише переважно неперевірюване, не може мати високий рейтинг.

Формула v2 (прозора евристика, відкалібрована на gold-тесті 2026-07-15):

    Кожен вердикт враховується З ВАГОЮ confidence судді (невпевнений вердикт
    важить менше). Зважені суми: sup_w, con_w, nod_w; total_w = їх сума.

    accuracy      = (sup_w + 1) / (sup_w + CONTRADICTED_WEIGHT * con_w + 2)
                    # Лапласове згладжування: мало даних -> ближче до 0.5;
                    # доведена неправда коштує в 1.5 раза дорожче, ніж
                    # доведена правда допомагає
    verifiability = (sup_w + con_w) / total_w
                    # яка частка тверджень взагалі перетинається з корпусом
    score         = accuracy * sqrt(verifiability)          # 0..1

  * contradicted б'є по accuracy напряму, з підсиленою вагою;
  * море no_data тягне вниз через verifiability, але м'яко (корінь) —
    no_data ≠ фейк, ексклюзив легітимно відсутній у корпусі;
  * канал без жодного вердикта score не отримує (NULL, а не 0).

  Відомі межі (виявлено на gold-тесті): канал-репостер реальних новин
  набирає supported чужим контентом, а видання з ексклюзивами платить
  verifiability'ю за унікальність. Розділення надійне лише на історії
  від ~30 тверджень і разом із калібруванням промптів судді/екстрактора
  (правила 4а та обережні дати), яке прибирає артефактні contradicted.

Вікно рахується по published_at ПОСТА (не по даті оцінки): рейтинг відповідає
на питання "наскільки правдивим був канал у цей період".

CLI:
    uv run python -m newsguard.scorer                  # усі user-джерела
    uv run python -m newsguard.scorer --window-days 7
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import datetime, timedelta, timezone

from . import db

log = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 7
# Доведена суперечність шкодить сильніше, ніж підтвердження допомагає
CONTRADICTED_WEIGHT = 1.5


def verdict_counts(conn, source_id: int, window_start: datetime,
                   window_end: datetime) -> dict[str, float]:
    """Лічильники вердиктів по постах джерела у вікні: і штучні (для БД),
    і зважені за confidence (для формули).

    Якщо claim судили кілька разів (переоцінка) — береться ОСТАННІЙ вердикт.
    """
    row = conn.execute(
        """
        SELECT count(*) FILTER (WHERE lv.verdict = 'supported')    AS supported,
               count(*) FILTER (WHERE lv.verdict = 'contradicted') AS contradicted,
               count(*) FILTER (WHERE lv.verdict = 'no_data')      AS no_data,
               COALESCE(sum(lv.confidence) FILTER (WHERE lv.verdict = 'supported'), 0)    AS sup_w,
               COALESCE(sum(lv.confidence) FILTER (WHERE lv.verdict = 'contradicted'), 0) AS con_w,
               COALESCE(sum(lv.confidence) FILTER (WHERE lv.verdict = 'no_data'), 0)      AS nod_w
        FROM (
            SELECT DISTINCT ON (c.id) c.id, v.verdict, v.confidence
            FROM claims c
            JOIN posts p    ON p.id = c.post_id
            JOIN verdicts v ON v.claim_id = c.id
            WHERE p.source_id = %s
              AND p.published_at BETWEEN %s AND %s
            ORDER BY c.id, v.judged_at DESC
        ) lv
        """,
        (source_id, window_start, window_end),
    ).fetchone()
    return {k: row[k] or 0 for k in
            ("supported", "contradicted", "no_data", "sup_w", "con_w", "nod_w")}


def compute_score(counts: dict[str, float]) -> float | None:
    """Формула v2 (див. docstring модуля). None, якщо вердиктів немає."""
    sup_w, con_w, nod_w = counts["sup_w"], counts["con_w"], counts["nod_w"]
    total_w = sup_w + con_w + nod_w
    if total_w == 0:
        return None
    accuracy = (sup_w + 1) / (sup_w + CONTRADICTED_WEIGHT * con_w + 2)
    verifiability = (sup_w + con_w) / total_w
    return round(accuracy * math.sqrt(verifiability), 4)


def score_source(conn, source_id: int, window_days: int,
                 now: datetime | None = None) -> dict:
    """Рахує та зберігає рейтинг одного джерела. Повертає підсумок."""
    window_end = now or datetime.now(timezone.utc)
    window_start = window_end - timedelta(days=window_days)
    counts = verdict_counts(conn, source_id, window_start, window_end)
    score = compute_score(counts)
    db.upsert_channel_score(
        conn, source_id=source_id,
        window_start=window_start, window_end=window_end,
        supported_cnt=counts["supported"],
        contradicted_cnt=counts["contradicted"],
        no_data_cnt=counts["no_data"],
        score=score,
    )
    return {**counts, "score": score,
            "window_start": window_start, "window_end": window_end}


def interpret(score: float | None) -> str:
    """Людське прочитання рейтингу для звітів і дайджестів."""
    if score is None:
        return "немає даних"
    if score >= 0.65:
        return "високої довіри"
    if score >= 0.45:
        return "помірної довіри"
    if score >= 0.25:
        return "низької довіри"
    return "критично низької довіри"


def main() -> None:
    parser = argparse.ArgumentParser(description="Рейтинг user-джерел за вікно")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    args = parser.parse_args()

    with db.get_conn() as conn:
        sources = conn.execute(
            "SELECT id, name FROM sources WHERE role = 'user' ORDER BY name"
        ).fetchall()
        if not sources:
            sys.exit("Немає user-джерел — додайте через test_channel.py --source")

        print(f"\nРейтинг за останні {args.window_days} дн. "
              f"(валідовано тверджень / supported / contradicted / no_data)\n")
        print(f"{'Джерело':<30} {'sup':>4} {'con':>4} {'n/d':>4} {'усього':>7} "
              f"{'score':>7}  оцінка")
        print("-" * 78)
        for src in sources:
            r = score_source(conn, src["id"], args.window_days)
            total = r["supported"] + r["contradicted"] + r["no_data"]
            score_txt = f"{r['score']:.3f}" if r["score"] is not None else "  —  "
            print(f"{src['name']:<30} {r['supported']:>4} {r['contradicted']:>4} "
                  f"{r['no_data']:>4} {total:>7} {score_txt:>7}  {interpret(r['score'])}")
        print("-" * 78)
        print("score = accuracy * sqrt(verifiability); no_data не є фейком, але\n"
              "знижує перевірюваність. Рейтинг надійний від ~30 тверджень у вікні.")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)
    main()
