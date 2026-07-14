"""Агрегація вердиктів у рейтинг джерела (channel_scores).

ПРИНЦИП: рейтингується не "канал загалом" одним промптом, а СТАТИСТИКА
вердиктів по атомарних твердженнях за ковзне вікно. no_data ≠ фейк, тому
непідтверджені твердження НЕ прирівнюються до спростованих — але канал,
який пише переважно неперевірюване, не може мати високий рейтинг.

Формула v1 (прозора евристика, підлягає калібруванню на реальних даних):

    accuracy      = (supported + 1) / (supported + contradicted + 2)
                    # Лапласове згладжування: мало даних -> ближче до 0.5
    verifiability = (supported + contradicted) / total
                    # яка частка тверджень взагалі перетинається з корпусом
    score         = accuracy * sqrt(verifiability)          # 0..1

  * contradicted б'є по accuracy напряму;
  * море no_data тягне вниз через verifiability, але м'яко (корінь);
  * канал без жодного вердикта score не отримує (NULL, а не 0).

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


def verdict_counts(conn, source_id: int, window_start: datetime,
                   window_end: datetime) -> dict[str, int]:
    """Лічильники вердиктів по постах джерела у вікні.

    Якщо claim судили кілька разів (переоцінка) — береться ОСТАННІЙ вердикт.
    """
    row = conn.execute(
        """
        SELECT count(*) FILTER (WHERE lv.verdict = 'supported')    AS supported,
               count(*) FILTER (WHERE lv.verdict = 'contradicted') AS contradicted,
               count(*) FILTER (WHERE lv.verdict = 'no_data')      AS no_data
        FROM (
            SELECT DISTINCT ON (c.id) c.id, v.verdict
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
    return {k: row[k] or 0 for k in ("supported", "contradicted", "no_data")}


def compute_score(counts: dict[str, int]) -> float | None:
    """Формула v1 (див. docstring модуля). None, якщо вердиктів немає."""
    sup, con, nod = counts["supported"], counts["contradicted"], counts["no_data"]
    total = sup + con + nod
    if total == 0:
        return None
    accuracy = (sup + 1) / (sup + con + 2)
    verifiability = (sup + con) / total
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
