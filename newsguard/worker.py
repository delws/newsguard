"""Воркер регулярного оновлення еталонного корпусу.

Цикл: зібрати нові пости еталонних джерел -> довекторизувати -> (опційно)
перерахувати рейтинги user-джерел. БЕЗ викликів LLM: екстракція та суд над
користувацькими постами запускаються окремо (test_channel.py) — корпус же
має оновлюватися часто й безкоштовно, бо новини живуть годинами і застарілий
корпус дає хибні no_data.

Два режими:
    uv run python -m newsguard.worker           # нескінченний цикл зі сну
    uv run python -m newsguard.worker --once    # один цикл (для планувальника
                                                # Windows Task Scheduler / cron)

Налаштування — config.yaml, секція worker (усі поля опційні):
    worker:
      interval_minutes: 60   # пауза між циклами в режимі демона
      fetch_days: 2          # глибина дозбору (більше не треба: дедуп відсіє)
      rescore: true          # перераховувати channel_scores після оновлення
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone

from . import db, embedder, fetcher
from .config import load_config

log = logging.getLogger(__name__)

DEFAULTS = {"interval_minutes": 60, "fetch_days": 2, "rescore": True}


def run_cycle(cfg: dict) -> dict:
    """Один цикл оновлення. Повертає підсумок для логів."""
    w = {**DEFAULTS, **(cfg.get("worker") or {})}
    started = time.monotonic()
    summary = {"new_posts": 0, "new_chunks": 0, "errors": 0, "rescored": 0}

    # свіже підключення на кожен цикл: демон переживає рестарти Postgres
    with db.get_conn() as conn:
        db.apply_schema(conn)
        for src in fetcher.sync_sources(conn, cfg, "sources"):
            try:
                new, _dup = fetcher.fetch_source(conn, src, int(w["fetch_days"]))
                summary["new_posts"] += new
                if new:
                    log.info("%s: +%d постів", src["name"], new)
            except Exception as exc:  # одне мертве джерело не зриває цикл
                summary["errors"] += 1
                log.error("%s: %s", src["name"], exc)

        summary["new_chunks"] = embedder.embed_new_posts(conn, cfg)

        if w["rescore"]:
            from . import scorer

            user_sources = conn.execute(
                "SELECT id FROM sources WHERE role = 'user'").fetchall()
            for row in user_sources:
                scorer.score_source(conn, row["id"], scorer.DEFAULT_WINDOW_DAYS)
            summary["rescored"] = len(user_sources)

    summary["seconds"] = round(time.monotonic() - started, 1)
    log.info(
        "Цикл завершено за %.0f с: +%d постів, +%d чанків, "
        "перераховано рейтингів: %d, помилок джерел: %d",
        summary["seconds"], summary["new_posts"], summary["new_chunks"],
        summary["rescored"], summary["errors"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Оновлення еталонного корпусу")
    parser.add_argument("--once", action="store_true",
                        help="один цикл і вихід (для зовнішнього планувальника)")
    args = parser.parse_args()

    cfg = load_config()
    interval = 60 * int({**DEFAULTS, **(cfg.get("worker") or {})}["interval_minutes"])

    if args.once:
        run_cycle(cfg)
        return

    log.info("Демон запущено, інтервал %d хв. Ctrl+C для зупинки.", interval // 60)
    while True:
        try:
            run_cycle(cfg)
        except KeyboardInterrupt:
            raise
        except Exception:  # цикл, що впав (мережа, БД), не вбиває демона
            log.exception("Цикл впав, наступна спроба о %s",
                          datetime.now(timezone.utc).strftime("%H:%M UTC"))
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            log.info("Зупинено.")
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
