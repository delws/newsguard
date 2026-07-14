"""Збір постів із джерел: RSS (feedparser), Telegram-канали (скрейп t.me/s/).

Нормалізація:
    * HTML знімається, пробіли стискаються;
    * пости коротші за MIN_POST_CHARS відкидаються (заголовки-заглушки, емодзі-пости);
    * репости в Telegram відкидаються (валідуємо власні твердження джерела).

Дедуп — на рівні БД: UNIQUE(source_id, external_id), upsert_post повертає None
для дубліката, і такий пост не потрапляє на повторне чанкування.

CLI:
    uv run python -m newsguard.fetcher --days 7            # еталонні джерела
    uv run python -m newsguard.fetcher --days 3 --role user
"""
from __future__ import annotations

import argparse
import logging
import random
import re
import time
from datetime import datetime, timedelta, timezone

import feedparser
import httpx
from bs4 import BeautifulSoup

from . import db
from .config import load_config

log = logging.getLogger(__name__)

MIN_POST_CHARS = 50
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
# Скільки сторінок t.me/s/ гортати максимум (захист від нескінченної пагінації)
TG_MAX_PAGES = 40


def _clean_text(html_or_text: str) -> str:
    """Знімає HTML-теги, стискає пробіли, зберігає розбиття на абзаци."""
    soup = BeautifulSoup(html_or_text, "html.parser")
    text = soup.get_text("\n")
    # стискаємо пробіли всередині рядків, але лишаємо порожні рядки між абзацами
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        if ln:
            out.append(ln)
        elif out and out[-1] != "":
            out.append("")
    return "\n".join(out).strip()


def _http_get(url: str, *, retries: int = 4) -> httpx.Response:
    """GET з експоненційним backoff на мережеві збої, 5xx та 429 (FloodWait)."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            delay = min(2.0 * 2 ** (attempt - 1), 60.0) + random.uniform(0, 1)
            log.info("Ретрай %d для %s через %.1f с", attempt, url, delay)
            time.sleep(delay)
        try:
            resp = httpx.get(url, headers={"User-Agent": UA},
                             timeout=30, follow_redirects=True)
        except httpx.HTTPError as exc:
            last = exc
            continue
        if resp.status_code == 429:
            # FloodWait: поважаємо Retry-After, якщо він є
            wait = float(resp.headers.get("retry-after", 30))
            log.warning("429 від %s — чекаю %.0f с", url, wait)
            time.sleep(wait)
            last = RuntimeError("429")
            continue
        if resp.status_code >= 500:
            last = RuntimeError(f"HTTP {resp.status_code}")
            continue
        return resp
    raise RuntimeError(f"Не вдалося отримати {url}: {last}")


# ---------------------------------------------------------------------------
# RSS
# ---------------------------------------------------------------------------

def fetch_rss(feed_url: str, since: datetime) -> list[dict]:
    """Повертає нормалізовані пости з RSS-фіда, не старші за `since`."""
    resp = _http_get(feed_url)
    parsed = feedparser.parse(resp.content)
    posts: list[dict] = []
    for entry in parsed.entries:
        ts = entry.get("published_parsed") or entry.get("updated_parsed")
        if not ts:
            continue  # без дати пост непридатний: часовий фільтр — обов'язковий
        published_at = datetime(*ts[:6], tzinfo=timezone.utc)
        if published_at < since:
            continue
        title = _clean_text(entry.get("title", ""))
        summary = _clean_text(entry.get("summary", "") or entry.get("description", ""))
        # уникаємо дублювання, коли summary починається з заголовка
        text = title if summary.startswith(title[:60]) else f"{title}\n\n{summary}"
        text = text.strip()
        if len(text) < MIN_POST_CHARS:
            continue
        posts.append({
            "external_id": entry.get("id") or entry.get("link"),
            "published_at": published_at,
            "text": text,
            "url": entry.get("link"),
        })
    return posts


# ---------------------------------------------------------------------------
# Telegram: публічна веб-версія t.me/s/<username>, без API-ключів
# ---------------------------------------------------------------------------

def fetch_telegram(username: str, since: datetime) -> list[dict]:
    """Гортає t.me/s/<username> назад у часі до `since`. Репости пропускаються."""
    posts: list[dict] = []
    before: str | None = None
    for _page in range(TG_MAX_PAGES):
        url = f"https://t.me/s/{username}" + (f"?before={before}" if before else "")
        soup = BeautifulSoup(_http_get(url).text, "html.parser")
        messages = soup.select(".tgme_widget_message")
        if not messages:
            break
        reached_older = False
        page_min_id: int | None = None
        for msg in messages:
            data_post = msg.get("data-post", "")  # формат "channel/12345"
            msg_id = data_post.rsplit("/", 1)[-1]
            if msg_id.isdigit():
                page_min_id = min(page_min_id or int(msg_id), int(msg_id))
            time_tag = msg.select_one("time[datetime]")
            if not time_tag:
                continue
            published_at = datetime.fromisoformat(time_tag["datetime"])
            if published_at < since:
                reached_older = True
                continue
            if msg.select_one(".tgme_widget_message_forwarded_from"):
                continue  # репост — не власне твердження каналу
            text_node = msg.select_one(".tgme_widget_message_text")
            if not text_node:
                continue  # фото/відео без тексту
            text = _clean_text(text_node.get_text("\n"))
            if len(text) < MIN_POST_CHARS:
                continue
            posts.append({
                "external_id": msg_id,
                "published_at": published_at,
                "text": text,
                "url": f"https://t.me/{data_post}",
            })
        if reached_older or page_min_id is None or page_min_id <= 1:
            break
        before = str(page_min_id)
        time.sleep(1.5 + random.uniform(0, 1))  # ввічлива пауза проти FloodWait
    return posts


# ---------------------------------------------------------------------------
# Оркестрація
# ---------------------------------------------------------------------------

def fetch_source(conn, source_row: dict, days: int) -> tuple[int, int]:
    """Збирає пости одного джерела. Повертає (нових, дублікатів)."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    kind = source_row["kind"]
    if kind == "rss":
        posts = fetch_rss(source_row["identifier"], since)
    elif kind == "telegram":
        posts = fetch_telegram(source_row["identifier"], since)
    else:
        raise NotImplementedError(
            f"kind={kind!r} поки не підтримується (заплановано: скрейп сайтів)")
    new = dup = 0
    for p in posts:
        post_id = db.upsert_post(
            conn,
            source_id=source_row["id"],
            external_id=p["external_id"],
            published_at=p["published_at"],
            text=p["text"],
            url=p["url"],
            topic=source_row.get("topic") or [],
        )
        if post_id is None:
            dup += 1
        else:
            new += 1
    return new, dup


def sync_sources(conn, cfg: dict, section: str = "sources") -> list[dict]:
    """Upsert джерел із config.yaml у БД; пропускає TODO. Повертає рядки з id."""
    rows: list[dict] = []
    for s in cfg.get(section) or []:
        if str(s.get("identifier", "")).strip().upper().startswith("TODO"):
            log.warning("Пропускаю %s: identifier = TODO", s.get("name"))
            continue
        source_id = db.upsert_source(
            conn,
            name=s["name"], kind=s["kind"], identifier=s["identifier"],
            role=s["role"], topic=s.get("topic") or [],
            trust_note=s.get("trust_note"), added_by=s.get("added_by"),
        )
        rows.append({**s, "id": source_id})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Збір постів у корпус")
    parser.add_argument("--days", type=int, default=7,
                        help="глибина збору в днях (типово 7)")
    parser.add_argument("--role", default="reference", choices=["reference", "user"],
                        help="які джерела з config.yaml збирати")
    args = parser.parse_args()

    cfg = load_config()
    with db.get_conn() as conn:
        db.apply_schema(conn)
        sources = sync_sources(conn, cfg, "sources") + sync_sources(conn, cfg, "gold_sources")
        total_new = total_dup = 0
        for src in sources:
            if src["role"] != args.role:
                continue
            try:
                new, dup = fetch_source(conn, src, args.days)
            except Exception as exc:  # одне мертве джерело не зриває збір
                log.error("%s: %s", src["name"], exc)
                continue
            total_new += new
            total_dup += dup
            print(f"  {src['name']:<22} [{src['kind']}] нових: {new:4d}, дублікатів: {dup:4d}")
        print(f"\nРазом: нових {total_new}, дублікатів {total_dup}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
