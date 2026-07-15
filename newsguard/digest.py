"""Markdown-дайджест за темою та період: сюжети довірених джерел + бейджі
надійності для перевірених користувацьких джерел.

Дизайн v1 — БЕЗ LLM (нуль вартості, детермінований результат):
  * «Головні сюжети» — кластеризація постів еталонного корпусу за близькістю
    векторів (жадібна, поріг косинусної схожості). Вага сюжету = скільки
    РІЗНИХ довірених джерел про нього пишуть: дайджест показує КОНСЕНСУС,
    а не гучність одного видання.
  * «Перевірені джерела» — свіжі claims користувацьких джерел з бейджами:
        [OK] supported / [X] contradicted / [?] no_data (НЕ фейк!)
    та рейтингом каналу з channel_scores.

CLI:
    uv run python -m newsguard.digest --topic politics --hours 24
    uv run python -m newsguard.digest --hours 24 --out digest.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

from . import db
from .scorer import interpret

log = logging.getLogger(__name__)

# Поріг "той самий сюжет" для e5-large (нормалізовані вектори, косинус).
# Воєнні зведення різних подій схожі між собою аж до ~0.87, тому поріг високий;
# порівнюємо з вектором ПЕРШОГО поста кластера, а не з центроїдом — центроїд
# дрейфує і зчіплює незв'язані новини у мега-кластери.
SAME_STORY_THRESHOLD = 0.885
MAX_STORIES = 8
BADGE = {"supported": "✅", "contradicted": "❌", "no_data": "⚪"}


def _headline(text: str, limit: int = 120) -> str:
    """Перший змістовний рядок: пропускаємо емодзі-рядки та короткі вигуки."""
    for line in text.strip().splitlines():
        line = line.strip()
        if len(line) >= 20 and any(ch.isalpha() for ch in line):
            return line[:limit] + ("…" if len(line) > limit else "")
    return text.strip()[:limit]


def cluster_stories(conn, topic: str | None, since: datetime) -> list[dict]:
    """Жадібна кластеризація еталонних постів за embedding першого чанка."""
    rows = conn.execute(
        """
        SELECT DISTINCT ON (p.id)
               p.id, p.text, p.url, p.published_at, s.name AS source_name,
               c.embedding
        FROM posts p
        JOIN sources s ON s.id = p.source_id AND s.role = 'reference'
        JOIN chunks c  ON c.post_id = p.id
        WHERE p.published_at >= %s
          AND (%s::text IS NULL OR %s = ANY(s.topic))
        ORDER BY p.id, c.id
        """,
        (since, topic, topic),
    ).fetchall()
    if not rows:
        return []

    clusters: list[dict] = []
    for row in rows:
        emb = row["embedding"]
        # pgvector може віддати numpy-масив або обгортку Vector — зводимо до numpy
        vec = np.asarray(emb.to_list() if hasattr(emb, "to_list") else emb,
                         dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) or 1.0)
        best, best_sim = None, SAME_STORY_THRESHOLD
        for cl in clusters:
            sim = float(vec @ cl["anchor"])  # якір, не центроїд: без дрейфу
            if sim >= best_sim:
                best, best_sim = cl, sim
        if best is None:
            clusters.append({"anchor": vec, "posts": [row]})
        else:
            best["posts"].append(row)

    # вага сюжету = кількість РІЗНИХ джерел (консенсус), потім розмір
    for cl in clusters:
        cl["sources"] = sorted({p["source_name"] for p in cl["posts"]})
        # заголовок — перший змістовний рядок якірного (найпершого) поста
        cl["title"] = _headline(cl["posts"][0]["text"])
    clusters.sort(key=lambda c: (len(c["sources"]), len(c["posts"])), reverse=True)
    return clusters[:MAX_STORIES]


def user_source_sections(conn, since: datetime) -> list[str]:
    """Секції по користувацьких джерелах: рейтинг + claims з бейджами."""
    sources = conn.execute(
        """
        SELECT s.id, s.name,
               (SELECT score FROM channel_scores cs
                WHERE cs.source_id = s.id ORDER BY cs.computed_at DESC LIMIT 1) AS score
        FROM sources s WHERE s.role = 'user' ORDER BY s.name
        """
    ).fetchall()
    out: list[str] = []
    for src in sources:
        rows = conn.execute(
            """
            SELECT DISTINCT ON (c.id)
                   c.claim_text, v.verdict, v.confidence,
                   v.evidence->'sources'->0->>'url'    AS ev_url,
                   v.evidence->'sources'->0->>'source' AS ev_source,
                   p.url AS post_url
            FROM claims c
            JOIN posts p    ON p.id = c.post_id AND p.source_id = %s
            JOIN verdicts v ON v.claim_id = c.id
            WHERE p.published_at >= %s
            ORDER BY c.id, v.judged_at DESC
            """,
            (src["id"], since),
        ).fetchall()
        if not rows:
            continue
        score_txt = f"{src['score']:.3f} — {interpret(src['score'])}" \
            if src["score"] is not None else "ще не обчислено"
        out.append(f"### {src['name']} · рейтинг {score_txt}\n")
        for r in rows:
            badge = BADGE.get(r["verdict"], "⚪")
            line = f"- {badge} `{r['verdict']}` ({r['confidence']:.2f}) {r['claim_text']}"
            if r["ev_url"]:
                line += f" — [{r['ev_source']}]({r['ev_url']})"
            line += f" · [пост]({r['post_url']})"
            out.append(line)
        out.append("")
    return out


def build_digest(conn, topic: str | None, hours: int) -> str:
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)
    lines: list[str] = [
        f"# NEWSGUARD дайджест{': ' + topic if topic else ''}",
        f"*Період: останні {hours} год (до {now:%Y-%m-%d %H:%M} UTC). "
        "Сюжети ранжовано за кількістю довірених джерел, що про них пишуть "
        "(консенсус, а не гучність).*",
        "",
        "## Головні сюжети довірених джерел",
        "",
    ]
    stories = cluster_stories(conn, topic, since)
    if not stories:
        lines.append("_Постів у корпусі за період немає — оновіть корпус "
                     "(python -m newsguard.fetcher)._")
    for i, st in enumerate(stories, 1):
        lines.append(f"### {i}. {st['title']}")
        lines.append(f"*{len(st['sources'])} джерел, {len(st['posts'])} матеріалів*")
        by_source: dict[str, str] = {}
        for p in st["posts"]:  # одне посилання на джерело — найсвіжіше
            if p["source_name"] not in by_source and p["url"]:
                by_source[p["source_name"]] = p["url"]
        lines.append(" · ".join(f"[{name}]({url})" for name, url in by_source.items()))
        lines.append("")

    user_sections = user_source_sections(conn, since)
    if user_sections:
        lines += ["## Перевірені користувацькі джерела", ""]
        lines += user_sections
        lines += [
            "---",
            "*Легенда: ✅ підтверджено корпусом довірених ЗМІ · ❌ суперечить "
            "корпусу · ⚪ немає даних у корпусі за ±48 год. **⚪ ≠ фейк**: "
            "ексклюзив чи регіональна новина може законно бути відсутньою "
            "в інших виданнях. Еталон — консенсус довірених джерел, "
            "а не істина в останній інстанції.*",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Markdown-дайджест за період")
    parser.add_argument("--topic", default=None, help="фільтр за темою (напр. politics)")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--out", default=None, help="файл; без нього — stdout")
    args = parser.parse_args()

    with db.get_conn() as conn:
        md = build_digest(conn, args.topic, args.hours)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"Дайджест записано: {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)
    main()
