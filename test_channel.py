"""ГОЛОВНИЙ ПЕРШИЙ РЕЗУЛЬТАТ: повний пайплайн валідації одного джерела.

    uv run python test_channel.py --source <tg_username|url> --days 3
    uv run python test_channel.py --gold [--days 3]

По кожному claim друкується: вердикт, confidence, знайдені еталонні фрагменти
з посиланнями та схожістю, reasoning судді. Все зберігається в БД
(claims + verdicts), тож повторний запуск пропускає вже оброблені пости.

--gold: проганяє обидва gold_sources з config.yaml (свідомо неякісне та якісне
джерела). Якщо в конфігу заповнено llm.judge_alt — кожен claim судять ДВА
судді, вердикти друкуються поруч для порівняння.

Пам'ятайте: no_data ≠ фейк. Оцінюйте картину загалом: у сміттєвого каналу
частка supported буде низькою, а contradicted/no_data — високою; в якісного —
навпаки.
"""
from __future__ import annotations

import argparse
import logging
import sys
from urllib.parse import urlparse

from newsguard import db, extractor, fetcher, judge, llm, retriever
from newsguard.config import load_config

log = logging.getLogger("test_channel")

# Скільки постів обробляти за запуск: повний пайплайн = 1 виклик екстрактора
# + по виклику судді на кожен claim, безкоштовний тир Groq обмежений RPM/RPD
DEFAULT_LIMIT = 10


def resolve_source(conn, arg: str) -> dict:
    """Перетворює аргумент --source (tg-хендл або URL) на рядок джерела в БД."""
    if arg.startswith("http") and "t.me/" not in arg:
        kind, identifier = "rss", arg
        name = urlparse(arg).netloc
    else:
        kind, identifier = "telegram", fetcher.normalize_telegram(arg)
        name = f"@{identifier}"
    source_id = db.upsert_source(
        conn, name=name, kind=kind, identifier=identifier,
        role="user", topic=["politics"], trust_note="додано через test_channel",
    )
    return {"id": source_id, "name": name, "kind": kind,
            "identifier": identifier, "role": "user", "topic": ["politics"]}


def pending_posts(conn, source_id: int, limit: int) -> list[dict]:
    """Пости джерела без claims — свіжі першими."""
    return conn.execute(
        """
        SELECT p.id, p.text, p.published_at, p.url
        FROM posts p
        LEFT JOIN claims c ON c.post_id = p.id
        WHERE p.source_id = %s AND c.id IS NULL
        ORDER BY p.published_at DESC
        LIMIT %s
        """,
        (source_id, limit),
    ).fetchall()


def judge_names(cfg: dict) -> list[tuple[str, dict]]:
    """Основний суддя + опціональний judge_alt для порівняння бік-о-бік."""
    judges = [("A", cfg["llm"]["judge"])]
    if cfg["llm"].get("judge_alt"):
        judges.append(("B", cfg["llm"]["judge_alt"]))
    return judges


def process_source(conn, cfg: dict, source: dict, days: int, limit: int) -> dict:
    """Повний пайплайн по одному джерелу. Повертає лічильники вердиктів судді A."""
    print(f"\n{'=' * 78}\nДЖЕРЕЛО: {source['name']}  [{source['kind']}:{source['identifier']}]")
    new, dup = fetcher.fetch_source(conn, source, days)
    print(f"Зібрано постів: нових {new}, вже було {dup}")

    posts = pending_posts(conn, source["id"], limit)
    if not posts:
        print("Немає необроблених постів (усі вже мають claims).")
        return {}
    print(f"Обробляю {len(posts)} постів (ліміт {limit}, свіжі першими)\n")

    judges = judge_names(cfg)
    counts: dict[str, int] = {"supported": 0, "contradicted": 0, "no_data": 0}

    for pi, post in enumerate(posts, 1):
        preview = post["text"][:120].replace("\n", " ")
        print(f"┌─ ПОСТ {pi}/{len(posts)}  {post['published_at']:%Y-%m-%d %H:%M} UTC")
        print(f"│  {preview}{'…' if len(post['text']) > 120 else ''}")
        print(f"│  {post['url']}")

        try:
            claims = extractor.extract_claims(
                post["text"], post["published_at"], cfg["llm"]["extractor"])
        except llm.LLMError as exc:
            print(f"└─ ЕКСТРАКТОР НЕДОСТУПНИЙ: {exc}\n   Зупиняюсь — далі немає сенсу.")
            break
        if not claims:
            print("└─ Перевірюваних тверджень не знайдено\n")
            continue

        for ci, claim_text in enumerate(claims, 1):
            claim_id = db.insert_claim(conn, post_id=post["id"], claim_text=claim_text)
            chunks = retriever.retrieve(conn, claim_text, post["published_at"], cfg)

            print(f"│\n│  CLAIM {ci}/{len(claims)}: {claim_text}")
            used: set[int] = set()
            for label, judge_cfg in judges:
                try:
                    v = judge.judge_claim(claim_text, chunks, judge_cfg)
                except llm.LLMError as exc:
                    print(f"│    СУДДЯ [{label}] НЕДОСТУПНИЙ: {exc}")
                    continue
                tag = f"[{label}: {judge_cfg['provider']}/{judge_cfg['model']}]" \
                    if len(judges) > 1 else ""
                print(f"│    ВЕРДИКТ: {v['verdict'].upper()}  "
                      f"(confidence {v['confidence']:.2f}) {tag}")
                print(f"│    Суддя: {v['reasoning']}")
                if label == "A":  # у БД пишемо вердикт основного судді
                    db.insert_verdict(conn, claim_id=claim_id, verdict=v["verdict"],
                                      confidence=v["confidence"],
                                      evidence=judge.build_evidence(v, chunks))
                    counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
                    used = set(v["used_chunk_ids"])

            if chunks:
                print("│    Фрагменти корпусу (топ-3 із "
                      f"{len(chunks)}, * = використані суддею):")
                for c in chunks[:3]:
                    mark = "*" if c["chunk_id"] in used else " "
                    frag = c["chunk_text"][:100].replace("\n", " ")
                    print(f"│     {mark} [{c['similarity']:.3f}] {c['source_name']}, "
                          f"{c['published_at']:%m-%d %H:%M}: {frag}…")
                    print(f"│         {c['url']}")
            else:
                print("│    Фрагментів у вікні ±48 год не знайдено "
                      "(no_data поставлено без виклику LLM)")
        print("└─\n")

    total = sum(counts.values()) or 1
    print(f"ПІДСУМОК {source['name']}: "
          + ",  ".join(f"{k}: {v} ({100 * v // total}%)" for k, v in counts.items()))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", help="tg-хендл, t.me-URL або URL RSS-фіда")
    group.add_argument("--gold", action="store_true",
                       help="прогнати обидва gold_sources з config.yaml")
    parser.add_argument("--days", type=int, default=3, help="глибина збору (типово 3)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"макс. постів на джерело за запуск (типово {DEFAULT_LIMIT})")
    parser.add_argument("--refresh-corpus", action="store_true",
                        help="перед оцінкою дозібрати еталонний корпус і довекторизувати "
                             "(інакше свіжі пости впираються у застарілий корпус -> хибні no_data)")
    args = parser.parse_args()

    cfg = load_config()
    with db.get_conn() as conn:
        db.apply_schema(conn)
        if args.refresh_corpus:
            from newsguard import embedder
            print("Оновлюю еталонний корпус…")
            for src in fetcher.sync_sources(conn, cfg, "sources"):
                try:
                    new, _dup = fetcher.fetch_source(conn, src, args.days)
                    if new:
                        print(f"  {src['name']}: +{new} нових постів")
                except Exception as exc:
                    log.warning("%s: %s", src["name"], exc)
            n = embedder.embed_new_posts(conn, cfg)
            print(f"Довекторизовано чанків: {n}\n")
        if args.gold:
            sources = fetcher.sync_sources(conn, cfg, "gold_sources")
            if not sources:
                sys.exit("У config.yaml не заповнені gold_sources")
            if len(judge_names(cfg)) > 1:
                print("Режим порівняння суддів: A = judge, B = judge_alt")
            results = {}
            for src in sources:
                results[src["name"]] = process_source(conn, cfg, src, args.days, args.limit)
            print(f"\n{'=' * 78}\nЗВЕДЕННЯ GOLD-ТЕСТУ")
            for name, counts in results.items():
                total = sum(counts.values()) or 1
                print(f"  {name:<28} supported {counts.get('supported', 0):>3} "
                      f"({100 * counts.get('supported', 0) // total}%)   "
                      f"contradicted {counts.get('contradicted', 0):>3}   "
                      f"no_data {counts.get('no_data', 0):>3}")
        else:
            source = resolve_source(conn, args.source)
            process_source(conn, cfg, source, args.days, args.limit)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # кирилиця в консолі Windows
    logging.basicConfig(level=logging.WARNING)
    main()
