"""Суддя: оцінка claim відносно фрагментів еталонного корпусу.

ПРИНЦИПОВІ ПРАВИЛА (зашиті в промпт і код):
  * no_data ≠ фейк. Ексклюзив чи регіональна новина може законно бути
    відсутньою в корпусі. За відсутності релевантних фрагментів суддя
    ЗОБОВ'ЯЗАНИЙ ставити no_data і НЕ додумувати.
  * Суддя порівнює claim ЛИШЕ з наданими фрагментами — власні знання моделі
    заборонені (вони застарілі й неперевірювані).
  * Еталон — не "істина", а консенсус довірених джерел на момент часу.

Якщо retriever не знайшов жодного фрагмента, вердикт no_data ставиться
БЕЗ виклику LLM — це економить токени і виключає галюцинації.
"""
from __future__ import annotations

import json
import logging

from . import llm

log = logging.getLogger(__name__)

VALID_VERDICTS = {"supported", "contradicted", "no_data"}

SYSTEM_PROMPT = """Ти — суддя фактичних тверджень. Тобі дають ТВЕРДЖЕННЯ з
новинного поста та ФРАГМЕНТИ з корпусу довірених українських ЗМІ, опубліковані
у вікні ±48 годин від поста.

Твоє завдання — винести вердикт, спираючись ВИКЛЮЧНО на надані фрагменти:

- "supported": фрагменти ПРЯМО підтверджують суть твердження. Дрібні
  розбіжності у формулюваннях допустимі, суть — ні.
- "contradicted": фрагменти ПРЯМО суперечать твердженню (інші цифри, інший
  результат події, спростування).
- "no_data": фрагменти не стосуються твердження або їх недостатньо, щоб
  підтвердити чи спростувати.

ЗАЛІЗНІ ПРАВИЛА:
1. Використовуй ЛИШЕ надані фрагменти. Твої власні знання про світ —
   ЗАБОРОНЕНІ: вони можуть бути застарілими.
2. Якщо релевантних фрагментів немає — став "no_data". НЕ додумуй.
   no_data — це нормальний, чесний вердикт: ексклюзивна чи регіональна
   новина може законно бути відсутньою в корпусі. no_data НЕ означає фейк.
3. "contradicted" — лише при ПРЯМІЙ суперечності. Відсутність підтвердження
   суперечністю НЕ є.
4. Часткове підтвердження (підтверджена подія, але інша ключова цифра) —
   це "contradicted" щодо цифри, якщо розбіжність суттєва, інакше "supported"
   з нижчим confidence.
5. confidence — твоя впевненість у вердикті від 0 до 1.
6. У used_chunk_ids вкажи id лише тих фрагментів, на які реально спирався.

Відповідай ЛИШЕ валідним JSON:
{"verdict": "supported|contradicted|no_data",
 "confidence": 0.0,
 "reasoning": "коротке пояснення українською",
 "used_chunk_ids": [1, 2]}"""

NO_CHUNKS_VERDICT = {
    "verdict": "no_data",
    "confidence": 1.0,
    "reasoning": "У корпусі довірених джерел немає жодного фрагмента в часовому "
                 "вікні ±48 год — твердження неможливо ані підтвердити, ані "
                 "спростувати (це не означає, що воно хибне).",
    "used_chunk_ids": [],
}


def judge_claim(claim_text: str, chunks: list[dict], cfg: dict) -> dict:
    """Повертає {verdict, confidence, reasoning, used_chunk_ids}.

    chunks — результат retriever.retrieve(); cfg — блок llm.judge з config.yaml.
    """
    if not chunks:
        return dict(NO_CHUNKS_VERDICT)

    fragments = "\n\n".join(
        f"[chunk_id={c['chunk_id']}] {c['source_name']}, "
        f"{c['published_at']:%Y-%m-%d %H:%M} UTC:\n{c['chunk_text']}"
        for c in chunks
    )
    user = f"ТВЕРДЖЕННЯ:\n{claim_text}\n\nФРАГМЕНТИ КОРПУСУ:\n{fragments}"
    result = llm.complete(SYSTEM_PROMPT, user, cfg)
    return _validate(result, chunks)


def _validate(result: dict, chunks: list[dict]) -> dict:
    """Санітизація відповіді судді: невалідне не має потрапити в БД."""
    verdict = str(result.get("verdict", "")).strip().lower()
    if verdict not in VALID_VERDICTS:
        log.warning("Невалідний вердикт %r -> no_data", verdict)
        return {
            "verdict": "no_data",
            "confidence": 0.0,
            "reasoning": f"Суддя повернув невалідну структуру: {json.dumps(result, ensure_ascii=False)[:200]}",
            "used_chunk_ids": [],
        }
    try:
        confidence = min(1.0, max(0.0, float(result.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    known_ids = {c["chunk_id"] for c in chunks}
    used = [i for i in (result.get("used_chunk_ids") or [])
            if isinstance(i, int) and i in known_ids]  # галюциновані id відкидаємо
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": str(result.get("reasoning", "")).strip(),
        "used_chunk_ids": used,
    }


def build_evidence(verdict_data: dict, chunks: list[dict]) -> dict:
    """Формує evidence-jsonb для таблиці verdicts: reasoning + використані джерела."""
    by_id = {c["chunk_id"]: c for c in chunks}
    sources = [
        {
            "chunk_id": cid,
            "source": by_id[cid]["source_name"],
            "url": by_id[cid]["url"],
            "published_at": by_id[cid]["published_at"].isoformat(),
        }
        for cid in verdict_data["used_chunk_ids"] if cid in by_id
    ]
    return {
        "chunk_ids": verdict_data["used_chunk_ids"],
        "reasoning": verdict_data["reasoning"],
        "sources": sources,
        "retrieved_count": len(chunks),
    }
