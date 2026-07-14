"""Легкий шар абстракції LLM-провайдерів. Це НЕ фреймворк.

Один вхід:
    complete(system, user, cfg) -> dict   # структурований JSON від моделі

Диспатч за cfg["provider"]:
    groq | together | openrouter | ollama | google_ai_studio
        -> OpenAI-сумісний POST {base_url}/chat/completions через httpx
    anthropic
        -> офіційний SDK; системний промпт кешується (prompt caching);
           для масових прогонів є complete_batch() з Batch API

Правила:
    * Модель НІКОЛИ не хардкодиться — лише cfg["model"] з config.yaml.
    * 429/5xx/таймаут -> експоненційний backoff; після вичерпання ретраїв —
      опціональний fallback на cfg["fallback"] (другий провайдер).
    * Відповідь парситься безпечно: знімаються ```json-обгортки; при
      невалідному JSON — один коригувальний ретрай.
    * Якщо в cfg задано ціни (price_in_per_mtok / price_out_per_mtok) —
      у лог пишеться оцінка вартості виклику.

Приклад cfg (шматок config.yaml):
    provider: groq
    model: llama-3.3-70b-versatile
    base_url: https://api.groq.com/openai/v1
    api_key_env: GROQ_API_KEY
    # необов'язково:
    temperature: 0.0
    max_tokens: 2048
    max_retries: 4
    fallback: {provider: ..., model: ..., ...}
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

OPENAI_COMPATIBLE = {"groq", "together", "openrouter", "ollama", "google_ai_studio"}

# Скільки чекати перед N-м ретраєм (секунди, плюс джитер)
_BACKOFF_BASE = 2.0
_BACKOFF_MAX = 60.0
# Стеля для Retry-After від провайдера: якщо просять чекати довше (вичерпано
# ДЕННИЙ ліміт) — чесно падаємо з поясненням, а не висимо годинами
_RETRY_AFTER_CAP = 120.0


class LLMError(Exception):
    """Помилка виклику провайдера після всіх ретраїв та fallback."""


# ---------------------------------------------------------------------------
# Публічний інтерфейс
# ---------------------------------------------------------------------------

def complete(system: str, user: str, cfg: dict) -> dict:
    """Викликає LLM і повертає розпарсений JSON-об'єкт.

    Кидає LLMError, якщо і основний провайдер, і fallback вичерпані.
    """
    try:
        return _complete_one(system, user, cfg)
    except LLMError:
        fallback = cfg.get("fallback")
        if not fallback:
            raise
        log.warning(
            "Провайдер %s/%s вичерпано, перемикаюсь на fallback %s/%s",
            cfg.get("provider"), cfg.get("model"),
            fallback.get("provider"), fallback.get("model"),
        )
        return _complete_one(system, user, fallback)


def complete_batch(items: list[tuple[str, str]], cfg: dict) -> list[dict]:
    """Масовий прогін списку (system, user)-пар.

    Для provider=anthropic використовує Batch API (−50% вартості, асинхронно);
    для решти провайдерів — послідовні виклики complete() (безкоштовні тири
    все одно обмежені RPM, паралелити нема сенсу).
    """
    if cfg.get("provider") == "anthropic":
        return _anthropic_batch(items, cfg)
    return [complete(system, user, cfg) for system, user in items]


# ---------------------------------------------------------------------------
# Внутрішнє: один провайдер, з ретраями та коригувальним JSON-ретраєм
# ---------------------------------------------------------------------------

def _complete_one(system: str, user: str, cfg: dict) -> dict:
    raw = _call_with_backoff(system, user, cfg)
    try:
        return _parse_json(raw)
    except ValueError:
        # Один коригувальний ретрай: просимо модель повернути чистий JSON
        log.warning("Невалідний JSON від %s/%s, коригувальний ретрай",
                    cfg.get("provider"), cfg.get("model"))
        fix_user = (
            f"{user}\n\nТвоя попередня відповідь не була валідним JSON:\n{raw[:2000]}\n\n"
            "Поверни ЛИШЕ валідний JSON-об'єкт, без пояснень і без markdown."
        )
        raw = _call_with_backoff(system, fix_user, cfg)
        try:
            return _parse_json(raw)
        except ValueError as exc:
            raise LLMError(f"Невалідний JSON після ретраю: {raw[:500]}") from exc


def _call_with_backoff(system: str, user: str, cfg: dict) -> str:
    provider = cfg.get("provider")
    max_retries = int(cfg.get("max_retries", 4))
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt:
            delay = min(_BACKOFF_BASE * (2 ** (attempt - 1)), _BACKOFF_MAX)
            delay += random.uniform(0, delay / 2)  # джитер проти синхронних ретраїв
            # Якщо сервер прислав Retry-After — поважаємо його, але з межею:
            # величезний Retry-After означає вичерпаний денний ліміт
            retry_after = getattr(last_exc, "retry_after", None)
            if retry_after:
                if retry_after > _RETRY_AFTER_CAP:
                    raise LLMError(
                        f"{provider}/{cfg.get('model')}: провайдер просить чекати "
                        f"{retry_after:.0f} с — схоже, вичерпано денний ліміт. "
                        "Спробуйте пізніше або налаштуйте fallback у config.yaml.")
                delay = max(delay, retry_after)
            log.info("Ретрай %d/%d через %.1f с (%s)", attempt, max_retries, delay, last_exc)
            time.sleep(delay)
        try:
            if provider in OPENAI_COMPATIBLE:
                return _call_openai_compatible(system, user, cfg)
            if provider == "anthropic":
                return _call_anthropic(system, user, cfg)
            raise LLMError(f"Невідомий provider: {provider!r}")
        except _RetryableError as exc:
            last_exc = exc
            continue

    raise LLMError(f"{provider}/{cfg.get('model')}: вичерпано ретраї: {last_exc}")


class _RetryableError(Exception):
    """429 / 5xx / мережевий збій — можна повторити."""

    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


def _api_key(cfg: dict) -> str:
    env = cfg.get("api_key_env")
    if not env:
        return "not-needed"  # напр., локальний ollama
    key = os.environ.get(env, "")
    if not key and cfg.get("provider") != "ollama":
        raise LLMError(f"Змінна оточення {env} порожня — заповніть .env")
    return key or "not-needed"


# ---------------------------------------------------------------------------
# OpenAI-сумісний шлях (groq / together / openrouter / ollama / google_ai_studio)
# ---------------------------------------------------------------------------

def _call_openai_compatible(system: str, user: str, cfg: dict) -> str:
    base_url = cfg["base_url"].rstrip("/")
    payload: dict[str, Any] = {
        "model": cfg["model"],  # тільки з конфіга, ніколи не хардкод
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": cfg.get("temperature", 0.0),
    }
    if cfg.get("max_tokens"):
        payload["max_tokens"] = cfg["max_tokens"]
    # JSON-режим підтримують усі перелічені провайдери; вимикається force_json: false
    if cfg.get("force_json", True):
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {_api_key(cfg)}"},
            timeout=float(cfg.get("timeout", 120)),
        )
    except httpx.HTTPError as exc:
        raise _RetryableError(f"мережа: {exc}") from exc

    if resp.status_code == 429 or resp.status_code >= 500:
        retry_after = None
        if resp.headers.get("retry-after"):
            try:
                retry_after = float(resp.headers["retry-after"])
            except ValueError:
                pass
        raise _RetryableError(f"HTTP {resp.status_code}: {resp.text[:300]}", retry_after)
    if resp.status_code != 200:
        # 4xx (крім 429) — помилка конфігурації, ретраїти безглуздо
        raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    _log_cost(cfg, data.get("usage") or {})
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Неочікувана структура відповіді: {json.dumps(data)[:500]}") from exc


# ---------------------------------------------------------------------------
# Anthropic: SDK + prompt caching системного промпту
# ---------------------------------------------------------------------------

def _anthropic_client():
    try:
        import anthropic
    except ImportError as exc:
        raise LLMError(
            "provider=anthropic потребує: uv sync --extra anthropic"
        ) from exc
    return anthropic.Anthropic()  # ключ бере з ANTHROPIC_API_KEY


def _call_anthropic(system: str, user: str, cfg: dict) -> str:
    import anthropic

    client = _anthropic_client()
    try:
        msg = client.messages.create(
            model=cfg["model"],  # тільки з конфіга
            max_tokens=int(cfg.get("max_tokens", 2048)),
            temperature=cfg.get("temperature", 0.0),
            # cache_control: системний промпт (він однаковий для всіх claims)
            # кешується на стороні API — до −90% вартості вхідних токенів
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.RateLimitError as exc:
        raise _RetryableError(f"anthropic rate limit: {exc}") from exc
    except anthropic.APIStatusError as exc:
        if exc.status_code >= 500:
            raise _RetryableError(f"anthropic {exc.status_code}") from exc
        raise LLMError(f"anthropic {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise _RetryableError(f"anthropic мережа: {exc}") from exc

    usage = {
        "prompt_tokens": msg.usage.input_tokens,
        "completion_tokens": msg.usage.output_tokens,
    }
    _log_cost(cfg, usage)
    return "".join(block.text for block in msg.content if block.type == "text")


def _anthropic_batch(items: list[tuple[str, str]], cfg: dict) -> list[dict]:
    """Batch API: −50% вартості, результат протягом години (зазвичай хвилини)."""
    client = _anthropic_client()
    requests = [
        {
            "custom_id": f"item-{i}",
            "params": {
                "model": cfg["model"],
                "max_tokens": int(cfg.get("max_tokens", 2048)),
                "temperature": cfg.get("temperature", 0.0),
                "system": [{"type": "text", "text": system,
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": user}],
            },
        }
        for i, (system, user) in enumerate(items)
    ]
    batch = client.messages.batches.create(requests=requests)
    log.info("Anthropic batch %s: %d запитів, чекаю завершення…", batch.id, len(items))
    while batch.processing_status == "in_progress":
        time.sleep(15)
        batch = client.messages.batches.retrieve(batch.id)

    results: dict[str, dict] = {}
    for entry in client.messages.batches.results(batch.id):
        if entry.result.type == "succeeded":
            text = "".join(b.text for b in entry.result.message.content if b.type == "text")
            results[entry.custom_id] = _parse_json(text)
        else:
            log.error("Batch-елемент %s: %s", entry.custom_id, entry.result.type)
            results[entry.custom_id] = {"error": entry.result.type}
    return [results.get(f"item-{i}", {"error": "missing"}) for i in range(len(items))]


# ---------------------------------------------------------------------------
# Утиліти
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _parse_json(raw: str) -> dict:
    """Безпечний парсинг: знімає ```json-обгортки, вирізає перший {...}-блок."""
    text = _FENCE_RE.sub("", raw.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Модель могла додати текст навколо JSON — беремо від першої { до останньої }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Не вдалося розпарсити JSON: {raw[:300]}")


def _log_cost(cfg: dict, usage: dict) -> None:
    """Лог токенів; якщо в конфізі є ціни — лог оцінки вартості (для платних провайдерів)."""
    tin = usage.get("prompt_tokens", 0)
    tout = usage.get("completion_tokens", 0)
    if not (tin or tout):
        return
    price_in = cfg.get("price_in_per_mtok")
    price_out = cfg.get("price_out_per_mtok")
    if price_in is not None and price_out is not None:
        cost = tin / 1e6 * float(price_in) + tout / 1e6 * float(price_out)
        log.info("%s/%s: %d in / %d out токенів, ~$%.5f",
                 cfg.get("provider"), cfg.get("model"), tin, tout, cost)
    else:
        log.debug("%s/%s: %d in / %d out токенів",
                  cfg.get("provider"), cfg.get("model"), tin, tout)
