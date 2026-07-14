-- NEWSGUARD: схема БД
-- PostgreSQL 16 + pgvector. Розмірність embedding = 1024 (intfloat/multilingual-e5-large).
-- Схема мультитенантна наперед: sources.added_by (поки NULL — один користувач).

CREATE EXTENSION IF NOT EXISTS vector;

-- Джерела: і еталонні (role='reference', білий список ІМІ), і користувацькі (role='user')
CREATE TABLE IF NOT EXISTS sources (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT        NOT NULL,
    kind        TEXT        NOT NULL CHECK (kind IN ('rss', 'telegram', 'web')),
    identifier  TEXT        NOT NULL,  -- URL фіда / telegram-хендл / URL сайту
    role        TEXT        NOT NULL CHECK (role IN ('reference', 'user')),
    topic       TEXT[]      NOT NULL DEFAULT '{}',
    trust_note  TEXT,                  -- звідки довіра (напр. "білий список ІМІ, 2 півріччя 2025")
    added_by    TEXT,                  -- мультитенантність; поки NULL
    added_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (kind, identifier)
);

-- Пости з джерел; дедуп за (source_id, external_id)
CREATE TABLE IF NOT EXISTS posts (
    id           BIGSERIAL PRIMARY KEY,
    source_id    BIGINT      NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    external_id  TEXT        NOT NULL,  -- guid з RSS / id telegram-повідомлення / URL
    published_at TIMESTAMPTZ NOT NULL,
    text         TEXT        NOT NULL,
    url          TEXT,
    topic        TEXT[]      NOT NULL DEFAULT '{}',
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_id, external_id)
);

-- Чанки еталонного корпусу з векторами.
-- published_at денормалізовано з posts, бо КОЖЕН векторний пошук фільтрується за часом (±48 год).
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    post_id      BIGINT       NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    chunk_text   TEXT         NOT NULL,
    published_at TIMESTAMPTZ  NOT NULL,
    embedding    vector(1024) NOT NULL
);

-- Атомарні фактичні твердження, витягнуті з користувацьких постів
CREATE TABLE IF NOT EXISTS claims (
    id           BIGSERIAL PRIMARY KEY,
    post_id      BIGINT      NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    claim_text   TEXT        NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Вердикти судді. ВАЖЛИВО: no_data — легітимний результат (відсутність у корпусі ≠ фейк).
CREATE TABLE IF NOT EXISTS verdicts (
    id         BIGSERIAL PRIMARY KEY,
    claim_id   BIGINT      NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    verdict    TEXT        NOT NULL CHECK (verdict IN ('supported', 'contradicted', 'no_data')),
    confidence REAL        NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    evidence   JSONB       NOT NULL DEFAULT '{}',  -- {chunk_ids: [...], reasoning: "...", sources: [...]}
    judged_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Рейтинг джерела = агрегат вердиктів за ковзне вікно, НЕ оцінка "каналу в цілому" одним промптом
CREATE TABLE IF NOT EXISTS channel_scores (
    id               BIGSERIAL PRIMARY KEY,
    source_id        BIGINT      NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    window_start     TIMESTAMPTZ NOT NULL,
    window_end       TIMESTAMPTZ NOT NULL,
    supported_cnt    INT         NOT NULL DEFAULT 0,
    contradicted_cnt INT         NOT NULL DEFAULT 0,
    no_data_cnt      INT         NOT NULL DEFAULT 0,
    score            REAL,
    computed_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_id, window_start, window_end)
);

-- Індекси
-- ivfflat: наближений пошук за косинусною відстанню; lists=100 — розумний старт для <1М рядків.
-- Індекс ефективний після наповнення; за потреби перебудувати: REINDEX INDEX chunks_embedding_idx;
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS chunks_published_at_idx ON chunks (published_at);
CREATE INDEX IF NOT EXISTS posts_published_at_idx  ON posts (published_at);
CREATE INDEX IF NOT EXISTS posts_topic_idx         ON posts USING gin (topic);
CREATE INDEX IF NOT EXISTS sources_topic_idx       ON sources USING gin (topic);
CREATE INDEX IF NOT EXISTS claims_post_id_idx      ON claims (post_id);
CREATE INDEX IF NOT EXISTS verdicts_claim_id_idx   ON verdicts (claim_id);
