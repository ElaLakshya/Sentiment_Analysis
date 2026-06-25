import sqlite3
from contextlib import contextmanager
from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS tweets (
    tweet_id        TEXT PRIMARY KEY,
    profile_handle  TEXT NOT NULL,
    text            TEXT,
    tweet_url       TEXT,
    created_at      TEXT,
    scraped_at      TEXT,
    processed       INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sentiment (
    tweet_id    TEXT PRIMARY KEY REFERENCES tweets(tweet_id),
    score       INTEGER CHECK(score IN (-1, 0, 1)),
    model       TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS entities (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tweet_id      TEXT REFERENCES tweets(tweet_id),
    entity_type   TEXT CHECK(entity_type IN ('person','organization','location','date')),
    entity_value  TEXT
);

CREATE TABLE IF NOT EXISTS taxonomy (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tweet_id        TEXT REFERENCES tweets(tweet_id),
    top_level       TEXT,
    sub_level       TEXT
);

CREATE INDEX IF NOT EXISTS idx_entities_tweet   ON entities(tweet_id);
CREATE INDEX IF NOT EXISTS idx_entities_type    ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_taxonomy_tweet   ON taxonomy(tweet_id);
CREATE INDEX IF NOT EXISTS idx_taxonomy_top     ON taxonomy(top_level);
CREATE INDEX IF NOT EXISTS idx_tweets_processed ON tweets(processed);
CREATE INDEX IF NOT EXISTS idx_tweets_handle    ON tweets(profile_handle);
"""

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)

def insert_tweet(data: dict):
    with get_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO tweets
               (tweet_id, profile_handle, text, tweet_url, created_at, scraped_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (data["tweet_id"], data["profile_handle"], data["text"],
             data["tweet_url"], data["created_at"], data["scraped_at"]),
        )

def get_unprocessed_tweets(limit: int = 100, profile_handle: str = None):
    """
    Returns unprocessed tweets. If profile_handle is given, only returns
    tweets from that profile — prevents old tweets from other profiles
    leaking into a new run.
    """
    with get_conn() as conn:
        if profile_handle:
            rows = conn.execute(
                """SELECT tweet_id, text, created_at FROM tweets
                   WHERE processed = 0 AND LOWER(profile_handle) = LOWER(?)
                   ORDER BY scraped_at LIMIT ?""",
                (profile_handle, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT tweet_id, text, created_at FROM tweets
                   WHERE processed = 0
                   ORDER BY scraped_at LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

def save_analysis(tweet_id: str, sentiment_score: int,
                  entities: dict, taxonomy_entries: list, model_name: str):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO sentiment (tweet_id, score, model)
               VALUES (?, ?, ?)
               ON CONFLICT(tweet_id) DO UPDATE SET
                   score=excluded.score, model=excluded.model,
                   created_at=datetime('now')""",
            (tweet_id, sentiment_score, model_name),
        )

        conn.execute("DELETE FROM entities WHERE tweet_id = ?", (tweet_id,))
        SINGULAR = {"persons": "person", "organizations": "organization",
                    "locations": "location", "dates": "date"}
        for key, singular in SINGULAR.items():
            for v in (entities.get(key) or []):
                conn.execute(
                    "INSERT INTO entities (tweet_id, entity_type, entity_value) VALUES (?, ?, ?)",
                    (tweet_id, singular, v),
                )

        conn.execute("DELETE FROM taxonomy WHERE tweet_id = ?", (tweet_id,))
        for t in taxonomy_entries:
            conn.execute(
                "INSERT INTO taxonomy (tweet_id, top_level, sub_level) VALUES (?, ?, ?)",
                (tweet_id, t.get("top_level"), t.get("sub_level")),
            )

        conn.execute("UPDATE tweets SET processed = 1 WHERE tweet_id = ?", (tweet_id,))

def reset_profile(profile_handle: str):
    """Mark all tweets from a profile as unprocessed so they get re-analyzed."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE tweets SET processed = 0 WHERE LOWER(profile_handle) = LOWER(?)",
            (profile_handle,)
        )
        print(f"Reset all tweets for @{profile_handle} to unprocessed.")

def stats(profile_handle: str = None):
    with get_conn() as conn:
        if profile_handle:
            total     = conn.execute("SELECT COUNT(*) c FROM tweets WHERE LOWER(profile_handle)=LOWER(?)", (profile_handle,)).fetchone()["c"]
            processed = conn.execute("SELECT COUNT(*) c FROM tweets WHERE processed=1 AND LOWER(profile_handle)=LOWER(?)", (profile_handle,)).fetchone()["c"]
        else:
            total     = conn.execute("SELECT COUNT(*) c FROM tweets").fetchone()["c"]
            processed = conn.execute("SELECT COUNT(*) c FROM tweets WHERE processed=1").fetchone()["c"]
        return {"total_tweets": total, "processed": processed, "pending": total - processed}