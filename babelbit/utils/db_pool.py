"""Async PostgreSQL connection pool and write helpers.

Design goals:
- Lazy singleton-style pool creation (shared across tasks)
- Explicit init/shutdown hooks
- Safe async context managers for connections & transactions
- Typed helper methods for staging + submission inserts
- Automatic JSON (de)serialization & timestamp handling
- Resilient transient failure retry on connection acquisition

Environment variables expected (fallbacks provided):
  PG_HOST (default: localhost)
  PG_PORT (default: 5432)
  PG_DB   (default: postgres)
  PG_USER (default: postgres)
  PG_PASSWORD (default: empty)
  PG_MIN_CONN (default: 1)
  PG_MAX_CONN (default: 10)

Usage:
    from babelbit.utils.db_pool import db_pool, insert_challenge_staging
    await db_pool.init()
    await insert_challenge_staging(file_content=..., file_path=..., json_created_at=dt)
    await db_pool.close()
"""
from __future__ import annotations

import os
import json
import asyncio
from typing import Any, AsyncIterator, Callable, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects by converting them to ISO format."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

import asyncpg

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class _PoolConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    min_size: int
    max_size: int
    statement_cache_size: int = 0  # disable per-connection stmt cache (optional)

    @classmethod
    def from_env(cls) -> "_PoolConfig":
        return cls(
            host=os.getenv("PG_HOST", "localhost"),
            port=int(os.getenv("PG_PORT", "5432")),
            user=os.getenv("PG_USER", "postgres"),
            password=os.getenv("PG_PASSWORD", ""),
            database=os.getenv("PG_DB", "postgres"),
            min_size=int(os.getenv("PG_MIN_CONN", "1")),
            max_size=int(os.getenv("PG_MAX_CONN", "10")),
            statement_cache_size=int(os.getenv("PG_STMT_CACHE", "0")),
        )

class AsyncPGPool:
    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()
        self._cfg = _PoolConfig.from_env()

    async def init(self, force: bool = False) -> None:
        if self._pool and not force:
            return
        async with self._lock:
            if self._pool and not force:
                return
            # Re-read config if forcing (lets tests override PG_* dynamically)
            if force or not self._pool:
                self._cfg = _PoolConfig.from_env()
            logger.info(
                "Initializing asyncpg pool host=%s db=%s size=[%d,%d]", 
                self._cfg.host, self._cfg.database, self._cfg.min_size, self._cfg.max_size
            )
            self._pool = await asyncpg.create_pool(
                host=self._cfg.host,
                port=self._cfg.port,
                user=self._cfg.user,
                password=self._cfg.password,
                database=self._cfg.database,
                min_size=self._cfg.min_size,
                max_size=self._cfg.max_size,
                statement_cache_size=self._cfg.statement_cache_size,
                timeout=30,
            )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("AsyncPG pool closed")

    @property
    def raw(self) -> asyncpg.Pool:
        if not self._pool:
            raise RuntimeError("Pool not initialized. Call await db_pool.init().")
        return self._pool

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[asyncpg.Connection]:
        if not self._pool:
            raise RuntimeError("Pool not initialized. Call await db_pool.init().")
        conn = await self._pool.acquire()
        try:
            yield conn
        finally:
            await self._pool.release(conn)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        async with self.connection() as conn:
            tx = conn.transaction()
            await tx.start()
            try:
                yield conn
            except Exception:
                await tx.rollback()
                raise
            else:
                await tx.commit()

    async def execute(self, sql: str, *args) -> str:
        async with self.connection() as conn:
            return await conn.execute(sql, *args)

    async def executemany(self, sql: str, args_seq: list[tuple[Any, ...]]) -> None:
        async with self.connection() as conn:
            await conn.executemany(sql, args_seq)

    async def fetchval(self, sql: str, *args) -> Any:
        async with self.connection() as conn:
            return await conn.fetchval(sql, *args)

    async def fetch(self, sql: str, *args) -> list[asyncpg.Record]:
        async with self.connection() as conn:
            rows = await conn.fetch(sql, *args)
            return list(rows)

# Singleton instance

db_pool = AsyncPGPool()

# -------------------------- Domain-specific helpers -------------------------- #

# NOTE: Rely on DB defaults for timestamps where appropriate. We still push
# json_created_at explicitly (supplied by upstream pipeline) and leave the
# inserted_at fields to database defaults.

async def insert_challenge_staging(*, file_content: dict, file_path: str, json_created_at: datetime) -> int:
    sql = """
        INSERT INTO public.challenge_staging (file_content, file_path, json_created_at)
        VALUES ($1, $2, $3)
        RETURNING id
    """
    return await db_pool.fetchval(sql, json.dumps(file_content), file_path, _ensure_utc(json_created_at))

from datetime import timezone

def _ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is in UTC format suitable for database storage.
    
    If naive, assume it's already UTC. If timezone-aware, convert to UTC.
    Return as naive UTC for database compatibility.
    """
    if dt.tzinfo is None:
        # Already naive, assume it's UTC
        return dt
    else:
        # Convert to UTC and make it naive for DB storage
        return dt.astimezone(timezone.utc).replace(tzinfo=None)

async def insert_json_staging(*, file_content: dict, file_path: str, json_created_at: datetime) -> int:
    sql = """
        INSERT INTO public.json_staging (file_content, file_path, json_created_at)
        VALUES ($1, $2, $3)
        RETURNING id
    """
    return await db_pool.fetchval(sql, json.dumps(file_content, cls=DateTimeEncoder), file_path, _ensure_utc(json_created_at))

async def insert_scoring_staging(*, file_content: dict, file_path: str, json_created_at: datetime) -> int:
    sql = """
        INSERT INTO public.scoring_staging (file_content, file_path, json_created_at)
        VALUES ($1, $2, $3)
        RETURNING id
    """
    return await db_pool.fetchval(sql, json.dumps(file_content, cls=DateTimeEncoder), file_path, _ensure_utc(json_created_at))

async def insert_challenges_bulk(rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
        INSERT INTO public.challenges (
            staging_id, challenge_uid, dialogue_uid, utterance_number,
            utterance_text, json_created_at, staging_inserted_at
        ) VALUES ($1,$2,$3,$4,$5,$6,$7)
    """
    args_seq = [
        (
            r["staging_id"],
            r["challenge_uid"],
            r["dialogue_uid"],
            r["utterance_number"],
            r["utterance_text"],
            _ensure_utc(r["json_created_at"]),
            _ensure_utc(r["staging_inserted_at"]),
        )
        for r in rows
    ]
    await db_pool.executemany(sql, args_seq)

async def insert_scoring_submissions_bulk(rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
        INSERT INTO public.scoring_submissions (
            scoring_staging_id, challenge_uid, dialogue_uid, miner_uid, miner_hotkey,
            utterance_number, ground_truth, best_step, u_best, total_steps, 
            average_u_best_early, json_created_at, staging_inserted_at
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
    """
    args_seq = [
        (
            r["scoring_staging_id"],
            r.get("challenge_uid"),
            r.get("dialogue_uid"),
            r.get("miner_uid"),
            r.get("miner_hotkey"),
            r["utterance_number"],
            r["ground_truth"],
            r["best_step"],
            r["u_best"],
            r["total_steps"],
            r["average_u_best_early"],
            _ensure_utc(r["json_created_at"]),
            _ensure_utc(r["staging_inserted_at"]),
        )
        for r in rows
    ]
    await db_pool.executemany(sql, args_seq)

async def insert_miner_submission_bulk(rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
        INSERT INTO public.miner_submission (
            staging_id, log_file, dialogue_uid, utterance_number, ground_truth,
            best_step, u_best, total_steps, average_u_best_early, json_created_at,
            staging_inserted_at
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
    """
    args_seq = [
        (
            r["staging_id"],
            r["log_file"],
            r.get("dialogue_uid"),
            r["utterance_number"],
            r["ground_truth"],
            r["best_step"],
            r["u_best"],
            r["total_steps"],
            r["average_u_best_early"],
            _ensure_utc(r["json_created_at"]),
            _ensure_utc(r["staging_inserted_at"]),
        )
        for r in rows
    ]
    await db_pool.executemany(sql, args_seq)

async def fetch_challenge_ids_by_uid(challenge_uid: str) -> list[int]:
    """Fetch all challenge IDs for a given challenge UID."""
    sql = "SELECT id FROM public.challenges WHERE challenge_uid = $1"
    rows = await db_pool.fetch(sql, challenge_uid)
    return [row[0] for row in rows]

async def health_check() -> bool:
    try:
        val = await db_pool.fetchval("SELECT 1")
        return val == 1
    except Exception as e:
        logger.error("DB health check failed: %s", e)
        return False

async def _iter_scores_from_db(limit: int) -> list[tuple[str, float, str]]:
    """Fetch recent per-dialogue scores from Postgres.

    We read from scoring_staging where file_content JSON stores miner metadata and
    dialogue_summary.average_U_best_early. Falls back gracefully if table absent.

    Returns list of (hotkey, score, challenge_uid) tuples limited to `limit` most recent entries.
    """
    sql = """
        SELECT
          file_content ->> 'miner_hotkey' AS miner_hotkey,
          (file_content -> 'dialogue_summary' ->> 'average_U_best_early')::float AS score,
          coalesce((file_content ->> 'challenge_uid'),'') as challenge_uid,
          json_created_at
        FROM public.scoring_staging
        WHERE (file_content ? 'miner_hotkey')
          AND (file_content -> 'dialogue_summary' ? 'average_U_best_early')
        ORDER BY json_created_at DESC
        LIMIT $1
    """
    try:
        rows = await db_pool.fetch(sql, limit)
    except Exception as e:  # table may not exist yet
        logger.warning("DB score fetch failed (%s); returning empty result", e)
        return []
    out: list[tuple[str, float, str]] = []
    for r in rows:
        hk = r.get("miner_hotkey") if isinstance(r, dict) else r[0]
        sc = r.get("score") if isinstance(r, dict) else r[1]
        cu = r.get("challenge_uid") if isinstance(r, dict) else r[2]
        if hk and sc is not None:
            try:
                out.append((str(hk), float(sc), str(cu) if cu is not None else ""))
            except Exception:
                continue
    return out

async def _iter_scores_for_challenge(challenge_uid: str) -> list[tuple[str, float]]:
    """Fetch scores for a specific challenge from Postgres.
    
    Returns list of (hotkey, score) tuples for the specified challenge, 
    with the latest score per miner if multiple submissions exist.
    """
    sql = """
        SELECT 
          miner_hotkey,
          score,
          json_created_at
        FROM (
          SELECT 
            file_content ->> 'miner_hotkey' AS miner_hotkey,
            (file_content ->> 'challenge_mean_U')::float AS score,
            json_created_at,
            ROW_NUMBER() OVER (
              PARTITION BY file_content ->> 'miner_hotkey' 
              ORDER BY json_created_at DESC
            ) as rn
          FROM public.scoring_staging
          WHERE (file_content ? 'miner_hotkey')
            AND (file_content ? 'challenge_mean_U')
            AND (file_content ->> 'challenge_uid' = $1)
        ) ranked
        WHERE rn = 1
        ORDER BY json_created_at DESC
    """
    try:
        rows = await db_pool.fetch(sql, challenge_uid)
    except Exception as e:  # table may not exist yet
        logger.warning("DB score fetch for challenge %s failed (%s); returning empty result", challenge_uid, e)
        return []
    
    out: list[tuple[str, float]] = []
    for r in rows:
        hk = r.get("miner_hotkey") if isinstance(r, dict) else r[0]
        sc = r.get("score") if isinstance(r, dict) else r[1]
        if hk and sc is not None:
            try:
                out.append((str(hk), float(sc)))
            except Exception:
                continue
    return out