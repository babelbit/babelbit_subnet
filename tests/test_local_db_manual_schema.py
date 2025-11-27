"""Integration tests against the local Postgres schema used by docker-compose-local-db.

These tests spin up a disposable Postgres container, apply the full
`manual_db_bb_schema.sql`, and verify that our write helpers populate the
staging tables in a shape that the local stored procedures expect.
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, UTC

import asyncpg
import pytest
import pytest_asyncio

from babelbit.utils.db_pool import (
    db_pool,
    insert_challenge_staging,
    insert_scoring_staging,
)

try:  # pragma: no cover
    from testcontainers.postgres import PostgresContainer
except Exception:  # pragma: no cover
    PostgresContainer = None  # type: ignore


SKIP_REASON = "Local DB tests need Docker + testcontainers; set NO_CONTAINER_TESTS=1 or USE_EXTERNAL_PG to skip"


def _should_start_container() -> bool:
    """Return True if we should attempt to start a Postgres testcontainer."""
    val = os.getenv("NO_CONTAINER_TESTS")
    if val is not None and val.strip() and val.strip().lower() not in ("0", "false", "no", "off"):
        return False
    return PostgresContainer is not None


@pytest.fixture(scope="session")
def manual_schema_sql() -> str:
    return Path("manual_db_bb_schema.sql").read_text()


@pytest.fixture(scope="session")
def local_pg_container():
    # Allow callers to provide an external Postgres instead of starting Docker
    if os.getenv("USE_EXTERNAL_PG") is not None:
        yield None
        return

    if not _should_start_container():
        pytest.skip(SKIP_REASON)
    # Use the same credentials as docker-compose-local-db to avoid CREATE DATABASE inside tests
    try:
        container = PostgresContainer(
            image="postgres:18-alpine",
            username="babelbit",
            password="babelbit",
            dbname="babelbit",
        )
    except Exception as exc:  # pragma: no cover - constructor failure (e.g., docker not reachable)
        pytest.skip(f"{SKIP_REASON}: {exc}")
        return

    try:
        container.start()
    except Exception as exc:  # pragma: no cover - setup failure path (e.g., no Docker socket)
        pytest.skip(f"{SKIP_REASON}: {exc}")
        return

    try:
        yield container
    finally:
        container.stop()


async def _apply_manual_schema(container: PostgresContainer, schema_sql: str) -> dict[str, str | int]:
    """Apply the checked-in schema inside the container using its default credentials."""
    host = container.get_container_host_ip()
    port = int(container.get_exposed_port(container.port))

    conn = await asyncpg.connect(
        host=host,
        port=port,
        user=container.username,
        password=container.password,
        database=container.dbname,
    )
    await conn.execute(schema_sql)
    await conn.close()

    return {
        "PG_HOST": host,
        "PG_PORT": port,
        "PG_DB": container.dbname,
        "PG_USER": container.username,
        "PG_PASSWORD": container.password,
    }


@pytest_asyncio.fixture(scope="session")
async def local_db_config(local_pg_container, manual_schema_sql):
    # External Postgres path (no Docker socket available)
    if local_pg_container is None:
        required = ["PG_HOST", "PG_PORT", "PG_DB", "PG_USER", "PG_PASSWORD"]
        if not all(os.getenv(k) for k in required):
            pytest.skip("USE_EXTERNAL_PG set but PG_* env vars incomplete; skipping local DB manual schema tests")
        try:
            conn = await asyncpg.connect(
                host=os.getenv("PG_HOST"),
                port=int(os.getenv("PG_PORT")),
                user=os.getenv("PG_USER"),
                password=os.getenv("PG_PASSWORD"),
                database=os.getenv("PG_DB"),
            )
            await conn.execute(manual_schema_sql)
            await conn.close()
        except Exception as exc:
            pytest.skip(f"External Postgres not reachable or schema apply failed: {exc}")
        return {k: os.getenv(k) for k in required}

    # Default path: start disposable container
    try:
        return await _apply_manual_schema(local_pg_container, manual_schema_sql)
    except Exception as exc:
        pytest.skip(f"Failed to apply manual schema in testcontainer: {exc}")


@pytest_asyncio.fixture
async def local_db(monkeypatch, local_db_config):
    # Ensure pool picks up local credentials
    for key, value in local_db_config.items():
        monkeypatch.setenv(key, str(value))

    # Restart pool against the local schema
    try:
        await db_pool.close()
    except RuntimeError as exc:
        if "Event loop is closed" in str(exc):
            pytest.skip("Event loop closed; skipping local DB manual schema tests in this environment")
        raise
    await db_pool.init(force=True)

    # Keep tables clean between tests
    await db_pool.execute(
        "TRUNCATE TABLE public.scoring_submissions, public.scoring_staging, public.challenges, public.challenge_staging RESTART IDENTITY CASCADE;"
    )

    yield

    try:
        await db_pool.close()
    except RuntimeError:
        # If the loop was torn down (e.g., in certain containerized runners), just ignore
        pass


@pytest.mark.asyncio
async def test_process_scoring_staging_populates_submissions(local_db):
    ts = datetime.now(UTC).replace(tzinfo=None)
    file_content = {
        "challenge_uid": "chal-local-1",
        "dialogue_uid": "dlg-1",
        "miner_uid": 42,
        "miner_hotkey": "local_hotkey",
        "dialogue_summary": {"average_U_best_early": 0.771},
        "utterances": [
            {"utterance_number": 0, "ground_truth": "hi", "best_step": 3, "U_best": 0.91, "total_steps": 5},
            {"utterance_number": 1, "ground_truth": "bye", "best_step": 2, "U_best": 0.82, "total_steps": 4},
        ],
    }

    staging_id = await insert_scoring_staging(
        file_content=file_content,
        file_path="/tmp/scoring.json",
        json_created_at=ts,
    )

    async with db_pool.connection() as conn:
        await conn.execute("CALL public.process_scoring_staging(NULL)")
        submissions = await conn.fetch(
            """
            SELECT scoring_staging_id, challenge_uid, dialogue_uid, miner_uid, miner_hotkey,
                   utterance_number, ground_truth, best_step, u_best, total_steps,
                   average_u_best_early, json_created_at, staging_inserted_at
            FROM public.scoring_submissions
            WHERE scoring_staging_id = $1
            ORDER BY utterance_number
            """,
            staging_id,
        )
        staging_row = await conn.fetchrow(
            "SELECT json_created_at, staging_inserted_at FROM public.scoring_staging WHERE id=$1",
            staging_id,
        )

    assert len(submissions) == 2, "Stored procedure should emit one row per utterance"
    assert {row["utterance_number"] for row in submissions} == {0, 1}
    assert {row["ground_truth"] for row in submissions} == {"hi", "bye"}
    assert all(row["challenge_uid"] == "chal-local-1" for row in submissions)
    assert all(row["dialogue_uid"] == "dlg-1" for row in submissions)
    assert all(row["miner_uid"] == 42 for row in submissions)
    assert all(row["miner_hotkey"] == "local_hotkey" for row in submissions)
    assert all(float(row["u_best"]) == pytest.approx(expected) for row, expected in zip(submissions, (0.91, 0.82)))
    assert all(
        float(row["average_u_best_early"]) == pytest.approx(0.771) for row in submissions
    ), "Average should be propagated from dialogue_summary"

    assert staging_row is not None
    assert all(row["json_created_at"] == ts for row in submissions)
    assert all(row["staging_inserted_at"] == staging_row["staging_inserted_at"] for row in submissions)


@pytest.mark.asyncio
async def test_process_challenge_staging_populates_challenges(local_db):
    ts = datetime.now(UTC).replace(tzinfo=None)
    file_content = {
        "challenge_uid": "chal-local-2",
        "dialogues": [
            {
                "dialogue_uid": "dlg-2",
                "language": "en",
                "utterances": ["hello world", "goodbye"],
            }
        ],
    }

    staging_id = await insert_challenge_staging(
        file_content=file_content,
        file_path="/tmp/challenge.json",
        json_created_at=ts,
    )

    async with db_pool.connection() as conn:
        await conn.execute("CALL public.process_challenge_staging(NULL)")
        challenges = await conn.fetch(
            """
            SELECT staging_id, challenge_uid, dialogue_uid, language,
                   utterance_number, utterance_text, json_created_at, staging_inserted_at
            FROM public.challenges
            WHERE staging_id = $1
            ORDER BY utterance_number
            """,
            staging_id,
        )
        staging_row = await conn.fetchrow(
            "SELECT json_created_at, staging_inserted_at FROM public.challenge_staging WHERE id=$1",
            staging_id,
        )

    assert len(challenges) == 2, "Procedure should explode dialogues into per-utterance rows"
    assert [row["utterance_number"] for row in challenges] == [0, 1]
    assert [row["utterance_text"] for row in challenges] == ["hello world", "goodbye"]
    assert all(row["language"] == "en" for row in challenges)
    assert all(row["challenge_uid"] == "chal-local-2" for row in challenges)
    assert all(row["dialogue_uid"] == "dlg-2" for row in challenges)

    assert staging_row is not None
    assert all(row["json_created_at"] == ts for row in challenges)
    assert all(row["staging_inserted_at"] == staging_row["staging_inserted_at"] for row in challenges)
