"""Functional tests for the async Postgres pool.

These tests use a disposable Postgres container (testcontainers) instead of an
"in-memory" database (Postgres has no true in-memory mode). To skip, set:

    export NO_CONTAINER_TESTS=1

or run with `-k 'not db_pool'` once a marker is added in future. Requires
Docker to be running locally.
"""

import os
from datetime import datetime, UTC
import pytest
import pytest_asyncio

from babelbit.utils.db_pool import (
    db_pool,
    insert_challenge_staging,
    insert_challenges_bulk,
    fetch_challenge_ids_by_uid,
    health_check,
)

try:  # pragma: no cover
    from testcontainers.postgres import PostgresContainer
except Exception:  # pragma: no cover
    PostgresContainer = None  # type: ignore

SKIP_REASON = "testcontainers not available / disabled / docker missing"


def _have_testcontainers() -> bool:
    """Return True if we should attempt to start a container.

    Rules:
      - NO_CONTAINER_TESTS (truthy: 1, true, yes, on) -> never start
        (If set to 0/false/empty treat as NOT requesting a skip)
      - FORCE_TESTCONTAINERS -> start even if PG_HOST/PG_DB already set
      - USE_EXTERNAL_PG -> do not start (user explicitly wants external DB)
      - Presence of PG_HOST/PG_DB alone NO LONGER suppresses container unless
        USE_EXTERNAL_PG is set (previous behavior caused unexpected skips).
    """
    val = os.getenv("NO_CONTAINER_TESTS")
    if val is not None:
        if val.strip() != "" and val.strip().lower() not in ("0", "false", "no", "off"):
            # Explicit request to disable container-based tests
            return False
    if PostgresContainer is None:
        return False
    if os.getenv("USE_EXTERNAL_PG") is not None:
        return False
    if os.getenv("FORCE_TESTCONTAINERS") is not None:
        return True
    return True


@pytest.fixture(scope="session")
def pg_container():
    """Provide a Postgres container or None.

    Modes:
      - USE_EXTERNAL_PG: do not start container (return None)
      - FORCE_TESTCONTAINERS: start container regardless of existing PG_* env
      - default: start container if available; if Docker fails -> skip
    """
    if os.getenv("USE_EXTERNAL_PG") is not None:
        print("[db_pool tests] Using external Postgres from env vars; not starting container")
        yield None
        return
    if not _have_testcontainers():
        pytest.skip(SKIP_REASON)
    try:
        with PostgresContainer(image="postgres:16-alpine") as container:
            print("[db_pool tests] Started Postgres test container")
            yield container
    except Exception as e:  # Docker not running or inaccessible
        pytest.skip(f"Skipping DB pool tests: {e}")


@pytest_asyncio.fixture(autouse=True)
async def setup_db(pg_container):
    # If we have a container, set env vars from it
    if pg_container is not None:
        os.environ["PG_HOST"] = pg_container.get_container_host_ip()
        os.environ["PG_PORT"] = str(pg_container.get_exposed_port(pg_container.port))
        os.environ["PG_DB"] = pg_container.dbname
        os.environ["PG_USER"] = pg_container.username
        os.environ["PG_PASSWORD"] = pg_container.password

    if not (os.getenv("PG_HOST") and os.getenv("PG_DB")):
        pytest.skip("No Postgres available (neither container nor external). Set USE_EXTERNAL_PG with PG_HOST/PG_DB or ensure Docker is running.")

    print(
        f"[db_pool tests] Using PG host={os.getenv('PG_HOST')} db={os.getenv('PG_DB')} (container={'yes' if pg_container else 'no'})"
    )

    await db_pool.init(force=True)

    # Create required tables (idempotent)
    await db_pool.execute(
        """
        CREATE TABLE IF NOT EXISTS public.challenge_staging (
            id BIGSERIAL PRIMARY KEY,
            file_content JSONB NOT NULL,
            file_path VARCHAR(1024) NOT NULL,
            json_created_at TIMESTAMP NOT NULL,
            staging_inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await db_pool.execute(
        """
        CREATE TABLE IF NOT EXISTS public.challenges (
            id BIGSERIAL PRIMARY KEY,
            staging_id BIGINT NOT NULL REFERENCES public.challenge_staging(id) ON DELETE CASCADE,
            challenge_uid VARCHAR(50) NOT NULL,
            dialogue_uid VARCHAR(50) NOT NULL,
            utterance_number INT NOT NULL CHECK (utterance_number >= 0),
            utterance_text TEXT NOT NULL,
            json_created_at TIMESTAMP NOT NULL,
            staging_inserted_at TIMESTAMP NOT NULL,
            submission_inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    yield


@pytest.mark.asyncio
async def test_db_pool_init():
    # health_check should pass
    assert await health_check() is True
    # Re-initializing without force should be idempotent (no exception)
    await db_pool.init()
    assert await health_check() is True


@pytest.mark.asyncio
async def test_insert_challenge_staging_and_fetch_ids():
    ts = datetime.now(UTC).replace(tzinfo=None)  # store naive UTC like schema
    staging_id = await insert_challenge_staging(
        file_content={"hello": "world"}, file_path="/tmp/a.json", json_created_at=ts
    )
    assert staging_id > 0

    # Insert one challenge row referencing it
    rows = [
        {
            "staging_id": staging_id,
            "challenge_uid": "c-123",
            "dialogue_uid": "d-1",
            "utterance_number": 0,
            "utterance_text": "Hi",
            "json_created_at": ts,
            "staging_inserted_at": ts,
        }
    ]
    await insert_challenges_bulk(rows)
    ids = await fetch_challenge_ids_by_uid("c-123")
    assert len(ids) == 1


@pytest.mark.asyncio
async def test_bulk_insert_multiple_challenges():
    ts = datetime.now(UTC).replace(tzinfo=None)
    s1 = await insert_challenge_staging(
        file_content={"a": 1}, file_path="/tmp/1.json", json_created_at=ts
    )
    s2 = await insert_challenge_staging(
        file_content={"b": 2}, file_path="/tmp/2.json", json_created_at=ts
    )
    rows = []
    for i, sid in enumerate((s1, s2)):
        rows.append(
            {
                "staging_id": sid,
                "challenge_uid": f"cu-{i}",
                "dialogue_uid": f"du-{i}",
                "utterance_number": i,
                "utterance_text": f"text-{i}",
                "json_created_at": ts,
                "staging_inserted_at": ts,
            }
        )
    await insert_challenges_bulk(rows)
    ids = await fetch_challenge_ids_by_uid("cu-0")
    assert ids, "Expected at least one row for cu-0"
