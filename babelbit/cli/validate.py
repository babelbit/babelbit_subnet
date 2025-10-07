import os
import time
import asyncio
import logging
import traceback
from datetime import datetime, timedelta, timezone

import aiohttp
import bittensor as bt

from babelbit.utils.bittensor_helpers import (
    get_subtensor,
    _set_weights_with_confirmation,
)
from babelbit.utils.prometheus import (
    LASTSET_GAUGE,
    CACHE_DIR,
    CACHE_FILES,
    SCORES_BY_UID,
    CURRENT_WINNER,
)
from babelbit.utils.settings import get_settings
from babelbit.utils.db_pool import db_pool, _iter_scores_from_db

logger = logging.getLogger("babelbit.validator")

for noisy in ["websockets", "websockets.client", "substrateinterface", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


async def _validate_main(tail: int, alpha: float, m_min: int, tempo: int):
    settings = get_settings()
    logger.info(
        "Validator starting tail=%d alpha=%.3f tempo=%d netuid=%d hotkey=%s",
        tail,
        alpha,
        tempo,
        settings.BABELBIT_NETUID,
        f"{settings.BITTENSOR_WALLET_HOT}",
    )

    NETUID = settings.BABELBIT_NETUID

    wallet = bt.wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )

    st = None
    last_done = -1
    while True:
        try:
            if st is None:
                try:
                    st = await asyncio.wait_for(get_subtensor(), timeout=20)
                except asyncio.TimeoutError:
                    logger.warning("Subtensor init timeout (20s) — retrying…")
                    await asyncio.sleep(5)
                    continue
                except Exception as e:
                    logger.warning("Subtensor init error: %s — retrying…", e)
                    await asyncio.sleep(5)
                    continue

            try:
                block = await asyncio.wait_for(st.get_current_block(), timeout=15)
            except asyncio.TimeoutError:
                logger.warning("get_current_block timed out (15s) — resetting subtensor")
                st = None
                continue
            except Exception as e:
                logger.warning("Error reading current block: %s — resetting subtensor", e)
                st = None
                await asyncio.sleep(3)
                continue

            logger.debug("Current block=%d", block)

            if block % tempo != 0 or block <= last_done:
                try:
                    await asyncio.wait_for(st.wait_for_block(), timeout=30)
                except asyncio.TimeoutError:
                    logger.warning("wait_for_block timeout (30s) — refreshing subtensor")
                    st = None
                continue

            uids, weights = await get_weights(tail=tail, alpha=alpha, m_min=m_min)
            if not uids:
                logger.warning("No eligible uids this round; skipping.")
                last_done = block
                continue

            ok = await retry_set_weights(wallet, uids, weights)
            if ok:
                LASTSET_GAUGE.set(time.time())
                logger.info("set_weights OK at block %d", block)
            else:
                logger.warning("set_weights failed at block %d", block)

            try:
                sz = sum(
                    f.stat().st_size for f in CACHE_DIR.glob("*.jsonl") if f.is_file()
                )
                CACHE_FILES.set(len(list(CACHE_DIR.glob("*.jsonl"))))
            except Exception:
                pass

            last_done = block

        except asyncio.CancelledError:
            break
        except Exception as e:
            traceback.print_exc()
            logger.warning("Validator loop error: %s — reconnecting…", e)
            st = None
            await asyncio.sleep(5)


# ---------------- Weights selection ---------------- #

async def get_weights(tail: int = 28800, alpha: float = 0.2, m_min: int = 25):  # tail & m_min kept for CLI compatibility, ignored
    """Select winning miner weights (current challenge only).

        - Gets the current active challenge ID from utterance engine
        - Filters scores to only include miners who participated in the current challenge
        - Winner = miner with highest score among current challenge participants
          who are currently in the metagraph (hotkey present)
        - Returns sparse weights: single winner with weight 1.0.

    Fallback:
        - If no current challenge or no scores for current challenge: default to uid 0
    """
    from babelbit.utils.predict_utterances import get_current_challenge_uid
    
    settings = get_settings()
    st = await get_subtensor()
    NETUID = settings.BABELBIT_NETUID
    meta = await st.metagraph(NETUID)
    hk_to_uid = {hk: i for i, hk in enumerate(meta.hotkeys)}

    # Get the current active challenge ID
    current_challenge_uid = None
    utterance_engine_url = os.getenv("BB_UTTERANCE_ENGINE_URL", "http://localhost:8000")
    if utterance_engine_url:
        try:
            current_challenge_uid = await get_current_challenge_uid(utterance_engine_url)
            logger.info(f"Current active challenge: {current_challenge_uid}")
        except Exception as e:
            logger.warning(f"Failed to get current challenge ID from {utterance_engine_url}: {e}")

    # Fetch scores based on whether we have a current challenge ID
    try:
        await db_pool.init()
        
        if current_challenge_uid:
            # Use efficient challenge-specific query
            from babelbit.utils.db_pool import _iter_scores_for_challenge
            challenge_scores = await _iter_scores_for_challenge(current_challenge_uid)
            if not challenge_scores:
                logger.warning(f"No scores found for current challenge {current_challenge_uid} → default weight uid 0")
                return [0], [1.0]
            logger.info(f"Found {len(challenge_scores)} miners with scores for current challenge {current_challenge_uid}")
            # Convert to the expected format for later processing
            current_rows = [(hk, score, current_challenge_uid) for hk, score in challenge_scores]
        else:
            # Fallback: fetch recent rows and use most recent challenge
            logger.warning("No current challenge ID available, falling back to most recent challenge in DB")
            db_rows = await _iter_scores_from_db(limit=tail)
            if not db_rows:
                logger.warning("No DB scoring data → default weight uid 0")
                return [0], [1.0]
            
            # Parse and find latest challenge
            enriched = []
            for tup in db_rows:
                if len(tup) == 3:
                    hk, score, challenge_uid = tup
                else:
                    hk, score = tup[0], tup[1]
                    challenge_uid = ""  # unknown
                enriched.append((hk, score, challenge_uid))
            
            # Use most recent challenge in DB
            latest_challenge = None
            for hk, score, cu in enriched:
                latest_challenge = cu
                break
            current_rows = [r for r in enriched if r[2] == latest_challenge]
            logger.info(f"Using fallback challenge {latest_challenge} with {len(current_rows)} scores")
            
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("DB init/fetch failure in get_weights: %s", e)
        logger.warning("No DB scoring data → default weight uid 0")
        return [0], [1.0]

    # Latest score per miner (first row is most recent due to DESC ordering)
    latest_per_hk: dict[str, float] = {}
    for hk, score, cu in current_rows:
        if hk not in hk_to_uid:
            continue
        if hk not in latest_per_hk:
            latest_per_hk[hk] = score  # first seen is latest

    if not latest_per_hk:
        logger.warning("No eligible miner scores for latest challenge → default uid 0")
        return [0], [1.0]

    winner_hk = max(latest_per_hk.keys(), key=lambda k: latest_per_hk[k])
    winner_uid = hk_to_uid.get(winner_hk, 0)

    # Sparse weights (lighter on chain)
    uids = [winner_uid]
    weights = [1.0]

    # Prometheus (optional)
    for hk, v in latest_per_hk.items():
        uid = hk_to_uid.get(hk)
        if uid is not None:
            SCORES_BY_UID.labels(uid=str(uid)).set(v)
    CURRENT_WINNER.set(winner_uid)

    logger.info(
        "Winner hk=%s uid=%d score=%.4f (n=%d source=%s)",
        winner_hk[:8] + "…",
        winner_uid,
        latest_per_hk[winner_hk],
        1,
        "db" if db_rows else "empty",
    )
    return uids, weights


async def retry_set_weights(wallet, uids, weights):
    """
    1) Tente /set_weights du signer (HTTP)
    2) Fallback: set_weights local + confirmation par lecture du metagraph
    """
    settings = get_settings()
    NETUID = settings.BABELBIT_NETUID
    signer_url = settings.SIGNER_URL

    import aiohttp

    try:
        timeout = aiohttp.ClientTimeout(connect=5, total=300)  # Increased timeout for block confirmation
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            resp = await sess.post(
                f"{signer_url}/set_weights",
                json={
                    "netuid": NETUID,
                    "uids": uids,
                    "weights": weights,
                    "wait_for_inclusion": False,  # Non-blocking: don't wait for confirmation
                },
            )
            try:
                data = await resp.json()
            except Exception:
                data = {"raw": await resp.text()}
            if resp.status == 200 and data.get("success"):
                return True
            logger.warning("Signer error status=%s body=%s", resp.status, data)
    except aiohttp.ClientConnectorError as e:
        logger.info("Signer unreachable: %s — falling back to local set_weights", e)
    except asyncio.TimeoutError:
        logger.warning("Signer timed out — falling back to local set_weights")

    # ---- Fallback local ----
    return await _set_weights_with_confirmation(wallet, NETUID, uids, weights)
