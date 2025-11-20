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
    reset_subtensor,
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
from babelbit.utils.utterance_auth import init_utterance_auth, authenticate_utterance_engine
from babelbit.utils.challenge_status import is_challenge_processed, is_challenge_processed_db

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

    # Initialize utterance engine authentication
    utterance_engine_url = os.getenv("BB_UTTERANCE_ENGINE_URL", "http://localhost:8000")
    if utterance_engine_url:
        try:
            init_utterance_auth(utterance_engine_url, settings.BITTENSOR_WALLET_COLD, settings.BITTENSOR_WALLET_HOT)
            await authenticate_utterance_engine()
            logger.info("✅ Utterance engine authentication successful")
        except Exception as e:
            logger.warning(f"Failed to authenticate with utterance engine: {e}")

    NETUID = settings.BABELBIT_NETUID

    wallet = bt.wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )

    st = None
    last_done = -1
    consecutive_skipped_epochs = 0
    last_epoch_block = -1  # Track last epoch boundary for counting
    last_challenge_uid = None  # Track current challenge to reset counter on change
    BB_MAX_SKIPPED_WEIGHT_EPOCHS = int(os.getenv("BB_MAX_SKIPPED_WEIGHT_EPOCHS", "6"))
    EPOCH_LENGTH = 360  # blocks per epoch
    
    while True:
        try:
            if st is None:
                try:
                    await reset_subtensor()  # Clear any stale cached connection
                    st = await asyncio.wait_for(get_subtensor(), timeout=20)
                except asyncio.TimeoutError:
                    logger.warning("Subtensor init timeout (20s) — retrying…")
                    st = None
                    await reset_subtensor()
                    await asyncio.sleep(5)
                    continue
                except Exception as e:
                    logger.warning("Subtensor init error: %s — retrying…", e)
                    st = None
                    await reset_subtensor()
                    await asyncio.sleep(5)
                    continue

            try:
                block = await asyncio.wait_for(st.get_current_block(), timeout=15)
            except asyncio.TimeoutError:
                logger.warning("get_current_block timed out (15s) — resetting subtensor")
                st = None
                await reset_subtensor()
                continue
            except Exception as e:
                logger.warning("Error reading current block: %s — resetting subtensor", e)
                st = None
                await reset_subtensor()
                await asyncio.sleep(3)
                continue

            logger.debug("Current block=%d", block)

            if block % tempo != 0 or block <= last_done:
                # Wait for next block or timeout
                # Note: Blocks are ~12s on finney, but can be delayed
                # Use a generous timeout and just retry on failure rather than resetting connection
                try:
                    await asyncio.wait_for(st.wait_for_block(), timeout=60)
                except asyncio.TimeoutError:
                    # Don't reset connection on timeout - just log and retry
                    # This is normal when blocks are slow or network is spotty
                    logger.debug("wait_for_block timeout (60s) — will retry on next iteration")
                    await asyncio.sleep(5)  # Brief sleep before retry
                except Exception as e:
                    logger.warning("wait_for_block error: %s — refreshing subtensor", e)
                    st = None
                    await reset_subtensor()
                    await asyncio.sleep(3)
                continue

            uids, weights, current_challenge_uid = await get_weights(
                tail=tail, 
                alpha=alpha, 
                m_min=m_min, 
                consecutive_skipped_epochs=consecutive_skipped_epochs
            )
            
            # Reset skip counter if challenge changed
            if current_challenge_uid and current_challenge_uid != last_challenge_uid:
                if last_challenge_uid:
                    logger.info(
                        f"Challenge changed from {last_challenge_uid} to {current_challenge_uid}. "
                        f"Resetting skip counter."
                    )
                consecutive_skipped_epochs = 0
                last_challenge_uid = current_challenge_uid
            
            if not uids:
                # Increment epoch counter if we're at a new epoch boundary
                current_epoch = block // EPOCH_LENGTH
                if current_epoch > (last_epoch_block // EPOCH_LENGTH):
                    consecutive_skipped_epochs += 1
                    last_epoch_block = block
                    
                logger.info(
                    f"No weights to set this round (waiting for runner to process current challenge) "
                    f"[skipped {consecutive_skipped_epochs}/{BB_MAX_SKIPPED_WEIGHT_EPOCHS} epochs]"
                )
                last_done = block
                continue
            
            # Reset skip counter on successful weight assignment
            consecutive_skipped_epochs = 0
            last_epoch_block = block

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
            await reset_subtensor()
            await asyncio.sleep(5)


# ---------------- Weights selection ---------------- #

async def get_weights(tail: int = 28800, alpha: float = 0.2, m_min: int = 25, consecutive_skipped_epochs: int = 0):  # tail & m_min kept for CLI compatibility, ignored
    """Select winning miner weights (current challenge only).

        - Gets the current active challenge ID from utterance engine
        - Filters scores to only include miners who participated in the current challenge
        - Winner = miner with highest score among current challenge participants
          who are currently in the metagraph (hotkey present)
        - Returns sparse weights: single winner with weight 1.0.

    Fallback:
        - If no current challenge or no scores for current challenge after MAX_SKIPPED_ROUNDS: default to uid 248
        
    Args:
        tail: (unused, kept for CLI compatibility)
        alpha: (unused, kept for CLI compatibility) 
        m_min: (unused, kept for CLI compatibility)
        consecutive_skipped_epochs: Number of consecutive epochs weights were skipped (for fallback logic)
        
    Returns:
        tuple: (uids, weights, challenge_uid) - The challenge_uid allows caller to detect challenge changes
    """
    from babelbit.utils.predict_utterances import get_current_challenge_uid
    
    settings = get_settings()
    st = await get_subtensor()
    NETUID = settings.BABELBIT_NETUID
    meta = await st.metagraph(NETUID)
    hk_to_uid = {hk: i for i, hk in enumerate(meta.hotkeys)}

    # Get max skipped epochs before falling back to default weight
    MAX_SKIPPED_EPOCHS = int(os.getenv("BB_MAX_SKIPPED_WEIGHT_EPOCHS", "6"))

    # Get the current active challenge ID
    current_challenge_uid = None
    utterance_engine_url = os.getenv("BB_UTTERANCE_ENGINE_URL", "http://localhost:8000")
    if utterance_engine_url:
        try:
            current_challenge_uid = await get_current_challenge_uid(utterance_engine_url)
            logger.info(f"Current active challenge: {current_challenge_uid}")
            logger.debug("get_weights: current_challenge_uid=%s from %s", current_challenge_uid, utterance_engine_url)
        except Exception as e:
            logger.warning(f"Failed to get current challenge ID from {utterance_engine_url}: {e}")

    # Check if current challenge has been processed before fetching scores
    challenge_processed = False
    if current_challenge_uid:
        # First check file-based status (fast)
        challenge_processed = is_challenge_processed(current_challenge_uid)
        logger.debug("get_weights: file-based challenge_processed=%s for %s", challenge_processed, current_challenge_uid)
        
        # If not found in files, check DB as fallback
        if not challenge_processed:
            logger.debug(f"Challenge {current_challenge_uid} not found in status files, checking DB...")
            challenge_processed = await is_challenge_processed_db(current_challenge_uid)
        logger.debug("get_weights: final challenge_processed=%s for %s", challenge_processed, current_challenge_uid)
    
    # If challenge hasn't been processed yet, use last challenge winner
    if current_challenge_uid and not challenge_processed:
        # Check if we've exceeded the maximum number of skipped epochs
        if consecutive_skipped_epochs >= MAX_SKIPPED_EPOCHS:
            logger.warning(
                f"Challenge {current_challenge_uid} has not been processed after {consecutive_skipped_epochs} epochs. "
                f"Falling back to default weight assignment (uid 248) to prevent stalling."
            )
            # Fall through to check DB - if still no scores, will return default at the end
        else:
            logger.info(
                f"Challenge {current_challenge_uid} has not been processed by runner yet. "
                f"Using last challenge winner for weights ({consecutive_skipped_epochs}/{MAX_SKIPPED_EPOCHS})."
            )
            # Use the last challenge instead of returning empty weights
            current_challenge_uid = None  # Will trigger fallback to most recent challenge in DB

    # Fetch scores based on whether we have a current challenge ID
    try:
        await db_pool.init()
        
        # Track which data source we used for logging
        data_source = None
        
        # Determine whether to use current challenge or fallback to last challenge
        use_current_challenge = current_challenge_uid is not None
        
        if use_current_challenge and current_challenge_uid:  # Type guard for mypy
            # Use efficient challenge-specific query
            from babelbit.utils.db_pool import _iter_scores_for_challenge
            challenge_scores = await _iter_scores_for_challenge(current_challenge_uid)
            logger.debug("get_weights: fetched %d score rows for current challenge %s", len(challenge_scores or []), current_challenge_uid)
            if not challenge_scores:
                # Check if we should fall back to default weight
                if consecutive_skipped_epochs >= MAX_SKIPPED_EPOCHS:
                    logger.warning(
                        f"No scores found for current challenge {current_challenge_uid} after {consecutive_skipped_epochs} epochs. "
                        f"Falling back to default weight (uid 248)."
                    )
                    return [248], [1.0], current_challenge_uid
                else:
                    logger.warning(
                        f"No scores found for current challenge {current_challenge_uid}. "
                        f"Using last challenge winner for weights ({consecutive_skipped_epochs}/{MAX_SKIPPED_EPOCHS})."
                    )
                    # Use the last challenge instead
                    use_current_challenge = False
                    logger.debug("get_weights: switching to fallback DB path (no scores for current challenge)")
            else:
                logger.info(f"Found {len(challenge_scores)} miners with scores for current challenge {current_challenge_uid}")
                # Convert to the expected format for later processing
                current_rows = [(hk, score, current_challenge_uid) for hk, score in challenge_scores]
                data_source = "current_challenge"
                logger.debug("get_weights: data_source=%s current_rows=%d", data_source, len(current_rows))
        
        if not use_current_challenge:
            # Fallback: fetch recent rows and use most recent challenge
            logger.info("Using most recent challenge in DB for weight assignment")
            db_rows = await _iter_scores_from_db(limit=tail)
            logger.debug("get_weights: fetched %d DB rows for fallback path", len(db_rows or []))
            if not db_rows:
                # No historical data available at all
                if consecutive_skipped_epochs >= MAX_SKIPPED_EPOCHS:
                    logger.warning(
                        f"No DB scoring data available (no historical challenges) after {consecutive_skipped_epochs} epochs. "
                        f"Falling back to default weight (uid 248)."
                    )
                    return [248], [1.0], None
                else:
                    logger.warning(
                        f"No DB scoring data available (no historical challenges). "
                        f"Cannot assign weights - waiting for first challenge to complete ({consecutive_skipped_epochs}/{MAX_SKIPPED_EPOCHS})."
                    )
                    return [], [], None
            
            # Parse and find latest challenge with timestamp check
            enriched = []
            for tup in db_rows:
                if len(tup) == 4:
                    hk, score, challenge_uid, timestamp = tup
                elif len(tup) == 3:
                    hk, score, challenge_uid = tup
                    timestamp = None
                else:
                    hk, score = tup[0], tup[1]
                    challenge_uid = ""  # unknown
                    timestamp = None
                enriched.append((hk, score, challenge_uid, timestamp))
            
            # Use most recent challenge in DB, but check if it's too old
            latest_challenge = None
            latest_timestamp = None
            for hk, score, cu, ts in enriched:
                latest_challenge = cu
                latest_timestamp = ts
                break
            logger.debug("get_weights: latest_challenge=%s latest_timestamp=%s", latest_challenge, latest_timestamp)
            
            # Check if the fallback challenge is older than 12 hours
            if latest_timestamp:
                age_hours = (datetime.now(timezone.utc) - latest_timestamp).total_seconds() / 3600
                if age_hours > 12:
                    logger.warning(
                        f"Latest challenge {latest_challenge} is {age_hours:.1f} hours old (>12h). "
                        f"Data too stale for weight assignment."
                    )
                    logger.debug("get_weights: fallback challenge age=%.2f hours exceeds threshold", age_hours)
                    if consecutive_skipped_epochs >= MAX_SKIPPED_EPOCHS:
                        logger.warning("Falling back to default weight (uid 248) after max skipped epochs.")
                        return [248], [1.0], None
                    else:
                        logger.info(f"Skipping weights this round ({consecutive_skipped_epochs}/{MAX_SKIPPED_EPOCHS}).")
                        return [], [], None
            
            current_rows = [(hk, score, cu) for hk, score, cu, ts in enriched if cu == latest_challenge]
            logger.info(f"Using fallback challenge {latest_challenge} with {len(current_rows)} scores")
            data_source = "fallback_db"
            logger.debug("get_weights: data_source=%s filtered current_rows=%d", data_source, len(current_rows))
            
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("DB init/fetch failure in get_weights: %s", e)
        # Check if we should fall back to default weight
        if consecutive_skipped_epochs >= MAX_SKIPPED_EPOCHS:
            logger.warning(
                f"No DB scoring data available due to error after {consecutive_skipped_epochs} epochs. "
                f"Falling back to default weight (uid 248)."
            )
            return [248], [1.0], None
        else:
            logger.warning(
                f"No DB scoring data available due to error. "
                f"Skipping weights this round (waiting for runner to process challenges) ({consecutive_skipped_epochs}/{MAX_SKIPPED_EPOCHS})."
            )
            return [], [], None

    # Latest score per miner (first row is most recent due to DESC ordering)
    latest_per_hk: dict[str, float] = {}
    for hk, score, cu in current_rows:
        if hk not in hk_to_uid:
            continue
        if hk not in latest_per_hk:
            latest_per_hk[hk] = score  # first seen is latest
    logger.debug("get_weights: latest_per_hk count=%d", len(latest_per_hk))

    # If we are using the current challenge and every score is 0, fall back to default uid 248
    try:
        if data_source == "current_challenge" and latest_per_hk and all((v or 0.0) == 0.0 for v in latest_per_hk.values()):
            logger.debug("get_weights: all zero scores for current challenge; scores=%s", list(latest_per_hk.values()))
            logger.warning(
                "All miner scores are 0.0 for current challenge %s. Falling back to default weight (uid 248).",
                current_challenge_uid,
            )
            return [248], [1.0], current_challenge_uid
    except NameError:
        # Defensive: if data_source isn't defined for some reason, ignore this check
        pass

    if not latest_per_hk:
        # Check if we should fall back to default weight
        logger.debug("get_weights: no eligible miners (data_source=%s, current_rows=%d)", data_source if 'data_source' in locals() else None, len(current_rows) if 'current_rows' in locals() else 0)
        if consecutive_skipped_epochs >= MAX_SKIPPED_EPOCHS:
            logger.warning(
                f"No eligible miner scores for latest challenge after {consecutive_skipped_epochs} epochs. "
                f"Falling back to default weight (uid 248)."
            )
            return [248], [1.0], current_challenge_uid
        else:
            logger.warning(
                f"No eligible miner scores for latest challenge. "
                f"Skipping weights this round (waiting for eligible miners) ({consecutive_skipped_epochs}/{MAX_SKIPPED_EPOCHS})."
            )
            return [], [], current_challenge_uid

    logger.debug("get_weights: selecting winner among %d miners", len(latest_per_hk))
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
        data_source or "unknown",
    )
    return uids, weights, current_challenge_uid


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
