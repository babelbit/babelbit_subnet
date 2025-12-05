import os
import time
import json
import asyncio
import logging
import traceback
from typing import Optional, Tuple, List

import aiohttp
import bittensor as bt

from babelbit.utils.bittensor_helpers import (
    get_subtensor,
    reset_subtensor,
    _set_weights_with_confirmation,
    load_hotkey_keypair,
)
from babelbit.utils.prometheus import (
    LASTSET_GAUGE,
    CACHE_DIR,
    CACHE_FILES,
    SCORES_BY_UID,
    CURRENT_WINNER,
)
from babelbit.utils.settings import get_settings
from babelbit.utils.async_clients import get_async_client
from babelbit.utils.utterance_auth import init_utterance_auth, authenticate_utterance_engine
from babelbit.utils.predict_utterances import get_current_challenge_uid
from babelbit.utils.signing import sign_message

logger = logging.getLogger("babelbit.validator")

for noisy in ["websockets", "websockets.client", "substrateinterface", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


def _reset_no_score_if_challenge_changed(
    current_challenge_uid: Optional[str],
    last_challenge_uid: Optional[str],
    no_score_rounds: int,
) -> Tuple[int, Optional[str]]:
    """
    Zero the no-score counter when the challenge changes and track the new uid.
    """
    if current_challenge_uid and current_challenge_uid != last_challenge_uid:
        return 0, current_challenge_uid
    return no_score_rounds, last_challenge_uid


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
    # Track consecutive rounds with no scores from API.
    no_score_rounds = 0
    MAX_NO_SCORE_ROUNDS = int(os.getenv("BB_MAX_SKIPPED_WEIGHT_EPOCHS", "12"))
    DEFAULT_FALLBACK_UID = int(os.getenv("BB_DEFAULT_FALLBACK_UID", "248"))
    last_set_weights: Optional[Tuple[List[int], List[float]]] = None
    validator_kp = load_hotkey_keypair(settings.BITTENSOR_WALLET_COLD, settings.BITTENSOR_WALLET_HOT)
    last_challenge_uid: Optional[str] = None
    
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

            # Determine current challenge for scoring API
            try:
                current_challenge_uid = await get_current_challenge_uid(utterance_engine_url)
                logger.debug("validate: current_challenge_uid=%s", current_challenge_uid)
            except Exception as e:
                current_challenge_uid = None
                logger.warning("Unable to fetch current challenge UID: %s", e)

            # Reset counters when the challenge changes so skip tracking is per-challenge.
            no_score_rounds, last_challenge_uid = _reset_no_score_if_challenge_changed(
                current_challenge_uid, last_challenge_uid, no_score_rounds
            )

            meta = await st.metagraph(NETUID)
            uids, weights, no_score_rounds = await get_weights(
                metagraph=meta,
                validator_kp=validator_kp,
                challenge_uid=current_challenge_uid,
                last_weights=last_set_weights,
                no_score_rounds=no_score_rounds,
                max_no_score_rounds=MAX_NO_SCORE_ROUNDS,
                default_uid=DEFAULT_FALLBACK_UID,
            )

            if not uids:
                logger.info(
                    f"No weights to set this round (no scores from API). "
                    f"[no_score_rounds={no_score_rounds}/{MAX_NO_SCORE_ROUNDS}]"
                )
                last_done = block
                continue

            ok = await retry_set_weights(wallet, uids, weights)
            if ok:
                LASTSET_GAUGE.set(time.time())
                logger.info("set_weights OK at block %d", block)
                last_set_weights = (uids, weights)
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

async def get_weights(
    metagraph,
    validator_kp,
    challenge_uid: Optional[str],
    last_weights: Optional[Tuple[List[int], List[float]]],
    no_score_rounds: int,
    max_no_score_rounds: int,
    default_uid: int,
):
    """
    Fetch scores from the submit API and pick a winner. If no scores are
    available, reuse the last weights; after max_no_score_rounds, fall back to
    default_uid.
    """
    settings = get_settings()
    hk_to_uid = {hk: i for i, hk in enumerate(metagraph.hotkeys)}

    scores = await fetch_scores_from_api(
        base_url=settings.BB_SUBMIT_API_URL,
        validator_kp=validator_kp,
        challenge_uid=challenge_uid,
    )

    if scores:
        latest_per_hk: dict[str, float] = {}
        for row in scores:
            hk = row.get("miner_hotkey") or row.get("hotkey")
            score = row.get("challenge_mean_score") or row.get("score")
            if hk is None or score is None:
                continue
            if hk not in hk_to_uid:
                continue
            # First occurrence wins (API already aggregates)
            if hk not in latest_per_hk:
                latest_per_hk[hk] = float(score)

        if latest_per_hk:
            winner_hk = max(latest_per_hk.keys(), key=lambda k: latest_per_hk[k])
            winner_uid = hk_to_uid[winner_hk]
            uids = [winner_uid]
            weights = [1.0]

            # Prometheus metrics
            for hk, v in latest_per_hk.items():
                uid = hk_to_uid.get(hk)
                if uid is not None:
                    SCORES_BY_UID.labels(uid=str(uid)).set(v)
            CURRENT_WINNER.set(winner_uid)

            logger.info(
                f"Winner hk={winner_hk[:8]}… uid={winner_uid} score={latest_per_hk[winner_hk]:.4f} (challenge={challenge_uid or 'unknown'})",
            )
            return uids, weights, 0

    # No scores available
    no_score_rounds += 1
    if last_weights:
        logger.info(
            f"No scores available from API; reusing last weights (round {no_score_rounds}/{max_no_score_rounds}).",
        )
        return last_weights[0], last_weights[1], no_score_rounds

    if no_score_rounds >= max_no_score_rounds:
        logger.warning(
            f"No scores from API after {no_score_rounds} rounds; falling back to default uid {default_uid}.",
        )
        return [default_uid], [1.0], no_score_rounds

    logger.info(
        f"No scores available from API (round {no_score_rounds}/{max_no_score_rounds}); waiting for next iteration.",
    )
    return [], [], no_score_rounds


async def fetch_scores_from_api(base_url: str, validator_kp, challenge_uid: Optional[str]):
    """Call the submit API /v1/get_scores endpoint and return the scores list."""
    if not challenge_uid:
        logger.debug("fetch_scores_from_api: missing challenge_uid; skipping call")
        return []

    url = base_url.rstrip("/") + "/v1/get_scores"
    timestamp = int(time.time())
    payload = {
        "hotkey": validator_kp.ss58_address,
        "timestamp": timestamp,
        "challenge_id": challenge_uid,
        "data": {"challenge_uid": challenge_uid},
    }
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    signature_hex = sign_message(validator_kp, canonical)

    params = {
        "hotkey": validator_kp.ss58_address,
        "timestamp": timestamp,
        "signature": signature_hex,
        "challenge_uid": challenge_uid,
    }

    session = await get_async_client()
    try:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.warning(
                    "get_scores returned %s: %s (challenge_uid=%s)",
                    resp.status,
                    text,
                    challenge_uid,
                )
                return []
            body = await resp.json()
            return body.get("scores") or []
    except Exception as e:
        logger.warning("get_scores call failed: %s", e)
        return []


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
            logger.warning(f"Signer error status={resp.status} body={data}")
    except aiohttp.ClientConnectorError as e:
        logger.info(f"Signer unreachable: {e} — falling back to local set_weights")
    except asyncio.TimeoutError:
        logger.warning("Signer timed out — falling back to local set_weights")

    # ---- Fallback local ----
    retries = int(os.getenv("BB_SET_WEIGHTS_RETRIES", os.getenv("SIGNER_RETRIES", "20")))
    delay_s = float(
        os.getenv("BB_SET_WEIGHTS_RETRY_DELAY", os.getenv("SIGNER_RETRY_DELAY", "2"))
    )
    return await _set_weights_with_confirmation(
        wallet, NETUID, uids, weights, retries=retries, delay_s=delay_s
    )
