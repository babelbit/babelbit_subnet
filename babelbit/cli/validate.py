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
from babelbit.utils.subtensor_gateway_client import SubtensorGatewayClient

logger = logging.getLogger("babelbit.validator")

for noisy in ["websockets", "websockets.client", "substrateinterface", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

TRAILING_INCENTIVE_FRACTION = 0.05
ARENA_INCENTIVE_PERCENT_ENV = "BB_ARENA_INCENTIVE_PERCENT"
WEIGHT_SUM_TOLERANCE = 1e-9
GET_SCORES_TIMEOUT_SECONDS = 30.0


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


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


def _coerce_score(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_mode(value, default: str = "main") -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "round1":
            return "main"
        if normalized == "round2":
            return "arena"
        if normalized in {"main", "arena"}:
            return normalized
    return default


def _to_api_challenge_type(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = _normalize_mode(value, default="main")
    if normalized == "main":
        return "qualifying"
    if normalized == "arena":
        return "arena"
    return None


def _get_arena_incentive_fraction() -> float:
    raw_percent = os.getenv(ARENA_INCENTIVE_PERCENT_ENV, "0")
    try:
        percent = float(raw_percent)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s=%r; defaulting to 0",
            ARENA_INCENTIVE_PERCENT_ENV,
            raw_percent,
        )
        return 0.0

    clamped_percent = min(max(percent, 0.0), 100.0)
    if clamped_percent != percent:
        logger.warning(
            "%s out of range (%s); clamping to %.1f",
            ARENA_INCENTIVE_PERCENT_ENV,
            raw_percent,
            clamped_percent,
        )
    return clamped_percent / 100.0


def _extract_mode_scores(
    scores,
    hk_to_uid: dict[str, int],
    default_mode: str = "main",
) -> Tuple[dict[str, float], dict[str, float]]:
    main_scores: dict[str, float] = {}
    arena_scores: dict[str, float] = {}

    for row in scores:
        hk = row.get("miner_hotkey") or row.get("hotkey")
        if hk is None or hk not in hk_to_uid:
            continue

        main_score = _coerce_score(row.get("main_score"))
        arena_score = _coerce_score(row.get("arena_score"))
        if main_score is not None or arena_score is not None:
            if main_score is not None and hk not in main_scores:
                main_scores[hk] = main_score
            if arena_score is not None and hk not in arena_scores:
                arena_scores[hk] = arena_score
            continue

        score = _coerce_score(row.get("challenge_mean_score"))
        if score is None:
            score = _coerce_score(row.get("score"))
        if score is None:
            continue

        mode = _normalize_mode(
            row.get("challenge_type") or row.get("challenge_round") or row.get("mode"),
            default=default_mode,
        )
        target = arena_scores if mode == "arena" else main_scores
        if hk not in target:
            target[hk] = score

    return main_scores, arena_scores


def _merge_first_scores(target: dict[str, float], incoming: dict[str, float]) -> None:
    for hotkey, score in incoming.items():
        if hotkey not in target:
            target[hotkey] = score


def _normalize_weight_vector(weights: List[float]) -> List[float]:
    if not weights:
        return []

    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        logger.warning("Computed non-positive total weight %.12f; dropping weight vector", total_weight)
        return []

    normalized = [float(weight) for weight in weights]
    if abs(total_weight - 1.0) > WEIGHT_SUM_TOLERANCE:
        logger.warning(
            "Weight vector sums to %.12f; renormalizing before submission",
            total_weight,
        )
        normalized = [weight / total_weight for weight in normalized]

    correction = 1.0 - sum(normalized)
    if abs(correction) > WEIGHT_SUM_TOLERANCE:
        normalized[-1] += correction

    return normalized


async def _validate_main(tail: int, alpha: float, m_min: int, tempo: int):
    settings = get_settings()
    arena_incentive_fraction = _get_arena_incentive_fraction()
    logger.info(
        (
            "Validator starting tail=%d alpha=%.3f tempo=%d netuid=%d hotkey=%s "
            "arena_split=%.2f%% main_split=%.2f%% trailing_split=%.2f%%"
        ),
        tail,
        alpha,
        tempo,
        settings.BABELBIT_NETUID,
        f"{settings.BITTENSOR_WALLET_HOT}",
        arena_incentive_fraction * 100.0,
        (1.0 - arena_incentive_fraction) * 100.0,
        TRAILING_INCENTIVE_FRACTION * 100.0,
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

    gateway = SubtensorGatewayClient()
    last_done = -1
    # Track consecutive rounds with no scores from API.
    no_score_rounds = 0
    MAX_NO_SCORE_ROUNDS = _get_int_env("BB_MAX_SKIPPED_WEIGHT_EPOCHS", 12)
    DEFAULT_FALLBACK_UID = _get_int_env("BB_DEFAULT_FALLBACK_UID", 248)
    last_set_weights: Optional[Tuple[List[int], List[float]]] = None
    validator_kp = load_hotkey_keypair(settings.BITTENSOR_WALLET_COLD, settings.BITTENSOR_WALLET_HOT)
    last_challenge_uid: Optional[str] = None
    while True:
        try:
            try:
                block = await asyncio.wait_for(gateway.get_current_block(), timeout=15)
            except asyncio.TimeoutError:
                logger.warning("get_current_block timed out (15s) from gateway")
                continue
            except Exception as e:
                logger.warning("Error reading current block from gateway: %s", e)
                await asyncio.sleep(3)
                continue

            logger.debug("Current block=%d", block)

            if block % tempo != 0 or block <= last_done:
                # Wait for next block or timeout
                # Note: Blocks are ~12s on finney, but can be delayed
                # Use a generous timeout and just retry on failure rather than resetting connection
                try:
                    await asyncio.wait_for(gateway.wait_for_block(), timeout=60)
                except asyncio.TimeoutError:
                    # Don't reset connection on timeout - just log and retry
                    # This is normal when blocks are slow or network is spotty
                    logger.debug("wait_for_block timeout (60s) — will retry on next iteration")
                    await asyncio.sleep(5)  # Brief sleep before retry
                except Exception as e:
                    logger.warning("wait_for_block error from gateway: %s", e)
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

            meta = await gateway.metagraph_object(netuid=NETUID, lite=False)
            uids, weights, no_score_rounds = await get_weights(
                metagraph=meta,
                validator_kp=validator_kp,
                challenge_uid=current_challenge_uid,
                last_weights=last_set_weights,
                no_score_rounds=no_score_rounds,
                max_no_score_rounds=MAX_NO_SCORE_ROUNDS,
                default_uid=DEFAULT_FALLBACK_UID,
                arena_incentive_fraction=arena_incentive_fraction,
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
            await asyncio.sleep(5)


def compute_weights(
    winner_uid: Optional[int],
    trailing_uid_dict: dict[int, float],
    arena_winner_uid: Optional[int] = None,
    arena_incentive_fraction: float = 0.0,
    burn_uid: Optional[int] = None,
):
    # Sparse weights: keep trailing miners at fixed 5%, split the remaining 95%
    # between main and arena winners.
    positive_trailing = [
        (uid, max(float(score or 0.0), 0.0))
        for uid, score in trailing_uid_dict.items()
        if max(float(score or 0.0), 0.0) > 0.0
    ]
    trailing_total = TRAILING_INCENTIVE_FRACTION if positive_trailing else 0.0
    core_total = 1.0 - trailing_total
    arena_fraction = min(max(float(arena_incentive_fraction or 0.0), 0.0), 1.0)

    main_core = core_total * (1.0 - arena_fraction)
    arena_core = core_total * arena_fraction
    main_recipient_uid = winner_uid
    arena_recipient_uid = arena_winner_uid

    if winner_uid is None and arena_winner_uid is None and burn_uid is not None:
        main_recipient_uid = burn_uid
        arena_recipient_uid = burn_uid

    if arena_winner_uid is None and arena_core > 0.0 and burn_uid is not None:
        arena_recipient_uid = burn_uid

    if arena_winner_uid is None:
        if burn_uid is None:
            main_core = core_total
            arena_core = 0.0
    if winner_uid is None:
        arena_core = core_total
        main_core = 0.0
        if arena_winner_uid is None and burn_uid is not None:
            arena_recipient_uid = burn_uid

    weights_by_uid: dict[int, float] = {}
    ordered_uids: List[int] = []

    def add_uid(uid: Optional[int], weight: float):
        if uid is None or weight <= 0.0:
            return
        if uid not in weights_by_uid:
            ordered_uids.append(uid)
            weights_by_uid[uid] = 0.0
        weights_by_uid[uid] += weight

    add_uid(main_recipient_uid, main_core)
    add_uid(arena_recipient_uid, arena_core)

    if positive_trailing and trailing_total > 0.0:
        total_trailing_score = sum(score for _, score in positive_trailing)
        for uid, score in positive_trailing:
            add_uid(uid, trailing_total * score / total_trailing_score)

    if not ordered_uids and winner_uid is not None:
        return [1.0], [winner_uid]
    if not ordered_uids and arena_winner_uid is not None:
        return [1.0], [arena_winner_uid]

    return _normalize_weight_vector([weights_by_uid[uid] for uid in ordered_uids]), ordered_uids
        
# ---------------- Weights selection ---------------- #

async def get_weights(
    metagraph,
    validator_kp,
    challenge_uid: Optional[str],
    last_weights: Optional[Tuple[List[int], List[float]]],
    no_score_rounds: int,
    max_no_score_rounds: int,
    default_uid: int,
    arena_incentive_fraction: Optional[float] = None,
):
    """
    Fetch scores from the submit API and pick a winner. If no scores are
    available, reuse the last weights; after max_no_score_rounds, fall back to
    default_uid.
    """
    settings = get_settings()
    if arena_incentive_fraction is None:
        arena_incentive_fraction = _get_arena_incentive_fraction()
    hk_to_uid = {hk: i for i, hk in enumerate(metagraph.hotkeys)}

    main_scores: dict[str, float] = {}
    arena_scores: dict[str, float] = {}

    main_mode_scores, arena_mode_scores = await asyncio.gather(
        fetch_scores_from_api(
            base_url=settings.BB_SUBMIT_API_URL,
            validator_kp=validator_kp,
            challenge_uid=challenge_uid,
            challenge_type="main",
        ),
        fetch_scores_from_api(
            base_url=settings.BB_SUBMIT_API_URL,
            validator_kp=validator_kp,
            challenge_uid=challenge_uid,
            challenge_type="arena",
        ),
    )

    if main_mode_scores:
        extracted_main, extracted_arena = _extract_mode_scores(
            main_mode_scores,
            hk_to_uid,
            default_mode="main",
        )
        _merge_first_scores(main_scores, extracted_main)
        _merge_first_scores(arena_scores, extracted_arena)

    if arena_mode_scores:
        extracted_main, extracted_arena = _extract_mode_scores(
            arena_mode_scores,
            hk_to_uid,
            default_mode="arena",
        )
        _merge_first_scores(main_scores, extracted_main)
        _merge_first_scores(arena_scores, extracted_arena)

    # Backward compatibility with older submit-api versions that do not
    # support challenge_type filtering.
    if not main_scores and not arena_scores:
        legacy_scores = await fetch_scores_from_api(
            base_url=settings.BB_SUBMIT_API_URL,
            validator_kp=validator_kp,
            challenge_uid=challenge_uid,
            challenge_type=None,
        )
        if legacy_scores:
            main_scores, arena_scores = _extract_mode_scores(legacy_scores, hk_to_uid)

    if main_scores or arena_scores:
        combined_hotkeys = list(dict.fromkeys(list(main_scores.keys()) + list(arena_scores.keys())))

        if combined_hotkeys:
            winner_hk = max(main_scores, key=main_scores.get) if main_scores else None
            arena_winner_hk = max(arena_scores, key=arena_scores.get) if arena_scores else None
            winner_uid = hk_to_uid.get(winner_hk) if winner_hk else None
            arena_winner_uid = hk_to_uid.get(arena_winner_hk) if arena_winner_hk else None

            trailing_uid_dict: dict[int, float] = {}
            for hk in combined_hotkeys:
                uid = hk_to_uid.get(hk)
                if uid is None:
                    continue
                if uid in {winner_uid, arena_winner_uid}:
                    continue
                main_score = main_scores.get(hk, 0.0)
                arena_score = arena_scores.get(hk, 0.0)
                composite = ((1.0 - arena_incentive_fraction) * main_score) + (
                    arena_incentive_fraction * arena_score
                )
                if composite > 0.0:
                    trailing_uid_dict[uid] = composite

            if winner_uid is not None:
                weights, uids = compute_weights(
                    winner_uid,
                    trailing_uid_dict,
                    arena_winner_uid=arena_winner_uid,
                    arena_incentive_fraction=arena_incentive_fraction,
                    burn_uid=default_uid,
                )
            elif arena_winner_uid is not None:
                weights, uids = compute_weights(
                    arena_winner_uid,
                    trailing_uid_dict,
                    arena_winner_uid=arena_winner_uid,
                    arena_incentive_fraction=1.0,
                    burn_uid=default_uid,
                )
            else:
                weights, uids = [], []

            # Prometheus (optional)
            for hk in combined_hotkeys:
                uid = hk_to_uid.get(hk)
                if uid is not None:
                    main_score = main_scores.get(hk, 0.0)
                    arena_score = arena_scores.get(hk, 0.0)
                    combined_score = ((1.0 - arena_incentive_fraction) * main_score) + (
                        arena_incentive_fraction * arena_score
                    )
                    SCORES_BY_UID.labels(uid=str(uid)).set(combined_score)
            if winner_uid is not None:
                CURRENT_WINNER.set(winner_uid)
            elif arena_winner_uid is not None:
                CURRENT_WINNER.set(arena_winner_uid)

            logger.info(
                (
                    "Weight winners main=%s arena=%s arena_split=%.2f%% main_split=%.2f%% "
                    "(challenge=%s)"
                ),
                f"{winner_hk[:8]}… uid={winner_uid}" if winner_hk else "none",
                f"{arena_winner_hk[:8]}… uid={arena_winner_uid}" if arena_winner_hk else "none",
                arena_incentive_fraction * 100.0,
                (1.0 - arena_incentive_fraction) * 100.0,
                challenge_uid or "unknown",
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


async def fetch_scores_from_api(
    base_url: str,
    validator_kp,
    challenge_uid: Optional[str],
    challenge_type: Optional[str] = None,
):
    """Call the submit API /v1/get_scores endpoint and return the scores list."""
    if not challenge_uid:
        logger.debug("fetch_scores_from_api: missing challenge_uid; skipping call")
        return []

    url = base_url.rstrip("/") + "/v1/get_scores"
    timestamp = int(time.time())
    api_challenge_type = _to_api_challenge_type(challenge_type)
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
    if api_challenge_type is not None:
        params["challenge_type"] = api_challenge_type

    session = await get_async_client()
    req_timeout = aiohttp.ClientTimeout(total=GET_SCORES_TIMEOUT_SECONDS)
    timeout_s = req_timeout.total
    try:
        async with session.get(url, params=params, timeout=req_timeout) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.warning(
                    "get_scores returned %s: %s (challenge_uid=%s challenge_type=%s)",
                    resp.status,
                    text,
                    challenge_uid,
                    api_challenge_type or "unspecified",
                )
                return []
            body = await resp.json()
            return body.get("scores") or []
    except asyncio.TimeoutError:
        logger.warning(
            "get_scores call timed out after %ss (challenge_uid=%s challenge_type=%s url=%s)",
            timeout_s if timeout_s is not None else "unknown",
            challenge_uid,
            api_challenge_type or "unspecified",
            url,
        )
        return []
    except Exception as e:
        logger.warning(
            "get_scores call failed (%s) challenge_uid=%s challenge_type=%s url=%s: %s",
            e.__class__.__name__,
            challenge_uid,
            api_challenge_type or "unspecified",
            url,
            e,
        )
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

    # ---- Fallback via subtensor gateway ----
    retries = int(os.getenv("BB_SET_WEIGHTS_RETRIES", os.getenv("SIGNER_RETRIES", "20")))
    delay_s = float(
        os.getenv("BB_SET_WEIGHTS_RETRY_DELAY", os.getenv("SIGNER_RETRY_DELAY", "2"))
    )
    return await _set_weights_with_confirmation(
        wallet=wallet,
        netuid=NETUID,
        uids=uids,
        weights=weights,
        retries=retries,
        delay_s=delay_s,
    )


async def _set_weights_with_confirmation(
    wallet,
    netuid: int,
    uids: list[int],
    weights: list[float],
    wait_for_inclusion: bool = False,
    retries: int = 20,
    delay_s: float = 2.0,
    log_prefix: str = "[bb-local]",
) -> bool:
    gateway = SubtensorGatewayClient()
    try:
        return await gateway.set_weights_and_confirm(
            netuid=netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=wait_for_inclusion,
            retries=retries,
            delay_s=delay_s,
            wallet_hotkey=wallet.hotkey.ss58_address,
        )
    except Exception as e:
        logger.warning(
            "%s gateway set_weights failed: %s: %s",
            log_prefix,
            type(e).__name__,
            e,
        )
        return False
