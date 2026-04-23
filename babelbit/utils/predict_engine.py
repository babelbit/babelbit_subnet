from json import dumps, loads
from logging import getLogger
from time import monotonic
import time
from threading import Lock
import uuid
import asyncio
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

from asyncio import TimeoutError
from aiohttp import ClientTimeout
from bittensor.utils import networking

from babelbit.schemas.prediction import BBPredictedUtterance, BBPredictOutput
from babelbit.utils.async_clients import get_async_client
from babelbit.utils.bittensor_helpers import load_hotkey_keypair
from babelbit.utils.settings import get_settings

logger = getLogger(__name__)

_VALIDATOR_IDENTITY_CACHE = None
_GATEWAY_AUTH_TOKEN_CACHE: dict[str, tuple[str, float]] = {}
_GATEWAY_AUTH_TOKEN_INFLIGHT: dict[str, asyncio.Future[tuple[str, int, str]]] = {}
_GATEWAY_AUTH_TOKEN_LOCK = Lock()
_GATEWAY_AUTH_TOKEN_TTL_FALLBACK_S = 1800


def _normalize_managed_predict_url(endpoint_url: str, predict_endpoint: str) -> str:
    """Normalize managed container base URL to a concrete predict URL."""
    url = (endpoint_url or "").strip()
    if not url:
        return ""

    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"

    endpoint = str(predict_endpoint or "predict").strip().lstrip("/")
    if not endpoint:
        return url

    parsed = urlparse(url)
    path = parsed.path or ""

    # Targon rows commonly expose the service base URL; Round2 must POST to /predict.
    normalized_path = path.rstrip("/")
    if not normalized_path:
        normalized_path = f"/{endpoint}"
    elif normalized_path.split("/")[-1] != endpoint:
        normalized_path = f"{normalized_path}/{endpoint}"

    return urlunparse(parsed._replace(path=normalized_path))


def _extract_prediction_from_provider_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        prediction = payload.get("prediction")
        if isinstance(prediction, str):
            return prediction

        output = payload.get("output")
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            nested_prediction = output.get("prediction")
            if isinstance(nested_prediction, str):
                return nested_prediction
            deep_output = output.get("output")
            if isinstance(deep_output, dict):
                deep_prediction = deep_output.get("prediction")
                if isinstance(deep_prediction, str):
                    return deep_prediction
    return ""


def _normalize_gateway_runsync_url(endpoint_url: str) -> str:
    url = (endpoint_url or "").strip()
    if not url:
        return ""
    if not url.startswith("http://") and not url.startswith("https://"):
        return f"http://{url}"
    return url


def _derive_gateway_auth_url(gateway_runsync_url: str, auth_path: str, runsync_path: str) -> str:
    normalized_runsync_url = _normalize_gateway_runsync_url(gateway_runsync_url)
    if not normalized_runsync_url:
        return ""

    parsed = urlparse(normalized_runsync_url)
    current_path = parsed.path or ""

    normalized_auth_path = str(auth_path or "/auth/token").strip()
    if not normalized_auth_path.startswith("/"):
        normalized_auth_path = f"/{normalized_auth_path}"

    normalized_runsync_path = str(runsync_path or "/runsync").strip()
    if not normalized_runsync_path.startswith("/"):
        normalized_runsync_path = f"/{normalized_runsync_path}"

    if current_path.endswith(normalized_runsync_path):
        base_path = current_path[: -len(normalized_runsync_path)]
        target_path = f"{base_path}{normalized_auth_path}"
    elif current_path.endswith("/runsync"):
        target_path = f"{current_path[:-len('/runsync')]}{normalized_auth_path}"
    else:
        target_path = normalized_auth_path

    return urlunparse(parsed._replace(path=target_path))


def _gateway_auth_cache_key(
    *,
    auth_url: str,
    validator_hotkey: str,
) -> str:
    return f"{auth_url}|{validator_hotkey}"


def _get_cached_gateway_auth_token(cache_key: str) -> str:
    now = time.time()
    with _GATEWAY_AUTH_TOKEN_LOCK:
        entry = _GATEWAY_AUTH_TOKEN_CACHE.get(cache_key)
        if not entry:
            return ""
        token, expires_at = entry
        if expires_at <= now:
            _GATEWAY_AUTH_TOKEN_CACHE.pop(cache_key, None)
            return ""
        return token


def _store_gateway_auth_token(cache_key: str, token: str, ttl_seconds: int) -> None:
    ttl = max(30, int(ttl_seconds))
    with _GATEWAY_AUTH_TOKEN_LOCK:
        _GATEWAY_AUTH_TOKEN_CACHE[cache_key] = (token, time.time() + ttl)


def _clear_gateway_auth_token(cache_key: str) -> None:
    with _GATEWAY_AUTH_TOKEN_LOCK:
        _GATEWAY_AUTH_TOKEN_CACHE.pop(cache_key, None)


async def _get_or_request_gateway_auth_token(
    *,
    cache_key: str,
    auth_url: str,
    validator_identity: dict[str, Any],
    miner_hotkey: str,
    miner_uid: int,
    timeout: float,
) -> tuple[str, int, str]:
    now = time.time()
    future: asyncio.Future[tuple[str, int, str]] | None = None
    created_future = False

    with _GATEWAY_AUTH_TOKEN_LOCK:
        cached = _GATEWAY_AUTH_TOKEN_CACHE.get(cache_key)
        if cached:
            token, expires_at = cached
            if expires_at > now:
                return token, max(0, int(expires_at - now)), ""
            _GATEWAY_AUTH_TOKEN_CACHE.pop(cache_key, None)

        future = _GATEWAY_AUTH_TOKEN_INFLIGHT.get(cache_key)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            _GATEWAY_AUTH_TOKEN_INFLIGHT[cache_key] = future
            created_future = True

    if not created_future:
        return await future

    try:
        token, expires_in, auth_error = await _request_gateway_auth_token(
            auth_url=auth_url,
            validator_identity=validator_identity,
            miner_hotkey=miner_hotkey,
            miner_uid=miner_uid,
            timeout=timeout,
        )
        if token:
            _store_gateway_auth_token(cache_key, token, expires_in)
        future.set_result((token, expires_in, auth_error))
        return token, expires_in, auth_error
    except Exception as exc:
        future.set_exception(exc)
        raise
    finally:
        with _GATEWAY_AUTH_TOKEN_LOCK:
            current = _GATEWAY_AUTH_TOKEN_INFLIGHT.get(cache_key)
            if current is future:
                _GATEWAY_AUTH_TOKEN_INFLIGHT.pop(cache_key, None)


async def _request_gateway_auth_token(
    *,
    auth_url: str,
    validator_identity: dict[str, Any],
    miner_hotkey: str,
    miner_uid: int,
    timeout: float,
) -> tuple[str, int, str]:
    session = await get_async_client()
    request_specs: list[tuple[dict[str, Any], dict[str, Any]]] = [
        # Newer gateway contract:
        # - body requires hotkey + miner_hotkey + uid
        # - signature canonical input is {"miner_hotkey", "uid"}
        {
            "miner_hotkey": miner_hotkey,
            "uid": miner_uid,
        },
        {
            "hotkey": validator_identity["hotkey"],
            "miner_hotkey": miner_hotkey,
            "uid": miner_uid,
        },
        # Legacy gateway contract:
        # - body requires hotkey only
        # - signature canonical input is {"scope":"gateway"}
        {
            "scope": "gateway",
        },
        {
            "hotkey": validator_identity["hotkey"],
        },
    ]
    payload: dict[str, Any] | None = None
    last_error = "gateway_auth_request_failed"

    for idx in range(0, len(request_specs), 2):
        auth_input_payload, body_fields = request_specs[idx], request_specs[idx + 1]
        timestamp_ms = int(time.time() * 1000)
        nonce = str(time.time_ns())
        gateway_message = f"{timestamp_ms}|{nonce}|{dumps(auth_input_payload, separators=(',', ':'), sort_keys=True)}"
        raw_signature = validator_identity["keypair"].sign(gateway_message.encode("utf-8"))
        if isinstance(raw_signature, (bytes, bytearray)):
            signature_hex = raw_signature.hex()
        else:
            signature_hex = str(raw_signature)
            if signature_hex.startswith("0x"):
                signature_hex = signature_hex[2:]

        request_body: dict[str, Any] = {
            **body_fields,
            "timestamp_ms": timestamp_ms,
            "nonce": nonce,
            "signature": signature_hex,
        }

        async with session.post(
            auth_url,
            headers={"Content-Type": "application/json"},
            json=request_body,
            timeout=ClientTimeout(total=timeout),
        ) as response:
            text = await response.text()
            if response.status != 200:
                last_error = f"status={response.status} body={text[:300]}"
                # Try legacy auth schema/signature if the first attempt fails.
                if idx == 0:
                    continue
                return "", 0, last_error

            try:
                loaded = loads(text)
            except Exception as exc:
                return "", 0, f"invalid_json:{exc}"

            if not isinstance(loaded, dict):
                return "", 0, "payload_not_object"
            payload = loaded
            break

    if payload is None:
        return "", 0, last_error

    token = payload.get("auth_token") or payload.get("token")
    if not isinstance(token, str) or not token.strip():
        return "", 0, "missing_auth_token"

    expires_in_raw = payload.get("expires_in", _GATEWAY_AUTH_TOKEN_TTL_FALLBACK_S)
    try:
        expires_in = int(expires_in_raw)
    except Exception:
        expires_in = _GATEWAY_AUTH_TOKEN_TTL_FALLBACK_S
    if expires_in <= 0:
        expires_in = _GATEWAY_AUTH_TOKEN_TTL_FALLBACK_S

    return token.strip(), expires_in, ""


def _get_validator_identity():
    """Get or cache validator identity information used for Axon requests."""

    global _VALIDATOR_IDENTITY_CACHE
    if _VALIDATOR_IDENTITY_CACHE is None:
        settings = get_settings()
        keypair = load_hotkey_keypair(
            settings.BITTENSOR_WALLET_COLD,
            settings.BITTENSOR_WALLET_HOT,
        )
        _VALIDATOR_IDENTITY_CACHE = {
            "keypair": keypair,
            "hotkey": keypair.ss58_address,
            "external_ip": networking.get_external_ip(),
            "uuid": str(uuid.uuid4()),
        }
        logger.info("Validator identity initialized: hotkey=%s...", keypair.ss58_address[:8])
    return _VALIDATOR_IDENTITY_CACHE


async def call_miner_axon_endpoint(
    axon_ip: str,
    axon_port: int,
    payload: BBPredictedUtterance,
    context_used: str,
    miner_hotkey: str,
    timeout: Optional[float] = None,
) -> BBPredictOutput:
    """Call a miner's axon endpoint directly for utterance prediction."""

    settings = get_settings()
    if timeout is None:
        timeout = float(getattr(settings, "BB_MINER_TIMEOUT_SEC", 10))

    try:
        validator_identity = _get_validator_identity()
    except Exception as e:
        return BBPredictOutput(
            success=False,
            model="axon",
            utterance=payload,
            error=f"validator_identity_error:{e}",
            context_used=context_used,
            complete=False,
        )

    try:
        if getattr(settings, "BB_DEV_MODE", False):
            local_ip = getattr(settings, "BB_LOCAL_MINER_IP", "") or None
            if axon_ip in ("127.0.0.1", "localhost", "0.0.0.0") or (local_ip and axon_ip == local_ip):
                logger.info("Dev mode: translating axon IP %s -> host.docker.internal", axon_ip)
                axon_ip = "host.docker.internal"
    except Exception:
        pass

    url = f"http://{axon_ip}:{axon_port}/{settings.BB_MINER_PREDICT_ENDPOINT}"
    session = await get_async_client()

    nonce = time.time_ns()
    body_hash = ""
    message = (
        f"{nonce}.{validator_identity['hotkey']}."
        f"{miner_hotkey}.{validator_identity['uuid']}.{body_hash}"
    )
    signature = f"0x{validator_identity['keypair'].sign(message).hex()}"

    headers = {
        "Content-Type": "application/json",
        "bt_header_dendrite_nonce": str(nonce),
        "bt_header_dendrite_hotkey": validator_identity["hotkey"],
        "bt_header_dendrite_signature": signature,
        "bt_header_dendrite_uuid": validator_identity["uuid"],
        "bt_header_dendrite_ip": validator_identity["external_ip"],
        "bt_header_dendrite_version": "7002000",
        "bt_header_axon_hotkey": miner_hotkey,
        "bt_header_axon_ip": axon_ip,
        "bt_header_axon_port": str(axon_port),
        "timeout": str(timeout),
        "name": "BBPredictedUtterance",
        "computed_body_hash": body_hash,
    }

    t0 = monotonic()
    try:
        async with session.post(
            url,
            headers=headers,
            json=payload.model_dump(mode="json"),
            timeout=ClientTimeout(total=timeout),
        ) as response:
            text = await response.text()
            if response.status != 200:
                logger.debug(
                    "Axon non-200: status=%s body='%s' url=%s miner_hk=%s",
                    response.status,
                    text[:200],
                    url,
                    (miner_hotkey[:16] + "...") if miner_hotkey else "?",
                )
                return BBPredictOutput(
                    success=False,
                    model="axon",
                    utterance=payload,
                    error=f"{response.status}:{text[:300]}",
                    context_used=context_used,
                    complete=False,
                )

            try:
                data = loads(text)
            except Exception as e:
                return BBPredictOutput(
                    success=False,
                    model="axon",
                    utterance=payload,
                    error=f"parse:{e}",
                    context_used=context_used,
                    complete=False,
                )

            prediction = data.get("prediction", "") if isinstance(data, dict) else ""
            return BBPredictOutput(
                success=True,
                model="axon",
                utterance=BBPredictedUtterance(
                    index=payload.index,
                    step=payload.step,
                    prefix=payload.prefix,
                    prediction=prediction,
                    context=context_used,
                ),
                error=None,
                context_used=context_used,
                complete=True,
            )

    except TimeoutError:
        logger.debug(
            "Axon timeout: url=%s miner_hk=%s timeout=%.2fs elapsed=%.2fs",
            url,
            (miner_hotkey[:16] + "...") if miner_hotkey else "?",
            timeout,
            monotonic() - t0,
        )
        return BBPredictOutput(
            success=False,
            model="axon",
            utterance=payload,
            error=f"timeout after {timeout}s",
            context_used=context_used,
            complete=False,
        )
    except Exception as e:
        logger.debug(
            "Axon error: url=%s miner_hk=%s err_type=%s err='%s'",
            url,
            (miner_hotkey[:16] + "...") if miner_hotkey else "?",
            type(e).__name__,
            str(e)[:300],
        )
        return BBPredictOutput(
            success=False,
            model="axon",
            utterance=payload,
            error=f"{type(e).__name__}:{e}",
            context_used=context_used,
            complete=False,
        )


async def call_managed_container_endpoint(
    endpoint_url: str,
    payload: BBPredictedUtterance,
    context_used: str,
    miner_hotkey: str,
    timeout: Optional[float] = None,
) -> BBPredictOutput:
    """Call a subnet-owner-managed container endpoint for utterance prediction."""

    settings = get_settings()
    if timeout is None:
        timeout = float(getattr(settings, "BB_ARENA_MINER_TIMEOUT_SEC", getattr(settings, "BB_MINER_TIMEOUT_SEC", 10)))

    url = endpoint_url.strip()
    if not url:
        return BBPredictOutput(
            success=False,
            model="managed_container",
            utterance=payload,
            error="empty_endpoint_url",
            context_used=context_used,
            complete=False,
        )

    url = _normalize_managed_predict_url(
        endpoint_url=url,
        predict_endpoint=str(getattr(settings, "BB_MINER_PREDICT_ENDPOINT", "predict")),
    )

    session = await get_async_client()
    t0 = monotonic()
    try:
        async with session.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload.model_dump(mode="json"),
            timeout=ClientTimeout(total=timeout),
        ) as response:
            text = await response.text()
            if response.status != 200:
                logger.debug(
                    "Managed container non-200: status=%s body='%s' url=%s miner_hk=%s",
                    response.status,
                    text[:200],
                    url,
                    (miner_hotkey[:16] + "...") if miner_hotkey else "?",
                )
                return BBPredictOutput(
                    success=False,
                    model="managed_container",
                    utterance=payload,
                    error=f"{response.status}:{text[:300]}",
                    context_used=context_used,
                    complete=False,
                )

            try:
                data = loads(text)
            except Exception as e:
                return BBPredictOutput(
                    success=False,
                    model="managed_container",
                    utterance=payload,
                    error=f"parse:{e}",
                    context_used=context_used,
                    complete=False,
                )

            prediction = data.get("prediction", "") if isinstance(data, dict) else ""
            return BBPredictOutput(
                success=True,
                model="managed_container",
                utterance=BBPredictedUtterance(
                    index=payload.index,
                    step=payload.step,
                    prefix=payload.prefix,
                    prediction=prediction,
                    context=context_used,
                ),
                error=None,
                context_used=context_used,
                complete=True,
            )
    except TimeoutError:
        logger.debug(
            "Managed container timeout: url=%s miner_hk=%s timeout=%.2fs elapsed=%.2fs",
            url,
            (miner_hotkey[:16] + "...") if miner_hotkey else "?",
            timeout,
            monotonic() - t0,
        )
        return BBPredictOutput(
            success=False,
            model="managed_container",
            utterance=payload,
            error=f"timeout after {timeout}s",
            context_used=context_used,
            complete=False,
        )
    except Exception as e:
        logger.debug(
            "Managed container error: url=%s miner_hk=%s err_type=%s err='%s'",
            url,
            (miner_hotkey[:16] + "...") if miner_hotkey else "?",
            type(e).__name__,
            str(e)[:300],
        )
        return BBPredictOutput(
            success=False,
            model="managed_container",
            utterance=payload,
            error=f"{type(e).__name__}:{e}",
            context_used=context_used,
            complete=False,
        )


async def call_gateway_runsync_endpoint(
    gateway_url: str,
    payload: BBPredictedUtterance,
    context_used: str,
    miner_hotkey: str,
    miner_uid: Optional[int] = None,
    timeout: Optional[float] = None,
) -> BBPredictOutput:
    """Call gateway /runsync endpoint for arena predictions."""

    settings = get_settings()
    if timeout is None:
        timeout = float(getattr(settings, "BB_ARENA_MINER_TIMEOUT_SEC", getattr(settings, "BB_MINER_TIMEOUT_SEC", 10)))

    url = _normalize_gateway_runsync_url(gateway_url)
    if not url:
        return BBPredictOutput(
            success=False,
            model="gateway",
            utterance=payload,
            error="empty_gateway_url",
            context_used=context_used,
            complete=False,
        )

    try:
        validator_identity = _get_validator_identity()
    except Exception as e:
        return BBPredictOutput(
            success=False,
            model="gateway",
            utterance=payload,
            error=f"validator_identity_error:{e}",
            context_used=context_used,
            complete=False,
        )
    if not isinstance(miner_uid, int) or miner_uid < 0:
        return BBPredictOutput(
            success=False,
            model="gateway",
            utterance=payload,
            error="missing_miner_uid_for_gateway",
            context_used=context_used,
            complete=False,
        )

    predict_payload = payload.model_dump(mode="json")

    dendrite_nonce = time.time_ns()
    body_hash = ""
    dendrite_message = (
        f"{dendrite_nonce}.{validator_identity['hotkey']}."
        f"{miner_hotkey}.{validator_identity['uuid']}.{body_hash}"
    )
    raw_dendrite_signature = validator_identity["keypair"].sign(dendrite_message)
    if isinstance(raw_dendrite_signature, (bytes, bytearray)):
        dendrite_signature_hex = raw_dendrite_signature.hex()
    else:
        dendrite_signature_hex = str(raw_dendrite_signature)
        if dendrite_signature_hex.startswith("0x"):
            dendrite_signature_hex = dendrite_signature_hex[2:]

    bt_headers: dict[str, str] = {
        "bt_header_dendrite_nonce": str(dendrite_nonce),
        "bt_header_dendrite_hotkey": validator_identity["hotkey"],
        "bt_header_dendrite_signature": f"0x{dendrite_signature_hex}",
        "bt_header_dendrite_uuid": validator_identity["uuid"],
        "bt_header_dendrite_ip": validator_identity["external_ip"],
        "bt_header_dendrite_version": "7002000",
        "bt_header_axon_hotkey": miner_hotkey,
        "bt_header_axon_ip": "0.0.0.0",
        "bt_header_axon_port": "0",
        "timeout": str(timeout),
        "name": "BBPredictedUtterance",
        "computed_body_hash": body_hash,
    }
    gateway_input_payload: dict[str, Any] = {
        "predict_payload": predict_payload,
        "bt_headers": bt_headers,
    }

    gateway_auth_url = _derive_gateway_auth_url(
        gateway_runsync_url=url,
        auth_path=getattr(settings, "BB_ARENA_GATEWAY_AUTH_API_PATH", "/auth/token"),
        runsync_path=getattr(settings, "BB_ARENA_RUNSYNC_API_PATH", "/runsync"),
    )
    if not gateway_auth_url:
        return BBPredictOutput(
            success=False,
            model="gateway",
            utterance=payload,
            error="empty_gateway_auth_url",
            context_used=context_used,
            complete=False,
        )

    auth_cache_key = _gateway_auth_cache_key(
        auth_url=gateway_auth_url,
        validator_hotkey=str(validator_identity["hotkey"]),
    )

    session = await get_async_client()
    t0 = monotonic()
    try:
        for attempt in range(2):
            auth_token = _get_cached_gateway_auth_token(auth_cache_key)
            if not auth_token:
                token, expires_in, auth_error = await _get_or_request_gateway_auth_token(
                    cache_key=auth_cache_key,
                    auth_url=gateway_auth_url,
                    validator_identity=validator_identity,
                    miner_hotkey=str(miner_hotkey),
                    miner_uid=int(miner_uid),
                    timeout=float(timeout),
                )
                if not token:
                    return BBPredictOutput(
                        success=False,
                        model="gateway",
                        utterance=payload,
                        error=f"gateway_auth_failed:{auth_error}",
                        context_used=context_used,
                        complete=False,
                    )
                auth_token = token

            request_body: dict[str, Any] = {
                "input": gateway_input_payload,
                "auth_token": auth_token,
                "uid": miner_uid,
                "miner_hotkey": miner_hotkey,
            }

            async with session.post(
                url,
                headers={"Content-Type": "application/json"},
                json=request_body,
                timeout=ClientTimeout(total=timeout),
            ) as response:
                text = await response.text()
                if response.status == 401 and attempt == 0:
                    _clear_gateway_auth_token(auth_cache_key)
                    auth_token = ""
                    continue
                if response.status != 200:
                    logger.debug(
                        "Gateway non-200: status=%s body='%s' url=%s miner_hk=%s",
                        response.status,
                        text[:200],
                        url,
                        (miner_hotkey[:16] + "...") if miner_hotkey else "?",
                    )
                    return BBPredictOutput(
                        success=False,
                        model="gateway",
                        utterance=payload,
                        error=f"{response.status}:{text[:300]}",
                        context_used=context_used,
                        complete=False,
                    )

                try:
                    data = loads(text)
                except Exception as e:
                    return BBPredictOutput(
                        success=False,
                        model="gateway",
                        utterance=payload,
                        error=f"parse:{e}",
                        context_used=context_used,
                        complete=False,
                    )

                prediction = _extract_prediction_from_provider_payload(data)
                status = str(data.get("status", "")).upper() if isinstance(data, dict) else ""
                if not prediction:
                    detail = ""
                    if isinstance(data, dict):
                        error_field = data.get("error")
                        detail = f":{error_field}" if error_field is not None else ""
                    if status and status in {"COMPLETED", "SUCCEEDED"}:
                        return BBPredictOutput(
                            success=False,
                            model="gateway",
                            utterance=payload,
                            error=f"empty_prediction_from_gateway{detail}",
                            context_used=context_used,
                            complete=False,
                        )
                if not prediction and status and status not in {"COMPLETED", "SUCCEEDED"}:
                    detail = ""
                    if isinstance(data, dict):
                        error_field = data.get("error")
                        detail = f":{error_field}" if error_field is not None else ""
                    return BBPredictOutput(
                        success=False,
                        model="gateway",
                        utterance=payload,
                        error=f"gateway_status={status}{detail}",
                        context_used=context_used,
                        complete=False,
                    )

                return BBPredictOutput(
                    success=True,
                    model="gateway",
                    utterance=BBPredictedUtterance(
                        index=payload.index,
                        step=payload.step,
                        prefix=payload.prefix,
                        prediction=prediction,
                        context=context_used,
                    ),
                    error=None,
                    context_used=context_used,
                    complete=True,
                )

        return BBPredictOutput(
            success=False,
            model="gateway",
            utterance=payload,
            error="gateway_auth_failed:unauthorized_after_refresh",
            context_used=context_used,
            complete=False,
        )
    except TimeoutError:
        logger.debug(
            "Gateway timeout: url=%s miner_hk=%s timeout=%.2fs elapsed=%.2fs",
            url,
            (miner_hotkey[:16] + "...") if miner_hotkey else "?",
            timeout,
            monotonic() - t0,
        )
        return BBPredictOutput(
            success=False,
            model="gateway",
            utterance=payload,
            error=f"timeout after {timeout}s",
            context_used=context_used,
            complete=False,
        )
    except Exception as e:
        logger.debug(
            "Gateway error: url=%s miner_hk=%s err_type=%s err='%s'",
            url,
            (miner_hotkey[:16] + "...") if miner_hotkey else "?",
            type(e).__name__,
            str(e)[:300],
        )
        return BBPredictOutput(
            success=False,
            model="gateway",
            utterance=payload,
            error=f"{type(e).__name__}:{e}",
            context_used=context_used,
            complete=False,
        )


async def call_managed_route_endpoint(
    *,
    route: Any,
    payload: BBPredictedUtterance,
    context_used: str,
    miner_hotkey: str,
    timeout: Optional[float] = None,
) -> BBPredictOutput:
    """Dispatch arena route invocation based on provider metadata."""

    endpoint_url = str(getattr(route, "endpoint_url", "") or "").strip()
    provider = str(getattr(route, "provider", "") or "").strip().lower()
    miner_uid_raw = getattr(route, "miner_uid", None)
    miner_uid: Optional[int] = None
    if miner_uid_raw is not None:
        try:
            miner_uid = int(miner_uid_raw)
        except Exception:
            miner_uid = None

    if provider == "gateway":
        return await call_gateway_runsync_endpoint(
            gateway_url=endpoint_url,
            payload=payload,
            context_used=context_used,
            miner_hotkey=miner_hotkey,
            miner_uid=miner_uid,
            timeout=timeout,
        )

    return await call_managed_container_endpoint(
        endpoint_url=endpoint_url,
        payload=payload,
        context_used=context_used,
        miner_hotkey=miner_hotkey,
        timeout=timeout,
    )
