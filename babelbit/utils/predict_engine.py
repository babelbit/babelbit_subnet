from typing import Any, Optional
from time import monotonic
from json import loads
from random import uniform
from logging import getLogger
import time
import uuid

from asyncio import TimeoutError, sleep, gather, wait_for
from aiohttp import ClientError, ClientTimeout
from substrateinterface import Keypair

# from babelbit.chute_template.schemas import SVPredictInput
from babelbit.utils.data_models import BBPredictedUtterance, BBPredictOutput
from babelbit.utils.settings import get_settings
from babelbit.utils.async_clients import get_async_client
# from babelbit.utils.challenges import prepare_challenge_payload
from babelbit.utils.chutes_helpers import get_chute_slug_and_id
from babelbit.utils.bittensor_helpers import load_hotkey_keypair
from bittensor.utils import networking

logger = getLogger(__name__)


async def create_chute_prediction_callback(slug: str, timeout: Optional[float] = None):
    """
    Create a prediction callback function for use with utterance engine.
    
    Args:
        slug: The chute slug to call for predictions
        timeout: Timeout in seconds for chute predictions (default: uses CHUTES_TIMEOUT_SEC from settings)
        
    Returns:
        Async callback function that can be used with interact_with_utterance_engine
    """
    if timeout is None:
        settings = get_settings()
        timeout = float(settings.CHUTES_TIMEOUT_SEC)
    async def chute_predictor(session_id: str, current_word: str, utterance_index: int, context: str = "") -> str:
        """
        Call the chute to predict the next token in an utterance.
        
        Args:
            session_id: Current utterance session ID
            current_word: Current word/token revealed
            utterance_index: Index of current utterance
            context: Dialogue history accumulated so far
            
        Returns:
            Predicted next token as string
        """
        try:
            # Create predicted utterance payload with proper context
            payload = BBPredictedUtterance(
                index=session_id,
                step=utterance_index,
                prefix=current_word or "",
                prediction="",  # This will be filled by the chute
                context=context,  # Set the context string field
            )
            
            # Call the chute
            result = await call_miner_model_on_chutes(slug=slug, payload=payload, context_used=context, timeout=timeout)
            
            if result.success and result.utterance:
                # Extract the prediction from the result
                prediction = result.utterance.prediction
                logger.debug(f"Chute predicted: '{prediction}' for prefix '{current_word}'")
                return prediction
            
            logger.warning(f"Chute prediction failed or empty: {result.error}")
            return ""  # Return empty string as fallback
            
        except Exception as e:
            logger.error(f"Error in chute prediction: {e}")
            return ""  # Return empty string on error
    
    return chute_predictor


async def call_miner_model_on_chutes(slug: str, 
                                     payload: BBPredictedUtterance, 
                                     context_used: str, 
                                     timeout: Optional[float] = None) -> BBPredictOutput:
    """Call a miner's chute for utterance prediction."""
    if timeout is None:
        settings = get_settings()
        timeout = float(settings.CHUTES_TIMEOUT_SEC)
    return await predict_utterance(payload=payload, slug=slug, context_used=context_used, timeout=timeout)

# Cache for validator identity to avoid repeated file I/O
_VALIDATOR_IDENTITY_CACHE = None

def _get_validator_identity():
    """Get or cache validator's keypair and identity information."""
    global _VALIDATOR_IDENTITY_CACHE
    if _VALIDATOR_IDENTITY_CACHE is None:
        settings = get_settings()
        try:
            keypair = load_hotkey_keypair(
                settings.BITTENSOR_WALLET_COLD, 
                settings.BITTENSOR_WALLET_HOT
            )
            external_ip = networking.get_external_ip()
            validator_uuid = str(uuid.uuid4())
            _VALIDATOR_IDENTITY_CACHE = {
                'keypair': keypair,
                'hotkey': keypair.ss58_address,
                'external_ip': external_ip,
                'uuid': validator_uuid
            }
            logger.info(f"Validator identity initialized: hotkey={keypair.ss58_address[:8]}...")
        except Exception as e:
            logger.error(f"Failed to load validator keypair: {e}")
            raise
    return _VALIDATOR_IDENTITY_CACHE


async def call_miner_axon_endpoint(
    axon_ip: str,
    axon_port: int,
    payload: BBPredictedUtterance,
    context_used: str,
    miner_hotkey: str,
    timeout: Optional[float] = None
) -> BBPredictOutput:
    """
    Call a miner's axon endpoint directly for utterance prediction using Bittensor protocol.
    
    This function implements the Bittensor dendrite-axon communication protocol with:
    - Cryptographic signatures for request authentication
    - Nonce-based replay attack prevention
    - Proper header construction following Bittensor standards
    
    Args:
        axon_ip: IP address of the miner's axon
        axon_port: Port of the miner's axon
        payload: The utterance prediction request
        context_used: Context string used for this prediction
        miner_hotkey: The SS58 address of the target miner's hotkey
        timeout: Request timeout in seconds
        
    Returns:
        BBPredictOutput with prediction result or error
    """
    # Load settings once
    settings = get_settings()

    if timeout is None:
        timeout = float(settings.CHUTES_TIMEOUT_SEC)
    
    # Get validator identity (cached)
    try:
        validator_identity = _get_validator_identity()
    except Exception as e:
        return BBPredictOutput(
            success=False,
            model="axon",
            utterance=payload,
            error=f"validator_identity_error:{str(e)}",
            context_used=context_used,
            complete=False
        )

    # If running in development/local mode, translate localhost or host IPs so
    # containers can reach services running on the Docker host via host.docker.internal
    try:
        if getattr(settings, "BB_DEV_MODE", False):
            local_ip = getattr(settings, "BB_LOCAL_MINER_IP", "") or None
            # common local addresses to translate
            if axon_ip in ("127.0.0.1", "localhost", "0.0.0.0") or (local_ip and axon_ip == local_ip):
                logger.info(f"Dev mode: translating axon IP {axon_ip} -> host.docker.internal")
                axon_ip = "host.docker.internal"
    except Exception:
        # Non-fatal; continue with original axon_ip
        pass

    url = f"http://{axon_ip}:{axon_port}/{settings.BB_MINER_PREDICT_ENDPOINT}"
    session = await get_async_client()
    
    # Build Bittensor protocol headers
    nonce = time.time_ns()
    body_hash = ""  # Could compute hash of payload if needed for extra security
    
    # Construct the message to sign following Bittensor protocol:
    # Format: {nonce}.{dendrite_hotkey}.{axon_hotkey}.{uuid}.{body_hash}
    message = f"{nonce}.{validator_identity['hotkey']}.{miner_hotkey}.{validator_identity['uuid']}.{body_hash}"
    
    # Sign the message with validator's keypair
    signature = f"0x{validator_identity['keypair'].sign(message).hex()}"
    
    # Construct headers following Bittensor synapse format
    headers = {
        "Content-Type": "application/json",
        # Dendrite (validator) information
        "bt_header_dendrite_nonce": str(nonce),
        "bt_header_dendrite_hotkey": validator_identity['hotkey'],
        "bt_header_dendrite_signature": signature,
        "bt_header_dendrite_uuid": validator_identity['uuid'],
        "bt_header_dendrite_ip": validator_identity['external_ip'],
        "bt_header_dendrite_version": "7002000",  # Bittensor version
        # Axon (miner) information
        "bt_header_axon_hotkey": miner_hotkey,
        "bt_header_axon_ip": axon_ip,
        "bt_header_axon_port": str(axon_port),
        # Request metadata
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
            timeout=ClientTimeout(total=timeout)
        ) as response:
            latency = monotonic() - t0
            text = await response.text()
            
            if response.status != 200:
                try:
                    logger.debug(
                        "Axon non-200: status=%s body='%s' url=%s miner_hk=%s",
                        response.status, text[:200], url, miner_hotkey[:16] + "...",
                    )
                except Exception:
                    pass
                return BBPredictOutput(
                    success=False,
                    model="axon",
                    utterance=payload,
                    error=f"{response.status}:{text[:300]}",
                    context_used=context_used,
                    complete=False
                )
            
            try:
                data = loads(text)
                prediction = data.get("prediction", "")
                
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
                    complete=True
                )
            except Exception as e:
                return BBPredictOutput(
                    success=False,
                    model="axon",
                    utterance=payload,
                    error=f"parse:{str(e)}",
                    context_used=context_used,
                    complete=False
                )
                
    except TimeoutError:
        try:
            logger.debug("Axon timeout: url=%s miner_hk=%s timeout=%.2fs", url, miner_hotkey[:16] + "...", timeout)
        except Exception:
            pass
        return BBPredictOutput(
            success=False,
            model="axon",
            utterance=payload,
            error=f"timeout after {timeout}s",
            context_used=context_used,
            complete=False
        )
    except Exception as e:
        try:
            logger.debug(
                "Axon error: url=%s miner_hk=%s err_type=%s err='%s'",
                url, miner_hotkey[:16] + "...", type(e).__name__, str(e)[:300],
            )
        except Exception:
            pass
        return BBPredictOutput(
            success=False,
            model="axon",
            utterance=payload,
            error=f"{type(e).__name__}:{str(e)}",
            context_used=context_used,
            complete=False
        )


async def predict_utterance(
    payload: BBPredictedUtterance,
    slug: str,
    context_used: str,
    timeout: Optional[float] = None,
) -> BBPredictOutput:
    """
    Call a chute to predict the next token in an utterance.
    
    Args:
        payload: The utterance prediction payload
        slug: The chute slug to call
        context_used: The context that was used for the prediction
        timeout: Timeout in seconds for the HTTP request (default: uses CHUTES_TIMEOUT_SEC from settings)
        
    Returns:
        BBPredictOutput with prediction results
    """
    settings = get_settings()
    
    if timeout is None:
        timeout = float(settings.CHUTES_TIMEOUT_SEC)
    
    # Hard timeout wrapper to prevent hangs
    # Give extra time for retries: timeout * (retries + 1) + backoff overhead
    max_total_time = timeout * (settings.CHUTES_API_N_RETRIES + 1) + 5.0
    
    try:
        return await wait_for(
            _predict_utterance_impl(payload, slug, context_used, timeout, settings),
            timeout=max_total_time
        )
    except TimeoutError:
        logger.error(f"Hard timeout ({max_total_time}s) reached for slug={slug}")
        return BBPredictOutput(
            success=False,
            model="unknown",
            utterance=payload,
            error=f"hard_timeout:{max_total_time}s",
            context_used=context_used,
            complete=False,
        )


async def _predict_utterance_impl(
    payload: BBPredictedUtterance,
    slug: str,
    context_used: str,
    timeout: float,
    settings: Any,
) -> BBPredictOutput:
    """
    Internal implementation of predict_utterance with retry logic.
    """

    base_url = settings.CHUTES_MINER_BASE_URL_TEMPLATE.format(slug=slug)
    url = f"{base_url}/{settings.CHUTES_MINER_PREDICT_ENDPOINT}"
    api_key = settings.CHUTES_API_KEY

    if not api_key.get_secret_value():
        return BBPredictOutput(
            success=False,
            model="unknown",
            utterance=payload,
            error="CHUTES_API_KEY missing",
            context_used=context_used,
            complete=False,
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.get_secret_value()}",
    }
    session = await get_async_client()

    # Track latency from first attempt
    t0 = monotonic()
    last_err = None
    backoff = 0.25  # Base backoff in seconds

    for attempt in range(1, settings.CHUTES_API_N_RETRIES + 2):
        try:
            # Removed semaphore to allow unlimited concurrent requests to miners
            # Validators shouldn't be throttled by client-side concurrency limits
            async with session.post(
                url, headers=headers, json=payload.model_dump(mode="json"), timeout=ClientTimeout(total=timeout)
            ) as response:
                text = await response.text()
                if response.status == 429:
                    last_err = f"busy:{text[:120]}"
                    raise RuntimeError("busy")
                if 400 <= response.status < 500:
                    return BBPredictOutput(
                        success=False,
                        model="unknown",
                        utterance=payload,
                        error=f"{response.status}:{text[:300]}",
                        context_used=context_used,
                        complete=False,
                    )
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {text[:300]}")

                data = loads(text)  # Expected to match BBPredictOutput format

                # Parse the response and create BBPredictOutput
                return BBPredictOutput(
                    success=bool(data.get("success", True)),
                    model=data.get("model", "unknown"),
                    utterance=BBPredictedUtterance.model_validate(data.get("utterance", payload.model_dump())),
                    error=data.get("error"),
                    context_used=data.get("context_used", context_used),
                    complete=bool(data.get("complete", False)),
                )

        except TimeoutError as e:
            last_err = f"timeout:{e}"
            logger.warning(f"Chute prediction timeout: URL={url} slug={slug} error={e}")
            try:
                logger.debug(
                    "Chute timeout: slug=%s url=%s attempt=%d/%d elapsed=%.2fs timeout=%.2fs",
                    slug, url, attempt, settings.CHUTES_API_N_RETRIES + 1, monotonic() - t0, timeout,
                )
            except Exception:
                pass
        except ClientError as e:
            last_err = f"client_error:{type(e).__name__}:{e}"
            logger.warning(f"Chute prediction failed: URL={url} slug={slug} error={type(e).__name__}:{e}")
            try:
                logger.debug(
                    "Chute client_error: slug=%s url=%s attempt=%d/%d err_type=%s err='%s'",
                    slug, url, attempt, settings.CHUTES_API_N_RETRIES + 1, type(e).__name__, str(e)[:300],
                )
            except Exception:
                pass
        except Exception as e:
            last_err = f"error:{type(e).__name__}:{e}"
            logger.warning(f"Chute prediction error: URL={url} slug={slug} error={type(e).__name__}:{e}")
            try:
                logger.debug(
                    "Chute error: slug=%s url=%s attempt=%d/%d err_type=%s err='%s'",
                    slug, url, attempt, settings.CHUTES_API_N_RETRIES + 1, type(e).__name__, str(e)[:300],
                )
            except Exception:
                pass

        # Exponential backoff with jitter if not last attempt
        if attempt <= settings.CHUTES_API_N_RETRIES:
            sleep_s = backoff * (2 ** (attempt - 1))
            sleep_s *= 1.0 + uniform(-0.15, 0.15)
            await sleep(max(0.05, sleep_s))

    # Failed after all retries
    try:
        logger.debug(
            "Chute failed after retries: slug=%s url=%s elapsed=%.2fs last_err='%s'",
            slug, url, monotonic() - t0, (last_err or "unknown_error")[:300],
        )
    except Exception:
        pass
    return BBPredictOutput(
        success=False,
        model="unknown",
        utterance=payload,
        error=last_err or "unknown_error",
        context_used=context_used,
        complete=False,
    )
