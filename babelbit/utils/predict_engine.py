from typing import Any
from time import monotonic
from json import loads
from random import uniform
from logging import getLogger

from asyncio import TimeoutError, sleep, gather
from aiohttp import ClientError, ClientTimeout

# from babelbit.chute_template.schemas import SVPredictInput
from babelbit.utils.data_models import BBPredictedUtterance, BBPredictOutput
from babelbit.utils.settings import get_settings
from babelbit.utils.async_clients import get_async_client, get_semaphore
# from babelbit.utils.challenges import prepare_challenge_payload
from babelbit.utils.chutes_helpers import get_chute_slug_and_id

logger = getLogger(__name__)


async def create_chute_prediction_callback(slug: str, timeout: float = 10.0):
    """
    Create a prediction callback function for use with utterance engine.
    
    Args:
        slug: The chute slug to call for predictions
        timeout: Timeout in seconds for chute predictions (default: 10.0)
        
    Returns:
        Async callback function that can be used with interact_with_utterance_engine
    """
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


async def call_miner_model_on_chutes(slug: str, payload: BBPredictedUtterance, context_used: str, timeout: float = 10.0) -> BBPredictOutput:
    """Call a miner's chute for utterance prediction."""
    return await predict_utterance(payload=payload, slug=slug, context_used=context_used, timeout=timeout)


async def predict_utterance(
    payload: BBPredictedUtterance,
    slug: str,
    context_used: str,
    timeout: float = 10.0,
) -> BBPredictOutput:
    """
    Call a chute to predict the next token in an utterance.
    
    Args:
        payload: The utterance prediction payload
        slug: The chute slug to call
        context_used: The context that was used for the prediction
        timeout: Timeout in seconds for the HTTP request (default: 10.0)
        
    Returns:
        BBPredictOutput with prediction results
    """
    settings = get_settings()

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
    semaphore = get_semaphore()

    # Track latency from first attempt
    t0 = monotonic()
    last_err = None
    backoff = 0.25  # Base backoff in seconds

    for attempt in range(1, settings.CHUTES_API_N_RETRIES + 2):
        try:
            async with semaphore:
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
        except ClientError as e:
            last_err = f"client_error:{type(e).__name__}:{e}"
            logger.warning(f"Chute prediction failed: URL={url} slug={slug} error={type(e).__name__}:{e}")
        except Exception as e:
            last_err = f"error:{type(e).__name__}:{e}"
            logger.warning(f"Chute prediction error: URL={url} slug={slug} error={type(e).__name__}:{e}")

        # Exponential backoff with jitter if not last attempt
        if attempt <= settings.CHUTES_API_N_RETRIES:
            sleep_s = backoff * (2 ** (attempt - 1))
            sleep_s *= 1.0 + uniform(-0.15, 0.15)
            await sleep(max(0.05, sleep_s))

    # Failed after all retries
    return BBPredictOutput(
        success=False,
        model="unknown",
        utterance=payload,
        error=last_err or "unknown_error",
        context_used=context_used,
        complete=False,
    )
