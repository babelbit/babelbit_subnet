from logging import getLogger
from time import perf_counter, time
from typing import Any, Callable, Optional, Union, Awaitable, List, Dict, Tuple
from inspect import iscoroutinefunction
from hashlib import sha256
from json import dumps
from random import shuffle, randint
import asyncio

from aiohttp import ClientResponseError

from babelbit.utils.settings import get_settings
from babelbit.utils.bittensor_helpers import load_hotkey_keypair, wait_until_block_modulo
from babelbit.utils.signing import sign_message
from babelbit.utils.async_clients import get_async_client
from babelbit.utils.utterance_auth import (
    get_auth_headers,
    authenticate_utterance_engine,
)
from babelbit.chute_template.schemas import BBPredictedUtterance

logger = getLogger(__name__)


async def retry_with_exponential_backoff(
    func: Callable,
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    **kwargs
):
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        backoff_factor: Multiplier for each retry (default: 2.0)
        *args, **kwargs: Arguments to pass to func
        
    Returns:
        Result of successful function call
        
    Raises:
        Exception: The last exception if all retries fail
    """
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(
                    f"All {max_retries} attempts failed for {func.__name__}: {type(e).__name__}: {e}"
                )
    
    if last_exception:
        raise last_exception
    raise RuntimeError(f"Retry failed without exception for {func.__name__}")


class BabelbitUtteranceError(Exception):
    pass


class UtteranceEngineError(Exception):
    pass


class ScoreVisionChallengeError(Exception):
    pass


BOUNDARY_TOKENS = {"EOF", "EOF EOF"}


def _is_content_token(token: Optional[str]) -> bool:
    return bool(token) and token not in BOUNDARY_TOKENS


async def _request_with_reauth(session, method: str, url: str, *, json_payload: Optional[dict] = None, allow_retry: bool = True):
    """
    Send an authenticated request and, on 401, refresh auth once then retry.
    Returns (status, parsed_json_or_text).
    """
    headers = await get_auth_headers()
    request_kwargs = {"headers": headers}
    if json_payload is not None:
        request_kwargs["json"] = json_payload

    # Prefer method-specific call (get/post) if available to match existing mocks
    caller = getattr(session, method.lower(), None)
    if caller is None:
        caller = session.request

    async with caller(method, url, **request_kwargs) if caller is session.request else caller(url, **request_kwargs) as response:
        if response.status == 401 and allow_retry:
            logger.warning("Utterance engine returned 401 — refreshing auth and retrying once.")
            await authenticate_utterance_engine()
            return await _request_with_reauth(session, method, url, json_payload=json_payload, allow_retry=False)

        try:
            data = await response.json()
        except Exception:
            data = await response.text()

        return response.status, data


async def _call_engine_json(
    session,
    method: str,
    url: str,
    *,
    payload: Optional[dict] = None,
    invalid_session_error: Optional[str] = None,
    error_label: str = "request",
) -> Dict[str, Any]:
    status, data = await _request_with_reauth(session, method, url, json_payload=payload)
    if status != 200:
        error_data = data if isinstance(data, dict) else {}
        if invalid_session_error and error_data.get("error") == "invalid or missing session_id":
            raise UtteranceEngineError(invalid_session_error)
        raise UtteranceEngineError(f"Failed to {error_label}: HTTP {status}")
    return data if isinstance(data, dict) else {}


def _build_engine_calls(session, base_url: str, *, solo: bool = False):
    prefix = "/solo" if solo else ""
    start_url = f"{base_url}{prefix}/start"
    next_url = f"{base_url}{prefix}/next"

    async def start():
        return await _call_engine_json(
            session,
            "GET",
            start_url,
            error_label=f"start {'solo ' if solo else ''}session",
        )

    async def advance(session_id: str, prediction: str = ""):
        payload = {"session_id": session_id, "prediction": prediction}
        return await _call_engine_json(
            session,
            "POST",
            next_url,
            payload=payload,
            invalid_session_error=f"Invalid session: {session_id}",
            error_label=f"get next {'solo ' if solo else ''}token",
        )

    return start, advance


def _ensure_dialogue_tracking(
    dialogues: Dict[str, List[BBPredictedUtterance]],
    contexts: Dict[str, str],
    dialogue_uid: Optional[str],
) -> None:
    if not dialogue_uid:
        return
    if dialogue_uid not in dialogues:
        dialogues[dialogue_uid] = []
    if dialogue_uid not in contexts:
        contexts[dialogue_uid] = ""


def _finalize_utterance(
    dialogues: Dict[str, List[BBPredictedUtterance]],
    contexts: Dict[str, str],
    dialogue_uid: Optional[str],
    tokens: List[str],
    *,
    mark_done: bool = True,
    update_context: bool = True,
) -> None:
    if not dialogue_uid or not tokens:
        return

    ground_truth_text = " ".join(tokens)
    steps = dialogues.get(dialogue_uid)
    if steps:
        if mark_done:
            steps[-1].done = True
        steps[-1].ground_truth = ground_truth_text

    if update_context:
        existing_context = contexts.get(dialogue_uid, "")
        contexts[dialogue_uid] = f"{existing_context} EOF {ground_truth_text}" if existing_context else ground_truth_text


async def get_current_challenge_uid(utterance_engine_url: str) -> Optional[str]:
    """
    Call the utterance engine /start endpoint to get the current challenge ID.
    
    Args:
        utterance_engine_url: Base URL of the utterance engine (e.g., "http://localhost:8000")

    Returns:
        The current challenge_uid, or None if not available or if there's an error

    Raises:
        UtteranceEngineError: If the API interaction fails
    """
    async def _get_challenge():
        session = await get_async_client()
        status, start_data = await _request_with_reauth(session, "GET", f"{utterance_engine_url}/start")
        if status != 200:
            raise UtteranceEngineError(f"Failed to get challenge ID: HTTP {status}")
        
        challenge_uid = start_data.get("challenge_uid") if isinstance(start_data, dict) else None
        
        logger.info(f"Current challenge ID: {challenge_uid}")
        return challenge_uid
    
    try:
        return await retry_with_exponential_backoff(_get_challenge, max_retries=3, initial_delay=1.0)
    except Exception as e:
        logger.error(f"Error getting challenge ID after retries: {e}")
        raise UtteranceEngineError(f"Failed to get challenge ID: {e}")


# async def predict_with_utterance_engine(
#     utterance_engine_url: str,
#     chute_slug: Optional[str] = None,
#     challenge_logger: Optional[Any] = None,
#     timeout: float = 10.0,
#     max_prediction_errors: int = 5
# ) -> Dict[str, List[BBPredictedUtterance]]:
#     """
#     Interact with the utterance engine API to get complete dialogues using a chute for predictions.
    
#     Args:
#         utterance_engine_url: Base URL of the utterance engine (e.g., "http://localhost:8000")
#         chute_slug: Optional chute slug to use for predictions. If None, uses empty predictions.
#         challenge_logger: Optional logger for recording step-by-step interaction events
#         timeout: Timeout for chute predictions in seconds
#         max_prediction_errors: Maximum consecutive prediction errors before interrupting (default: 5)
    
#     Returns:
#         Dict mapping dialogue_uid to List of BBPredictedUtterance objects for each dialogue
    
#     Raises:
#         UtteranceEngineError: If the API interaction fails
#         BabelbitUtteranceError: If too many consecutive prediction errors occur
#     """
#     from babelbit.utils.predict_engine import call_miner_model_on_chutes
    
#     session = await get_async_client()
#     start_call, next_call = _build_engine_calls(session, utterance_engine_url)
#     dialogues = {}  # dialogue_uid -> List[BBPredictedUtterance]
#     consecutive_prediction_errors = 0  # Track consecutive prediction failures
    
#     try:
#         session_complete = False
#         current_dialogue_uid = None
#         context_memory = ""  # Accumulates full dialogue history
        
#         # Start the session (which can contain multiple dialogues) with retry
#         start_data = await retry_with_exponential_backoff(start_call, max_retries=3, initial_delay=1.0)
        
#         # Check if session is immediately done (no dialogues)
#         if start_data.get("done", False):
#             logger.info("Session immediately complete - no dialogues")
#             return dialogues
        
#         # Get first token and session info
#         session_id = start_data["session_id"]
#         first_token = start_data.get("word", start_data.get("token"))
#         utterance_index = start_data.get("utterance_index", 0)
#         dialogue_uid = start_data.get("dialogue_uid")
#         challenge_uid = start_data.get("challenge_uid")
        
#         logger.info(f"Started session {session_id}, first dialogue {dialogue_uid}")
        
#         # Process the first token and continue until session is complete
#         current_token = first_token
#         current_utterance_tokens = []
#         current_utterance_index = utterance_index
#         first_token_processed = False  # Track if we've processed the first token
        
#         # Add the first token to the utterance tokens if it's not a boundary token
#         if _is_content_token(first_token):
#             current_utterance_tokens.append(first_token)
#             logger.debug(f"Added first token '{first_token}' to utterance {current_utterance_index}")
        
#         while not session_complete:
#             # Handle dialogue/utterance initialization
#             if dialogue_uid and dialogue_uid not in dialogues:
#                 dialogues[dialogue_uid] = []
#                 context_memory = ""  # Reset context for new dialogue
#                 current_dialogue_uid = dialogue_uid
#                 logger.info(f"Started new dialogue: {dialogue_uid}")
#             elif dialogue_uid != current_dialogue_uid:
#                 # Switched to different dialogue  
#                 context_memory = ""  # Reset context
#                 current_dialogue_uid = dialogue_uid
#                 logger.info(f"Switched to dialogue: {dialogue_uid}")
            
#             # Process current token
#             if _is_content_token(current_token):
#                 # Regular token - add to current utterance (skip if it's the first token already added)
#                 if not (current_token == first_token and not first_token_processed):
#                     current_utterance_tokens.append(current_token)
#                     logger.debug(f"Added token '{current_token}' to utterance {current_utterance_index}")
                
#                 # Mark first token as processed after we handle it
#                 if current_token == first_token:
#                     first_token_processed = True
                
#                 # Get prediction for complete utterance if we have a chute
#                 prediction_text = ""
#                 if chute_slug and current_utterance_tokens:
#                     try:
#                         prefix_text = " ".join(current_utterance_tokens)
#                         payload = BBPredictedUtterance(
#                             index=session_id,
#                             step=len(current_utterance_tokens),
#                             prefix=prefix_text,
#                             prediction="",  # Will be filled by chute
#                             context=context_memory,  # context is a string containing dialogue history
#                         )
                        
#                         result = await call_miner_model_on_chutes(slug=chute_slug, 
#                                                                   payload=payload, 
#                                                                   context_used=context_memory,
#                                                                   timeout=timeout)
                        
#                         if result.success and result.utterance:
#                             prediction_text = result.utterance.prediction
#                             consecutive_prediction_errors = 0  # Reset error counter on success
#                             logger.debug(f"Chute predicted: '{prediction_text}' for prefix '{prefix_text}'")
                            
#                             # Save this prediction step to dialogues
#                             step_utterance = BBPredictedUtterance(
#                                 index=session_id,
#                                 step=len(current_utterance_tokens) - 1,  # Step is 0-indexed
#                                 prefix=prefix_text,
#                                 prediction=prediction_text,
#                                 context=context_memory,
#                                 done=False,  # Not done yet, more tokens may come
#                             )
#                             if current_dialogue_uid:
#                                 dialogues[current_dialogue_uid].append(step_utterance)
                            
#                             # Log prediction event
#                             if challenge_logger:
#                                 challenge_logger.log_predicted_event(
#                                     utterance_index=current_utterance_index,
#                                     step=len(current_utterance_tokens) - 1,  # Step is 0-indexed
#                                     prefix=prefix_text,
#                                     prediction=prediction_text,
#                                     context_used=context_memory.split(" EOF ") if context_memory else []
#                                 )
#                         else:
#                             consecutive_prediction_errors += 1
#                             logger.warning(
#                                 f"Chute prediction failed - {consecutive_prediction_errors} consecutive errors "
#                                 f"(max: {max_prediction_errors}): {result.error}"
#                             )
#                             logger.debug(
#                                 "Chute fail details: slug=%s utt_idx=%d tokens=%d prefix='%s' ctx_len=%d err=%s",
#                                 chute_slug,
#                                 current_utterance_index,
#                                 len(current_utterance_tokens),
#                                 prefix_text[:120] if prefix_text else "",
#                                 len(context_memory.split(" EOF ")) if context_memory else 0,
#                                 str(result.error)[:200] if result.error else None,
#                             )
                            
#                             if consecutive_prediction_errors >= max_prediction_errors:
#                                 error_msg = (
#                                     f"Interrupting miner {chute_slug}: {max_prediction_errors} consecutive prediction errors. "
#                                     f"Last error: {result.error}"
#                                 )
#                                 logger.error(error_msg)
#                                 raise BabelbitUtteranceError(error_msg)
                            
#                     except BabelbitUtteranceError:
#                         # Re-raise interruption errors
#                         raise
#                     except Exception as e:
#                         consecutive_prediction_errors += 1
#                         logger.error(
#                             f"Error calling chute - {consecutive_prediction_errors} consecutive errors "
#                             f"(max: {max_prediction_errors}): {e}"
#                         )
#                         try:
#                             logger.debug(
#                                 "Chute exception details: slug=%s utt_idx=%d tokens=%d prefix='%s' ctx_len=%d err_type=%s err='%s'",
#                                 chute_slug,
#                                 current_utterance_index,
#                                 len(current_utterance_tokens),
#                                 prefix_text[:120] if prefix_text else "",
#                                 len(context_memory.split(" EOF ")) if context_memory else 0,
#                                 type(e).__name__,
#                                 str(e)[:300],
#                             )
#                         except Exception:
#                             pass
                        
#                         if consecutive_prediction_errors >= max_prediction_errors:
#                             error_msg = (
#                                 f"Interrupting miner {chute_slug}: {max_prediction_errors} consecutive prediction errors. "
#                                 f"Last error: {e}"
#                             )
#                             logger.error(error_msg)
#                             raise BabelbitUtteranceError(error_msg)
                        
#             elif current_token == "EOF":
#                 # End of utterance - mark the last step with ground truth and done=True
#                 if current_utterance_tokens and current_dialogue_uid:
#                     ground_truth_text = " ".join(current_utterance_tokens)
                    
#                     # Update the last prediction step to include ground truth and done=True
#                     if dialogues.get(current_dialogue_uid):
#                         # Find all steps for this utterance and mark the last one as done
#                         last_step = dialogues[current_dialogue_uid][-1]
#                         last_step.ground_truth = ground_truth_text
#                         last_step.done = True
                        
#                         # Log utterance completion event
#                         if challenge_logger:
#                             challenge_logger.log_utterance_complete_event(
#                                 utterance_index=current_utterance_index,
#                                 ground_truth=ground_truth_text,
#                                 final_prediction=last_step.prediction or ""
#                             )
                    
#                     # Update context with completed utterance, separated by EOF
#                     if context_memory:
#                         context_memory += f" EOF {ground_truth_text}"
#                     else:
#                         context_memory = ground_truth_text
                    
#                     logger.info(f"Completed utterance {current_utterance_index} with {len(current_utterance_tokens)} tokens")
                
#                 # Reset for next utterance
#                 current_utterance_tokens = []
#                 current_utterance_index += 1
                
#             elif current_token == "EOF EOF":
#                 # End of dialogue - already handled utterance completion above if needed
#                 logger.info(f"Completed dialogue {current_dialogue_uid}")
#                 current_utterance_index = 0  # Reset for next dialogue
            
#             # Get next token with retry
#             prediction = prediction_text if 'prediction_text' in locals() else ""
#             next_data = await retry_with_exponential_backoff(
#                 next_call, 
#                 session_id, 
#                 prediction,
#                 max_retries=3, 
#                 initial_delay=1.0
#             )
            
#             # Check if session is complete
#             session_complete = next_data.get("done", False)
#             if session_complete:
#                 logger.info("Session completed")
#                 break
            
#             # Update current token and metadata
#             current_token = next_data.get("word", next_data.get("token"))
#             dialogue_uid = next_data.get("dialogue_uid", dialogue_uid)
#             utterance_index = next_data.get("utterance_index", utterance_index)
#             current_utterance_index = utterance_index
            
#             # Log revealed event
#             if challenge_logger and current_token:
#                 is_done = current_token in BOUNDARY_TOKENS
#                 revealed_token = None if current_token in BOUNDARY_TOKENS else current_token
                
#                 challenge_logger.log_revealed_event(
#                     utterance_index=current_utterance_index,
#                     step=len(current_utterance_tokens),
#                     revealed_next=revealed_token,
#                     done=is_done
#                 )
                
#     except Exception as e:
#         logger.error(f"Error during utterance engine interaction: {e}")
#         raise UtteranceEngineError(f"Utterance engine interaction failed: {e}")
    
#     total_utterances = sum(len(utterances) for utterances in dialogues.values())
#     logger.info(f"Retrieved {len(dialogues)} dialogues with {total_utterances} total utterances")
#     return dialogues


# async def interact_with_utterance_engine_using_chute(
#     utterance_engine_url: str,
#     chute_slug: Optional[str] = None
# ) -> Dict[str, List[BBPredictedUtterance]]:
#     """
#     Interact with utterance engine using a chute for predictions.
    
#     Args:
#         utterance_engine_url: URL of the utterance engine
#         chute_slug: Slug of the chute to use for predictions
        
#     Returns:
#         Dict mapping dialogue_uid to List of dialogue turns with chute predictions
        
#     Example:
#         from babelbit.utils.predict import create_chute_prediction_callback
        
#         # Get prediction callback
#         predictor = await create_chute_prediction_callback("my-model-slug")
        
#         # Use with utterance engine
#         dialogues = await interact_with_utterance_engine(
#             "http://localhost:8000", 
#             predictor
#         )
#     """
#     logger.info(f"Starting prediction with utterance engine at {utterance_engine_url} using chute {chute_slug}")
#     return await predict_with_utterance_engine(utterance_engine_url, chute_slug)


async def predict_with_utterance_engine_multi_miner(
    utterance_engine_url: str,
    miners: List[Any],
    prediction_callback: Callable[[Any, BBPredictedUtterance, str], Awaitable[str]],
    timeout: float = 10.0,
    max_prediction_errors: int = 5,
    subtensor: Optional[Any] = None,
    step_block_modulo: int = 2,
    *,
    solo: bool = False,
    miner_key_fn: Optional[Callable[[Any], str]] = None,
    return_challenge_uid: bool = False,
    return_miner_status: bool = False,
) -> Union[
    Dict[str, Dict[str, List[BBPredictedUtterance]]],
    Tuple[Dict[str, Dict[str, List[BBPredictedUtterance]]], Optional[str]],
    Tuple[Dict[str, Dict[str, List[BBPredictedUtterance]]], Dict[str, bool]],
    Tuple[Dict[str, Dict[str, List[BBPredictedUtterance]]], Optional[str], Dict[str, bool]],
]:
    """
    Interact with the utterance engine API using multiple miners sharing a single session.
    
    Iterates through utterance engine steps first, then queries all miners in parallel for each step.
    This ensures all miners see the same challenge in the same order.
    
    Block Synchronization: If subtensor is provided and step_block_modulo > 0, each step waits
    until the next block that is a multiple of step_block_modulo. This ensures all validators
    progress through steps at the same block-synchronized pace.
    
    NOTE: Each validator gets its own independent session from the utterance engine.
    If multiple validators need to evaluate the same challenge simultaneously, the utterance
    engine itself must handle session/challenge coordination (e.g., time-based challenge windows).
    This function does NOT coordinate between different validator instances.
    
    Args:
        utterance_engine_url: Base URL of the utterance engine (e.g., "http://localhost:8000")
        miners: List of miner objects (with .slug attribute)
        prediction_callback: Async function(miner, payload, context) -> prediction_text
        timeout: Timeout for miner predictions in seconds
        max_prediction_errors: Maximum consecutive prediction errors per miner before skipping
        subtensor: Optional subtensor instance for block synchronization
        step_block_modulo: If > 0 and subtensor provided, wait for blocks divisible by this number
        solo: If True, use the solo challenge endpoints (/solo/start, /solo/next)
        miner_key_fn: Optional function to derive a miner key; defaults to miner.hotkey
        return_challenge_uid: If True, also return the challenge_uid with the dialogues

    Returns:
        Dict mapping miner_slug to Dict[dialogue_uid -> List[BBPredictedUtterance]], optionally paired with
        the challenge_uid when return_challenge_uid is True.
        
    Raises:
        UtteranceEngineError: If the API interaction fails
    """
    session = await get_async_client()
    start_call, next_call = _build_engine_calls(session, utterance_engine_url, solo=solo)

    # Helper to get unique identifier for a miner (use hotkey since it exists for all miners)
    def get_miner_key(miner):
        if miner_key_fn:
            return miner_key_fn(miner)
        return miner.hotkey
    
    async def timed_prediction(miner, payload, context):
        start = perf_counter()
        try:
            logger.debug(f"Starting timed prediction for miner {get_miner_key(miner)}")
            pred = await prediction_callback(miner, payload, context)
        except Exception as e:
            logger.warning(f"Prediction error for miner {get_miner_key(miner)}: {e}, finished in {perf_counter() - start:.2f} seconds")
            return e, perf_counter() - start
        finally:
            duration = perf_counter() - start
            logger.debug(f"Timed prediction for miner {get_miner_key(miner)} took {duration:.2f} seconds")
        return pred, duration

    # Initialize tracking structures for all miners
    miner_dialogues = {get_miner_key(m): {} for m in miners}  # miner_key -> {dialogue_uid -> [steps]}
    miner_contexts = {get_miner_key(m): {} for m in miners}   # miner_key -> {dialogue_uid -> context_str}
    miner_error_counts = {get_miner_key(m): 0 for m in miners}  # Track consecutive errors per miner
    miner_active = {get_miner_key(m): True for m in miners}  # Track which miners are still active
    
    try:
        session_complete = False
        current_dialogue_uid = None
        
        # Start the session (shared by all miners) with retry
        start_data = await retry_with_exponential_backoff(start_call, max_retries=3, initial_delay=1.0)
        challenge_uid = start_data.get("challenge_uid")
        
        # Check if session is immediately done (no dialogues)
        if start_data.get("done", False):
            logger.info("Session immediately complete - no dialogues")
            return miner_dialogues
        
        # Get first token and session info
        session_id = start_data["session_id"]
        first_token = start_data.get("word", start_data.get("token"))
        utterance_index = start_data.get("utterance_index", 0)
        dialogue_uid = start_data.get("dialogue_uid")
        
        logger.info(f"Started shared session {session_id} for {len(miners)} miners, first dialogue {dialogue_uid}")
        
        # Process the first token and continue until session is complete
        current_token = first_token
        current_utterance_tokens = []
        current_utterance_index = utterance_index
        
        while not session_complete:
            # Handle dialogue/utterance initialization
            if dialogue_uid and dialogue_uid != current_dialogue_uid:
                current_dialogue_uid = dialogue_uid
                logger.info(f"Processing dialogue: {dialogue_uid}")
                
                # Initialize dialogue tracking for all miners
                for miner in miners:
                    miner_key = get_miner_key(miner)
                    _ensure_dialogue_tracking(
                        miner_dialogues[miner_key],
                        miner_contexts[miner_key],
                        dialogue_uid,
                    )
            
            # Process current token
            if _is_content_token(current_token):
                # Regular token - add to current utterance
                current_utterance_tokens.append(current_token)
                step_num = len(current_utterance_tokens)
                
                # Block-based step synchronization: Wait for next block boundary, then compute within N blocks
                step_start_block = None
                step_target_block = None
                if subtensor and step_block_modulo > 0:
                    try:
                        # First, wait for the next sync block boundary
                        await wait_until_block_modulo(subtensor, step_block_modulo)
                        step_start_block = await subtensor.get_current_block()
                        step_target_block = step_start_block + step_block_modulo
                        logger.info(f"[Utterance {current_utterance_index} Step {step_num}] Starting at block {step_start_block}, must complete by block {step_target_block}")
                    except Exception as e:
                        logger.warning(f"Block synchronization failed: {e}. Continuing without sync.")
                
                logger.info(f"[Utterance {current_utterance_index} Step {step_num}] Token: '{current_token}' - Querying {sum(1 for a in miner_active.values() if a)}/{len(miners)} active miners")
                
                # Check if all miners are deactivated - if so, stop processing this session
                active_count = sum(1 for a in miner_active.values() if a)
                if active_count == 0:
                    logger.warning(f"[Utterance {current_utterance_index} Step {step_num}] All miners deactivated, stopping session early")
                    
                    # Before breaking, save ground truth for any incomplete utterance
                    if current_utterance_tokens and current_dialogue_uid:
                        ground_truth_text = " ".join(current_utterance_tokens)
                        logger.info(f"[Utterance {current_utterance_index}] Saving ground truth for incomplete utterance: '{ground_truth_text}'")
                        
                        for miner in miners:
                            miner_key = get_miner_key(miner)
                            # Mark last step with ground truth for incomplete utterance
                            _finalize_utterance(
                                miner_dialogues[miner_key],
                                miner_contexts[miner_key],
                                current_dialogue_uid,
                                current_utterance_tokens,
                                mark_done=False,
                                update_context=False,
                            )
                            if miner_dialogues[miner_key].get(current_dialogue_uid):
                                logger.debug(
                                    "Set ground_truth for miner %s last step in dialogue %s",
                                    getattr(miner, "slug", None),
                                    current_dialogue_uid,
                                )
                    
                    break
                
                # Query ALL active miners in parallel for this step
                prefix_text = " ".join(current_utterance_tokens)
                
                # Create prediction tasks for all active miners
                prediction_tasks = []
                for miner in miners:
                    miner_key = get_miner_key(miner)
                    if not miner_active[miner_key]:
                        continue  # Skip inactive miners
                        
                    context = miner_contexts[miner_key].get(current_dialogue_uid, "")
                    
                    payload = BBPredictedUtterance(
                        index=session_id,
                        step=len(current_utterance_tokens) - 1,  # 0-indexed
                        prefix=prefix_text,
                        prediction="",  # Will be filled by callback
                        context=context,
                        done=False,
                    )
                    
                    # Create async task for each miner
                    task = asyncio.create_task(timed_prediction(miner, payload, context))
                    prediction_tasks.append((miner, task))
                
                # Gather all predictions (with error handling per miner)
                # Calculate timeout based on block deadline
                # Use step_block_modulo directly since we just synced to the start block
                if step_target_block and subtensor and step_block_modulo > 0:
                    try:
                        # We just synced to the start block, so we have exactly step_block_modulo blocks
                        # Use 90% of available time to ensure we finish before target block
                        block_timeout = step_block_modulo * 12 * 0.9
                        effective_timeout = block_timeout
                        logger.debug(f"[Utterance {current_utterance_index} Step {step_num}] Using timeout {effective_timeout:.1f}s ({step_block_modulo} blocks allocated)")
                    except Exception:
                        effective_timeout = timeout
                else:
                    effective_timeout = timeout
                
                done, pending = await asyncio.wait([t for _, t in prediction_tasks],
                                                   timeout=effective_timeout)
                
                result_map: dict[asyncio.Task, tuple[Any, float | None]] = {}
                for t in done:
                    try:
                        result_map[t] = t.result()            # (prediction, duration)
                    except Exception as e:
                        result_map[t] = (e, None)

                # mark only pending tasks as timed out
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for t in pending:
                    result_map[t] = (asyncio.TimeoutError(f"Prediction timeout after {effective_timeout:.1f}s"), None)

                all_run_times = []
                for miner, task in prediction_tasks:
                    miner_id = miner.slug or f"uid_{miner.uid}"
                    miner_key = get_miner_key(miner)
                    result, run_time = result_map.get(task, (RuntimeError("Missing prediction result"), None))
                    try:
                        if isinstance(result, Exception):
                            raise result
                        
                        prediction_text = result if isinstance(result, str) else ""
                        all_run_times.append(run_time)
                        # Success - reset error counter
                        if miner_error_counts[miner_key] > 0:
                            logger.info(f"[Utterance {current_utterance_index} Step {step_num}] Miner {miner_id} recovered after {miner_error_counts[miner_key]} errors")
                            miner_error_counts[miner_key] = 0
                        
                        # Store this prediction step
                        step_utterance = BBPredictedUtterance(
                            index=session_id,
                            step=len(current_utterance_tokens) - 1,
                            prefix=prefix_text,
                            prediction=prediction_text,
                            context=miner_contexts[miner_key].get(current_dialogue_uid, ""),
                            done=False,
                        )
                        miner_dialogues[miner_key][current_dialogue_uid].append(step_utterance)
                        
                        logger.debug(f"[Utterance {current_utterance_index} Step {step_num}] Miner {miner_id} predicted: '{prediction_text[:50]}...'")
                        
                    except Exception as e:
                        # Handle per-miner errors
                        miner_error_counts[miner_key] += 1
                        
                        # Only log warning every 5 errors to reduce noise, or when approaching limit
                        if (miner_error_counts[miner_key] % 5 == 1 or 
                            miner_error_counts[miner_key] >= max_prediction_errors - 2):
                            error_msg = str(e)
                            # Truncate very long error messages
                            if len(error_msg) > 200:
                                error_msg = error_msg[:200] + "..."
                            logger.warning(
                                f"[Utterance {current_utterance_index} Step {step_num}] Miner {miner_id} prediction failed "
                                f"({miner_error_counts[miner_key]}/{max_prediction_errors}): {error_msg}"
                            )
                        # Always emit a debug line with rich failure context
                        try:
                            logger.debug(
                                "[Utterance %d Step %d] Miner fail: miner_id=%s hk=%s dlg=%s errs=%d/%d err_type=%s err='%s' prefix='%s'",
                                current_utterance_index,
                                step_num,
                                miner_id,
                                miner.hotkey[:16] + "..." if getattr(miner, 'hotkey', None) else None,
                                current_dialogue_uid,
                                miner_error_counts[miner_key],
                                max_prediction_errors,
                                type(e).__name__,
                                str(e)[:300],
                                prefix_text[:120] if prefix_text else "",
                            )
                        except Exception:
                            pass
                        
                        if miner_error_counts[miner_key] >= max_prediction_errors:
                            miner_active[miner_key] = False
                            logger.error(
                                f"[Utterance {current_utterance_index} Step {step_num}] Deactivating miner {miner_id} after {max_prediction_errors} consecutive errors"
                            )
                            try:
                                deactivated = sum(1 for a in miner_active.values() if not a)
                                logger.debug(
                                    "[Utterance %d Step %d] Deactivated miner %s (total deactivated=%d/%d)",
                                    current_utterance_index,
                                    step_num,
                                    miner_id,
                                    deactivated,
                                    len(miners),
                                )
                            except Exception:
                                pass
                        
                        # Store empty prediction for failed step
                        step_utterance = BBPredictedUtterance(
                            index=session_id,
                            step=len(current_utterance_tokens) - 1,
                            prefix=prefix_text,
                            prediction="",
                            context=miner_contexts[miner_key].get(current_dialogue_uid, ""),
                            done=False,
                        )
                        miner_dialogues[miner_key][current_dialogue_uid].append(step_utterance)
                
                avg_run_time = sum(all_run_times) / len(all_run_times) if all_run_times else 0.0
                max_run_time = max(all_run_times) if all_run_times else 0.0
                min_run_time = min(all_run_times) if all_run_times else 0.0
                logger.info(f"[Utterance {current_utterance_index} Step {step_num}] "
                            f"Completed miner predictions in avg {avg_run_time:.2f}s "
                            f"(min {min_run_time:.2f}s, max {max_run_time:.2f}s) "
                            f"over {len(all_run_times)} miners")

                # Wait for target block if we finished early
                if step_target_block and subtensor:
                    try:
                        current_block = await subtensor.get_current_block()
                        if current_block < step_target_block:
                            blocks_to_wait = step_target_block - current_block
                            logger.info(f"[Utterance {current_utterance_index} Step {step_num}] Finished early, waiting {blocks_to_wait} blocks until {step_target_block}")
                            while True:
                                await asyncio.sleep(6)  # Check every ~half block
                                current_block = await subtensor.get_current_block()
                                if current_block >= step_target_block:
                                    logger.info(f"[Utterance {current_utterance_index} Step {step_num}] Reached target block {current_block}")
                                    break
                        else:
                            logger.info(f"[Utterance {current_utterance_index} Step {step_num}] Completed at block {current_block} (target was {step_target_block})")
                    except Exception as e:
                        logger.warning(f"Failed to wait for target block: {e}")
            
            elif current_token == "EOF":
                # End of utterance - update all miners' contexts and mark done
                if current_utterance_tokens and current_dialogue_uid:
                    for miner in miners:
                        miner_key = get_miner_key(miner)
                        _finalize_utterance(
                            miner_dialogues[miner_key],
                            miner_contexts[miner_key],
                            current_dialogue_uid,
                            current_utterance_tokens,
                        )
                    
                    logger.info(f"Completed utterance {current_utterance_index} with {len(current_utterance_tokens)} tokens")
                
                # Reset for next utterance
                current_utterance_tokens = []
                current_utterance_index += 1
                
            elif current_token == "EOF EOF":
                # End of dialogue
                logger.info(f"Completed dialogue {current_dialogue_uid}")
                current_utterance_index = 0  # Reset for next dialogue
            
            # Get next token with retry (always use empty prediction)
            next_data = await retry_with_exponential_backoff(
                next_call, 
                session_id,
                "",  # Empty string - utterance engine doesn't use it
                max_retries=3, 
                initial_delay=1.0
            )
            
            # Check if session is complete
            session_complete = next_data.get("done", False)
            if session_complete:
                logger.info("Session completed")
                break
            
            # Update current token and metadata
            current_token = next_data.get("word", next_data.get("token"))
            dialogue_uid = next_data.get("dialogue_uid", dialogue_uid)
            utterance_index = next_data.get("utterance_index", utterance_index)
            current_utterance_index = utterance_index
                
    except Exception as e:
        logger.error(f"Error during multi-miner utterance engine interaction: {e}")
        raise UtteranceEngineError(f"Utterance engine interaction failed: {e}")
    
    # Log summary
    total_active = sum(1 for active in miner_active.values() if active)
    total_dialogues = sum(len(dialogues) for dialogues in miner_dialogues.values())
    logger.info(
        f"Completed session: {total_active}/{len(miners)} miners active, "
        f"{total_dialogues} total dialogue sets collected"
    )
    
    if return_challenge_uid and return_miner_status:
        return miner_dialogues, challenge_uid, miner_active
    if return_challenge_uid:
        return miner_dialogues, challenge_uid
    if return_miner_status:
        return miner_dialogues, miner_active
    return miner_dialogues


async def predict_solo_challenge_for_miners(
    utterance_engine_url: str,
    miners: List[Any],
    prediction_callback: Callable[[Any, BBPredictedUtterance, str], Awaitable[str]],
    timeout: float = 10.0,
    max_prediction_errors: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Run a single solo challenge session shared by all miners (lock-step).
    Even though the session is shared per validator, miners cannot pre-cache because
    each validator receives a different session from the engine.
    """
    if not miners:
        logger.info("[Solo Challenge] No miners to evaluate")
        return {}

    def _miner_key(miner: Any) -> str:
        return (
            getattr(miner, "hotkey", None)
        or getattr(miner, "slug", None)
        or f"uid_{getattr(miner, 'uid', '?')}"
    )

    shared_dialogues, challenge_uid, miner_status = await predict_with_utterance_engine_multi_miner(
        utterance_engine_url,
        miners,
        prediction_callback,
        timeout=timeout,
        max_prediction_errors=max_prediction_errors,
        subtensor=None,
        step_block_modulo=0,
        solo=True,
        miner_key_fn=_miner_key,
        return_challenge_uid=True,
        return_miner_status=True,
    )

    return {
        miner_key: {"challenge_uid": challenge_uid, "dialogues": dialogues}
        for miner_key, dialogues in shared_dialogues.items()
        if miner_status.get(miner_key, True)
    }


# async def simple_utterance_engine_test(base_url: str) -> Dict[str, List[BBPredictedUtterance]]:
#     """
#     Simple test function that retrieves dialogues without making predictions.
    
#     Args:
#         base_url: Base URL of the utterance engine
        
#     Returns:
#         Dict mapping dialogue_uid to List of BBPredictedUtterance objects
#     """
#     return await predict_with_utterance_engine(base_url, None)
