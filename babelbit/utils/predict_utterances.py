from logging import getLogger
from time import time
from typing import Any, Callable, Optional, Union, Awaitable, List, Dict
from inspect import iscoroutinefunction
from hashlib import sha256
from json import dumps
from random import shuffle, randint
import asyncio

from aiohttp import ClientResponseError

from babelbit.utils.settings import get_settings
from babelbit.utils.bittensor_helpers import load_hotkey_keypair
from babelbit.utils.signing import sign_message
from babelbit.utils.async_clients import get_async_client
from babelbit.utils.utterance_auth import get_auth_headers
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
        headers = await get_auth_headers()
        
        async with session.get(f"{utterance_engine_url}/start", headers=headers) as response:
            if response.status != 200:
                raise UtteranceEngineError(f"Failed to get challenge ID: HTTP {response.status}")
            
            start_data = await response.json()
            challenge_uid = start_data.get("challenge_uid")
            
            logger.info(f"Current challenge ID: {challenge_uid}")
            return challenge_uid
    
    try:
        return await retry_with_exponential_backoff(_get_challenge, max_retries=3, initial_delay=1.0)
    except Exception as e:
        logger.error(f"Error getting challenge ID after retries: {e}")
        raise UtteranceEngineError(f"Failed to get challenge ID: {e}")


async def predict_with_utterance_engine(
    utterance_engine_url: str,
    chute_slug: Optional[str] = None,
    challenge_logger: Optional[Any] = None,
    timeout: float = 10.0,
    max_prediction_errors: int = 5
) -> Dict[str, List[BBPredictedUtterance]]:
    """
    Interact with the utterance engine API to get complete dialogues using a chute for predictions.
    
    Args:
        utterance_engine_url: Base URL of the utterance engine (e.g., "http://localhost:8000")
        chute_slug: Optional chute slug to use for predictions. If None, uses empty predictions.
        challenge_logger: Optional logger for recording step-by-step interaction events
        timeout: Timeout for chute predictions in seconds
        max_prediction_errors: Maximum consecutive prediction errors before interrupting (default: 5)
    
    Returns:
        Dict mapping dialogue_uid to List of BBPredictedUtterance objects for each dialogue
    
    Raises:
        UtteranceEngineError: If the API interaction fails
        BabelbitUtteranceError: If too many consecutive prediction errors occur
    """
    from babelbit.utils.predict_engine import call_miner_model_on_chutes
    
    session = await get_async_client()
    headers = await get_auth_headers()
    dialogues = {}  # dialogue_uid -> List[BBPredictedUtterance]
    consecutive_prediction_errors = 0  # Track consecutive prediction failures
    
    async def _call_start():
        """Helper to call /start endpoint with retry logic"""
        async with session.get(f"{utterance_engine_url}/start", headers=headers) as response:
            if response.status != 200:
                raise UtteranceEngineError(f"Failed to start session: HTTP {response.status}")
            return await response.json()
    
    async def _call_next(session_id: str, prediction: str):
        """Helper to call /next endpoint with retry logic"""
        next_payload = {
            "session_id": session_id,
            "prediction": prediction
        }
        
        async with session.post(
            f"{utterance_engine_url}/next", 
            json=next_payload,
            headers=headers
        ) as next_response:
            if next_response.status != 200:
                error_data = await next_response.json()
                if error_data.get("error") == "invalid or missing session_id":
                    raise UtteranceEngineError(f"Invalid session: {session_id}")
                else:
                    raise UtteranceEngineError(f"Failed to get next token: HTTP {next_response.status}")
            
            return await next_response.json()
    
    try:
        session_complete = False
        current_dialogue_uid = None
        context_memory = ""  # Accumulates full dialogue history
        
        # Start the session (which can contain multiple dialogues) with retry
        start_data = await retry_with_exponential_backoff(_call_start, max_retries=3, initial_delay=1.0)
        
        # Check if session is immediately done (no dialogues)
        if start_data.get("done", False):
            logger.info("Session immediately complete - no dialogues")
            return dialogues
        
        # Get first token and session info
        session_id = start_data["session_id"]
        first_token = start_data.get("word", start_data.get("token"))
        utterance_index = start_data.get("utterance_index", 0)
        dialogue_uid = start_data.get("dialogue_uid")
        challenge_uid = start_data.get("challenge_uid")
        
        logger.info(f"Started session {session_id}, first dialogue {dialogue_uid}")
        
        # Process the first token and continue until session is complete
        current_token = first_token
        current_utterance_tokens = []
        current_utterance_index = utterance_index
        
        while not session_complete:
            # Handle dialogue/utterance initialization
            if dialogue_uid and dialogue_uid not in dialogues:
                dialogues[dialogue_uid] = []
                context_memory = ""  # Reset context for new dialogue
                current_dialogue_uid = dialogue_uid
                logger.info(f"Started new dialogue: {dialogue_uid}")
            elif dialogue_uid != current_dialogue_uid:
                # Switched to different dialogue  
                context_memory = ""  # Reset context
                current_dialogue_uid = dialogue_uid
                logger.info(f"Switched to dialogue: {dialogue_uid}")
            
            # Process current token
            if current_token and current_token not in ("EOF", "EOF EOF"):
                # Regular token - add to current utterance
                current_utterance_tokens.append(current_token)
                logger.debug(f"Added token '{current_token}' to utterance {current_utterance_index}")
                
                # Get prediction for complete utterance if we have a chute
                prediction_text = ""
                if chute_slug and current_utterance_tokens:
                    try:
                        prefix_text = " ".join(current_utterance_tokens)
                        payload = BBPredictedUtterance(
                            index=session_id,
                            step=len(current_utterance_tokens),
                            prefix=prefix_text,
                            prediction="",  # Will be filled by chute
                            context=context_memory,  # context is a string containing dialogue history
                        )
                        
                        result = await call_miner_model_on_chutes(slug=chute_slug, 
                                                                  payload=payload, 
                                                                  context_used=context_memory,
                                                                  timeout=timeout)
                        
                        if result.success and result.utterance:
                            prediction_text = result.utterance.prediction
                            consecutive_prediction_errors = 0  # Reset error counter on success
                            logger.debug(f"Chute predicted: '{prediction_text}' for prefix '{prefix_text}'")
                            
                            # Save this prediction step to dialogues
                            step_utterance = BBPredictedUtterance(
                                index=session_id,
                                step=len(current_utterance_tokens) - 1,  # Step is 0-indexed
                                prefix=prefix_text,
                                prediction=prediction_text,
                                context=context_memory,
                                done=False,  # Not done yet, more tokens may come
                            )
                            if current_dialogue_uid:
                                dialogues[current_dialogue_uid].append(step_utterance)
                            
                            # Log prediction event
                            if challenge_logger:
                                challenge_logger.log_predicted_event(
                                    utterance_index=current_utterance_index,
                                    step=len(current_utterance_tokens) - 1,  # Step is 0-indexed
                                    prefix=prefix_text,
                                    prediction=prediction_text,
                                    context_used=context_memory.split(" EOF ") if context_memory else []
                                )
                        else:
                            consecutive_prediction_errors += 1
                            logger.warning(
                                f"Chute prediction failed - {consecutive_prediction_errors} consecutive errors "
                                f"(max: {max_prediction_errors}): {result.error}"
                            )
                            
                            if consecutive_prediction_errors >= max_prediction_errors:
                                error_msg = (
                                    f"Interrupting miner {chute_slug}: {max_prediction_errors} consecutive prediction errors. "
                                    f"Last error: {result.error}"
                                )
                                logger.error(error_msg)
                                raise BabelbitUtteranceError(error_msg)
                            
                    except BabelbitUtteranceError:
                        # Re-raise interruption errors
                        raise
                    except Exception as e:
                        consecutive_prediction_errors += 1
                        logger.error(
                            f"Error calling chute - {consecutive_prediction_errors} consecutive errors "
                            f"(max: {max_prediction_errors}): {e}"
                        )
                        
                        if consecutive_prediction_errors >= max_prediction_errors:
                            error_msg = (
                                f"Interrupting miner {chute_slug}: {max_prediction_errors} consecutive prediction errors. "
                                f"Last error: {e}"
                            )
                            logger.error(error_msg)
                            raise BabelbitUtteranceError(error_msg)
                        
            elif current_token == "EOF":
                # End of utterance - mark the last step with ground truth and done=True
                if current_utterance_tokens and current_dialogue_uid:
                    ground_truth_text = " ".join(current_utterance_tokens)
                    
                    # Update the last prediction step to include ground truth and done=True
                    if dialogues.get(current_dialogue_uid):
                        # Find all steps for this utterance and mark the last one as done
                        last_step = dialogues[current_dialogue_uid][-1]
                        last_step.ground_truth = ground_truth_text
                        last_step.done = True
                        
                        # Log utterance completion event
                        if challenge_logger:
                            challenge_logger.log_utterance_complete_event(
                                utterance_index=current_utterance_index,
                                ground_truth=ground_truth_text,
                                final_prediction=last_step.prediction or ""
                            )
                    
                    # Update context with completed utterance, separated by EOF
                    if context_memory:
                        context_memory += f" EOF {ground_truth_text}"
                    else:
                        context_memory = ground_truth_text
                    
                    logger.info(f"Completed utterance {current_utterance_index} with {len(current_utterance_tokens)} tokens")
                
                # Reset for next utterance
                current_utterance_tokens = []
                current_utterance_index += 1
                
            elif current_token == "EOF EOF":
                # End of dialogue - already handled utterance completion above if needed
                logger.info(f"Completed dialogue {current_dialogue_uid}")
                current_utterance_index = 0  # Reset for next dialogue
            
            # Get next token with retry
            prediction = prediction_text if 'prediction_text' in locals() else ""
            next_data = await retry_with_exponential_backoff(
                _call_next, 
                session_id, 
                prediction,
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
            
            # Log revealed event
            if challenge_logger and current_token:
                is_done = current_token in ("EOF", "EOF EOF")
                revealed_token = None if current_token in ("EOF", "EOF EOF") else current_token
                
                challenge_logger.log_revealed_event(
                    utterance_index=current_utterance_index,
                    step=len(current_utterance_tokens),
                    revealed_next=revealed_token,
                    done=is_done
                )
                
    except Exception as e:
        logger.error(f"Error during utterance engine interaction: {e}")
        raise UtteranceEngineError(f"Utterance engine interaction failed: {e}")
    
    total_utterances = sum(len(utterances) for utterances in dialogues.values())
    logger.info(f"Retrieved {len(dialogues)} dialogues with {total_utterances} total utterances")
    return dialogues


async def interact_with_utterance_engine_using_chute(
    utterance_engine_url: str,
    chute_slug: Optional[str] = None
) -> Dict[str, List[BBPredictedUtterance]]:
    """
    Interact with utterance engine using a chute for predictions.
    
    Args:
        utterance_engine_url: URL of the utterance engine
        chute_slug: Slug of the chute to use for predictions
        
    Returns:
        Dict mapping dialogue_uid to List of dialogue turns with chute predictions
        
    Example:
        from babelbit.utils.predict import create_chute_prediction_callback
        
        # Get prediction callback
        predictor = await create_chute_prediction_callback("my-model-slug")
        
        # Use with utterance engine
        dialogues = await interact_with_utterance_engine(
            "http://localhost:8000", 
            predictor
        )
    """
    logger.info(f"Starting prediction with utterance engine at {utterance_engine_url} using chute {chute_slug}")
    return await predict_with_utterance_engine(utterance_engine_url, chute_slug)


async def simple_utterance_engine_test(base_url: str) -> Dict[str, List[BBPredictedUtterance]]:
    """
    Simple test function that retrieves dialogues without making predictions.
    
    Args:
        base_url: Base URL of the utterance engine
        
    Returns:
        Dict mapping dialogue_uid to List of BBPredictedUtterance objects
    """
    return await predict_with_utterance_engine(base_url, None)

