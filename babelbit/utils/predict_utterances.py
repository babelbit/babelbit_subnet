from logging import getLogger
from time import time
from typing import Any, Callable, Optional, Union, Awaitable, List, Dict
from inspect import iscoroutinefunction
from hashlib import sha256
from json import dumps
from random import shuffle, randint

from aiohttp import ClientResponseError

from babelbit.utils.settings import get_settings
from babelbit.utils.bittensor_helpers import load_hotkey_keypair
from babelbit.utils.signing import sign_message
from babelbit.utils.async_clients import get_async_client
from babelbit.utils.utterance_auth import get_auth_headers
from babelbit.chute_template.schemas import BBPredictedUtterance

logger = getLogger(__name__)


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
    session = await get_async_client()
    headers = await get_auth_headers()
    
    try:
        async with session.get(f"{utterance_engine_url}/start", headers=headers) as response:
            if response.status != 200:
                raise UtteranceEngineError(f"Failed to get challenge ID: HTTP {response.status}")
            
            start_data = await response.json()
            challenge_uid = start_data.get("challenge_uid")
            
            logger.info(f"Current challenge ID: {challenge_uid}")
            return challenge_uid
            
    except Exception as e:
        logger.error(f"Error getting challenge ID: {e}")
        raise UtteranceEngineError(f"Failed to get challenge ID: {e}")


async def predict_with_utterance_engine(
    utterance_engine_url: str,
    chute_slug: Optional[str] = None,
    challenge_logger: Optional[Any] = None,
    timeout: float = 10.0
) -> Dict[str, List[BBPredictedUtterance]]:
    """
    Interact with the utterance engine API to get complete dialogues using a chute for predictions.
    
    Args:
        utterance_engine_url: Base URL of the utterance engine (e.g., "http://localhost:8000")
        chute_slug: Optional chute slug to use for predictions. If None, uses empty predictions.
        challenge_logger: Optional logger for recording step-by-step interaction events
    
    Returns:
        Dict mapping dialogue_uid to List of BBPredictedUtterance objects for each dialogue
    
    Raises:
        UtteranceEngineError: If the API interaction fails
    """
    from babelbit.utils.predict_engine import call_miner_model_on_chutes
    
    session = await get_async_client()
    headers = await get_auth_headers()
    dialogues = {}  # dialogue_uid -> List[BBPredictedUtterance]
    
    try:
        session_complete = False
        current_dialogue_uid = None
        context_memory = ""  # Accumulates full dialogue history
        
        # Start the session (which can contain multiple dialogues)
        async with session.get(f"{utterance_engine_url}/start", headers=headers) as response:
            if response.status != 200:
                raise UtteranceEngineError(f"Failed to start session: HTTP {response.status}")
            
            start_data = await response.json()
            
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
                            logger.debug(f"Chute predicted: '{prediction_text}' for prefix '{prefix_text}'")
                            
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
                            logger.warning(f"Chute prediction failed: {result.error}")
                            
                    except Exception as e:
                        logger.error(f"Error calling chute: {e}")
                        
            elif current_token == "EOF":
                # End of utterance - create utterance object
                if current_utterance_tokens:
                    utterance = BBPredictedUtterance(
                        index=session_id,
                        step=current_utterance_index,
                        prefix=" ".join(current_utterance_tokens[:-1]) if len(current_utterance_tokens) > 1 else "",
                        prediction=current_utterance_tokens[-1] if current_utterance_tokens else "",
                        ground_truth=" ".join(current_utterance_tokens),
                        done=True,
                    )
                    
                    if current_dialogue_uid:
                        dialogues[current_dialogue_uid].append(utterance)
                        
                        # Update context with completed utterance, separated by EOF
                        utterance_text = " ".join(current_utterance_tokens)
                        if context_memory:
                            context_memory += f" EOF {utterance_text}"
                        else:
                            context_memory = utterance_text
                        
                        # Log utterance completion event
                        if challenge_logger:
                            challenge_logger.log_utterance_complete_event(
                                utterance_index=current_utterance_index,
                                ground_truth=utterance.ground_truth or "",
                                final_prediction=utterance.prediction or ""
                            )
                    
                    logger.info(f"Completed utterance {current_utterance_index} with {len(current_utterance_tokens)} tokens")
                
                # Reset for next utterance
                current_utterance_tokens = []
                current_utterance_index += 1
                
            elif current_token == "EOF EOF":
                # End of dialogue - already handled utterance completion above if needed
                logger.info(f"Completed dialogue {current_dialogue_uid}")
                current_utterance_index = 0  # Reset for next dialogue
            
            # Get next token
            next_payload = {
                "session_id": session_id,
                "prediction": prediction_text if 'prediction_text' in locals() else ""
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
                
                next_data = await next_response.json()
                
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

