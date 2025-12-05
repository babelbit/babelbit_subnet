"""
Challenge status tracking utilities.

Provides functions to check if a challenge has been processed by the runner,
preventing the validator from assigning default weights before scores are available.
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone
from logging import getLogger
from typing import Optional, Dict, Any

logger = getLogger(__name__)


def get_challenge_status_dir() -> Path:
    """Get the directory where challenge status files are stored."""
    status_dir = Path(os.getenv("BB_CHALLENGE_STATUS_DIR", "data/challenge_status"))
    status_dir.mkdir(parents=True, exist_ok=True)
    return status_dir


def mark_challenge_processed(
    challenge_uid: str,
    miner_count: int,
    total_dialogues: int,
    mean_score: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Mark a challenge as processed by the runner.
    
    This creates a status file that the validator can check before
    assigning default weights.
    
    Args:
        challenge_uid: The challenge ID that was processed
        miner_count: Number of miners that were evaluated
        total_dialogues: Total number of dialogues scored
        mean_score: Optional mean score across all miners
        metadata: Optional additional metadata to store
    """
    status_dir = get_challenge_status_dir()
    status_file = status_dir / f"{challenge_uid}.json"
    
    status_data = {
        "challenge_uid": challenge_uid,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "miner_count": miner_count,
        "total_dialogues": total_dialogues,
        "mean_score": mean_score,
        "metadata": metadata or {}
    }
    
    try:
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        logger.info(f"Marked challenge {challenge_uid} as processed ({miner_count} miners, {total_dialogues} dialogues)")
    except Exception as e:
        logger.warning(f"Failed to mark challenge {challenge_uid} as processed: {e}")


def is_challenge_processed(challenge_uid: str) -> bool:
    """
    Check if a challenge has been processed by the runner.
    
    Args:
        challenge_uid: The challenge ID to check
        
    Returns:
        True if the challenge has been processed, False otherwise
    """
    status_dir = get_challenge_status_dir()
    status_file = status_dir / f"{challenge_uid}.json"
    
    exists = status_file.exists()
    if exists:
        logger.debug(f"Challenge {challenge_uid} has been processed")
    else:
        logger.debug(f"Challenge {challenge_uid} has NOT been processed yet")
    
    return exists


def get_challenge_status(challenge_uid: str) -> Optional[Dict[str, Any]]:
    """
    Get the status information for a processed challenge.
    
    Args:
        challenge_uid: The challenge ID to get status for
        
    Returns:
        Status data dictionary if available, None otherwise
    """
    status_dir = get_challenge_status_dir()
    status_file = status_dir / f"{challenge_uid}.json"
    
    if not status_file.exists():
        return None
    
    try:
        with open(status_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read challenge status for {challenge_uid}: {e}")
        return None
