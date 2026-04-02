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


_CHALLENGE_TYPE_ALIASES = {
    "round1": "main",
    "round2": "arena",
}


def _normalize_challenge_type(challenge_type: object) -> Optional[str]:
    if isinstance(challenge_type, str):
        normalized = challenge_type.strip().lower()
        if normalized:
            return _CHALLENGE_TYPE_ALIASES.get(normalized, normalized)
    return None


def get_challenge_status_dir() -> Path:
    """Get the directory where challenge status files are stored."""
    status_dir = Path(os.getenv("BB_CHALLENGE_STATUS_DIR", "data/challenge_status"))
    status_dir.mkdir(parents=True, exist_ok=True)
    return status_dir


def _status_file_path(status_dir: Path, challenge_uid: str, challenge_type: Optional[str] = None) -> Path:
    if challenge_type:
        return status_dir / f"{challenge_uid}__{challenge_type}.json"
    return status_dir / f"{challenge_uid}.json"


def mark_challenge_processed(
    challenge_uid: str,
    miner_count: int,
    total_dialogues: int,
    mean_score: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    challenge_type: Optional[str] = None,
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
        challenge_type: Optional challenge type namespace (e.g. main, solo, arena)
    """
    status_dir = get_challenge_status_dir()
    normalized_type = _normalize_challenge_type(challenge_type)
    status_file = _status_file_path(status_dir, challenge_uid, normalized_type)

    status_data = {
        "challenge_uid": challenge_uid,
        "challenge_type": normalized_type,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "miner_count": miner_count,
        "total_dialogues": total_dialogues,
        "mean_score": mean_score,
        "metadata": metadata or {},
    }

    try:
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        if normalized_type:
            logger.info(
                "Marked challenge %s (type=%s) as processed (%s miners, %s dialogues)",
                challenge_uid,
                normalized_type,
                miner_count,
                total_dialogues,
            )
        else:
            logger.info(
                "Marked challenge %s as processed (%s miners, %s dialogues)",
                challenge_uid,
                miner_count,
                total_dialogues,
            )
    except Exception as e:
        logger.warning(f"Failed to mark challenge {challenge_uid} as processed: {e}")


def is_challenge_processed(challenge_uid: str, challenge_type: Optional[str] = None) -> bool:
    """
    Check if a challenge has been processed by the runner.

    Args:
        challenge_uid: The challenge ID to check
        challenge_type: Optional challenge type namespace

    Returns:
        True if the challenge has been processed, False otherwise
    """
    status_dir = get_challenge_status_dir()
    normalized_type = _normalize_challenge_type(challenge_type)
    status_file = _status_file_path(status_dir, challenge_uid, normalized_type)

    exists = status_file.exists()
    if not exists:
        if normalized_type == "main":
            # Legacy compatibility with historical untyped or first-stage typed main status files.
            exists = _status_file_path(status_dir, challenge_uid).exists() or _status_file_path(
                status_dir, challenge_uid, "round1"
            ).exists()
        elif normalized_type == "arena":
            exists = _status_file_path(status_dir, challenge_uid, "round2").exists()
        elif normalized_type is None:
            # Compatibility: callers without a type should still observe typed main status files.
            exists = _status_file_path(status_dir, challenge_uid, "main").exists()
            if not exists:
                exists = _status_file_path(status_dir, challenge_uid, "round1").exists()

    if exists:
        if normalized_type:
            logger.debug("Challenge %s (type=%s) has been processed", challenge_uid, normalized_type)
        else:
            logger.debug("Challenge %s has been processed", challenge_uid)
    else:
        if normalized_type:
            logger.debug("Challenge %s (type=%s) has NOT been processed yet", challenge_uid, normalized_type)
        else:
            logger.debug("Challenge %s has NOT been processed yet", challenge_uid)

    return exists


def get_challenge_status(challenge_uid: str, challenge_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get the status information for a processed challenge.

    Args:
        challenge_uid: The challenge ID to get status for
        challenge_type: Optional challenge type namespace

    Returns:
        Status data dictionary if available, None otherwise
    """
    status_dir = get_challenge_status_dir()
    normalized_type = _normalize_challenge_type(challenge_type)
    status_file = _status_file_path(status_dir, challenge_uid, normalized_type)

    if not status_file.exists():
        if normalized_type == "main":
            legacy_file = _status_file_path(status_dir, challenge_uid)
            legacy_main_alias_file = _status_file_path(status_dir, challenge_uid, "round1")
            if legacy_file.exists():
                status_file = legacy_file
            elif legacy_main_alias_file.exists():
                status_file = legacy_main_alias_file
            else:
                return None
        elif normalized_type == "arena":
            legacy_round2_typed_file = _status_file_path(status_dir, challenge_uid, "round2")
            if legacy_round2_typed_file.exists():
                status_file = legacy_round2_typed_file
            else:
                return None
        elif normalized_type is None:
            # Compatibility: allow untyped callers to read typed main status files.
            typed_main_file = _status_file_path(status_dir, challenge_uid, "main")
            legacy_main_alias_file = _status_file_path(status_dir, challenge_uid, "round1")
            if typed_main_file.exists():
                status_file = typed_main_file
            elif legacy_main_alias_file.exists():
                status_file = legacy_main_alias_file
            else:
                return None
        else:
            return None

    try:
        with open(status_file, 'r') as f:
            status_data = json.load(f)
        if isinstance(status_data, dict):
            status_data["challenge_type"] = _normalize_challenge_type(status_data.get("challenge_type"))
        return status_data
    except Exception as e:
        logger.warning(f"Failed to read challenge status for {challenge_uid}: {e}")
        return None


async def is_challenge_processed_db(challenge_uid: str) -> bool:
    """
    Check if a challenge has been processed by querying the database.

    This is a fallback method when file-based status tracking is not available.

    Args:
        challenge_uid: The challenge ID to check

    Returns:
        True if the challenge has scores in the database, False otherwise
    """
    try:
        from babelbit.utils.db_pool import db_pool, _iter_scores_for_challenge

        await db_pool.init()
        scores = await _iter_scores_for_challenge(challenge_uid)
        has_scores = len(scores) > 0

        if has_scores:
            logger.debug(f"Challenge {challenge_uid} has {len(scores)} scores in DB")
        else:
            logger.debug(f"Challenge {challenge_uid} has no scores in DB yet")

        return has_scores
    except Exception as e:
        logger.warning(f"Failed to check challenge status in DB: {e}")
        return False
