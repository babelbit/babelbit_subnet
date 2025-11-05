"""
Tests for challenge status tracking functionality.
"""

import os
import json
import pytest
from pathlib import Path
from babelbit.utils.challenge_status import (
    mark_challenge_processed,
    is_challenge_processed,
    get_challenge_status,
    get_challenge_status_dir,
)


def test_mark_and_check_challenge_processed(tmp_path, monkeypatch):
    """Test marking a challenge as processed and checking its status."""
    # Use temp directory for test
    monkeypatch.setenv("BB_CHALLENGE_STATUS_DIR", str(tmp_path / "status"))
    
    challenge_uid = "test-challenge-123"
    
    # Initially should not be processed
    assert not is_challenge_processed(challenge_uid)
    
    # Mark as processed
    mark_challenge_processed(
        challenge_uid=challenge_uid,
        miner_count=5,
        total_dialogues=10,
        mean_score=0.85,
        metadata={"test": "data"}
    )
    
    # Now should be processed
    assert is_challenge_processed(challenge_uid)
    
    # Should be able to retrieve status
    status = get_challenge_status(challenge_uid)
    assert status is not None
    assert status["challenge_uid"] == challenge_uid
    assert status["miner_count"] == 5
    assert status["total_dialogues"] == 10
    assert status["mean_score"] == 0.85
    assert status["metadata"]["test"] == "data"
    assert "processed_at" in status


def test_get_challenge_status_nonexistent(tmp_path, monkeypatch):
    """Test getting status for a non-existent challenge."""
    monkeypatch.setenv("BB_CHALLENGE_STATUS_DIR", str(tmp_path / "status"))
    
    status = get_challenge_status("nonexistent-challenge")
    assert status is None


def test_challenge_status_dir_creation(tmp_path, monkeypatch):
    """Test that status directory is created automatically."""
    status_dir_path = tmp_path / "new_status_dir"
    monkeypatch.setenv("BB_CHALLENGE_STATUS_DIR", str(status_dir_path))
    
    # Should not exist yet
    assert not status_dir_path.exists()
    
    # Calling get_challenge_status_dir should create it
    status_dir = get_challenge_status_dir()
    assert status_dir.exists()
    assert status_dir.is_dir()


def test_mark_challenge_without_optional_fields(tmp_path, monkeypatch):
    """Test marking a challenge with only required fields."""
    monkeypatch.setenv("BB_CHALLENGE_STATUS_DIR", str(tmp_path / "status"))
    
    challenge_uid = "test-challenge-minimal"
    
    mark_challenge_processed(
        challenge_uid=challenge_uid,
        miner_count=3,
        total_dialogues=6
    )
    
    status = get_challenge_status(challenge_uid)
    assert status is not None
    assert status["challenge_uid"] == challenge_uid
    assert status["miner_count"] == 3
    assert status["total_dialogues"] == 6
    assert status["mean_score"] is None
    assert status["metadata"] == {}


@pytest.mark.asyncio
async def test_is_challenge_processed_db_mock(monkeypatch):
    """Test DB-based challenge status checking with mock."""
    from babelbit.utils.challenge_status import is_challenge_processed_db
    from unittest.mock import AsyncMock, patch
    
    challenge_uid = "test-challenge-db"
    
    # Mock the DB functions
    async def mock_iter_scores(challenge_uid_param):
        if challenge_uid_param == "test-challenge-db":
            return [("hotkey1", 0.85), ("hotkey2", 0.90)]
        return []
    
    # Mock db_pool.init
    mock_db_pool = AsyncMock()
    mock_db_pool.init = AsyncMock()
    
    with patch('babelbit.utils.db_pool.db_pool', mock_db_pool), \
         patch('babelbit.utils.db_pool._iter_scores_for_challenge', new=mock_iter_scores):
        
        # Challenge with scores should return True
        result = await is_challenge_processed_db(challenge_uid)
        assert result is True
        
        # Challenge without scores should return False
        result = await is_challenge_processed_db("nonexistent")
        assert result is False

