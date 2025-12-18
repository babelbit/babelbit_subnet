#!/usr/bin/env python3
"""
Simplified test for first token capture fix

Tests that the validator properly captures the first token from /start endpoint.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from babelbit.utils.predict_utterances import predict_with_utterance_engine_multi_miner


@pytest.mark.asyncio
async def test_single_token_captures_first_token():
    """Test that a single-token utterance captures the first token from /start"""
    
    # Mock /start response
    start_response = {
        "session_id": "test-123",
        "word": "Hello",
        "token": "Hello",
        "done": False,
        "utterance_index": 0,
        "dialogue_uid": "dlg-001",
        "challenge_uid": "ch-001"
    }
    
    # Mock /next responses
    next_responses = [
        {"token": "EOF", "word": "EOF", "done": False, "utterance_index": 1, "dialogue_uid": "dlg-001"},
        {"token": "EOF EOF", "done": True, "utterance_index": 1, "dialogue_uid": "dlg-001"}
    ]
    
    # Setup HTTP mocks
    mock_response_start = AsyncMock()
    mock_response_start.status = 200
    mock_response_start.json = AsyncMock(return_value=start_response)
    
    call_count = [0]
    
    def mock_get(*args, **kwargs):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_response_start)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx
    
    def mock_post(*args, **kwargs):
        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(return_value=next_responses[call_count[0]])
        call_count[0] += 1
        
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=response)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx
    
    mock_session = MagicMock()
    mock_session.get = mock_get
    mock_session.post = mock_post
    
    class Miner:
        def __init__(self, hotkey: str):
            self.hotkey = hotkey
            self.slug = hotkey

    miner = Miner("miner-1")

    async def mock_predict(miner_obj, payload, context):
        return "test"

    with patch('babelbit.utils.predict_utterances.get_async_client', new_callable=AsyncMock) as mock_client, \
         patch('babelbit.utils.predict_utterances.get_auth_headers', new_callable=AsyncMock) as mock_headers:
        
        mock_client.return_value = mock_session
        mock_headers.return_value = {}
        
        # Run prediction
        dialogues = await predict_with_utterance_engine_multi_miner(
            utterance_engine_url="http://test:8000",
            miners=[miner],
            prediction_callback=mock_predict,
        )
        
        # Verify
        assert miner.hotkey in dialogues, "Miner dialogues should be present"
        miner_dialogues = dialogues[miner.hotkey]
        assert "dlg-001" in miner_dialogues, "Dialogue should be present"
        assert len(miner_dialogues["dlg-001"]) > 0, "Should have utterance steps"
        
        last_step = miner_dialogues["dlg-001"][-1]
        assert last_step.ground_truth == "Hello", \
            f"Expected ground_truth='Hello', got '{last_step.ground_truth}'"


@pytest.mark.asyncio
async def test_multi_token_includes_first_token():
    """Test that a two-token utterance includes the first token"""
    
    start_response = {
        "session_id": "test-456",
        "word": "Hello",
        "token": "Hello",
        "done": False,
        "utterance_index": 0,
        "dialogue_uid": "dlg-002",
        "challenge_uid": "ch-002"
    }
    
    next_responses = [
        {"token": "world", "word": "world", "done": False, "utterance_index": 0, "dialogue_uid": "dlg-002"},
        {"token": "EOF", "word": "EOF", "done": False, "utterance_index": 1, "dialogue_uid": "dlg-002"},
        {"token": "EOF EOF", "done": True, "utterance_index": 1, "dialogue_uid": "dlg-002"}
    ]
    
    mock_response_start = AsyncMock()
    mock_response_start.status = 200
    mock_response_start.json = AsyncMock(return_value=start_response)
    
    call_count = [0]
    
    def mock_get(*args, **kwargs):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_response_start)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx
    
    def mock_post(*args, **kwargs):
        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(return_value=next_responses[call_count[0]])
        call_count[0] += 1
        
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=response)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx
    
    mock_session = MagicMock()
    mock_session.get = mock_get
    mock_session.post = mock_post

    class Miner:
        def __init__(self, hotkey: str):
            self.hotkey = hotkey
            self.slug = hotkey

    miner = Miner("miner-2")

    async def mock_predict(miner_obj, payload, context):
        return "test"

    with patch('babelbit.utils.predict_utterances.get_async_client', new_callable=AsyncMock) as mock_client, \
         patch('babelbit.utils.predict_utterances.get_auth_headers', new_callable=AsyncMock) as mock_headers:
        
        mock_client.return_value = mock_session
        mock_headers.return_value = {}
        
        dialogues = await predict_with_utterance_engine_multi_miner(
            utterance_engine_url="http://test:8000",
            miners=[miner],
            prediction_callback=mock_predict,
        )
        
        assert miner.hotkey in dialogues
        miner_dialogues = dialogues[miner.hotkey]
        assert "dlg-002" in miner_dialogues
        assert len(miner_dialogues["dlg-002"]) > 0
        
        last_step = miner_dialogues["dlg-002"][-1]
        assert last_step.ground_truth == "Hello world", \
            f"Expected 'Hello world', got '{last_step.ground_truth}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
