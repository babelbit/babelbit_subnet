#!/usr/bin/env python3
"""
Simplified test for first token capture fix

Tests that the validator properly captures the first token from /start endpoint.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from babelbit.utils.predict_utterances import predict_with_utterance_engine


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
    
    # Mock chute prediction - Import inside to avoid import errors
    async def mock_chute(*args, **kwargs):
        from babelbit.chute_template.schemas import BBPredictedUtterance
        from babelbit.utils.predict_engine import call_miner_model_on_chutes
        
        # Create minimal result structure
        result = MagicMock()
        result.success = True
        result.error = None
        result.utterance = BBPredictedUtterance(
            index="test-123",
            step=0,
            prefix="Hello",
            prediction="test",
            context=""
        )
        return result
    
    with patch('babelbit.utils.predict_utterances.get_async_client', new_callable=AsyncMock) as mock_client, \
         patch('babelbit.utils.predict_utterances.get_auth_headers', new_callable=AsyncMock) as mock_headers, \
         patch('babelbit.utils.predict_engine.call_miner_model_on_chutes', new_callable=AsyncMock) as mock_chute_call:
        
        mock_client.return_value = mock_session
        mock_headers.return_value = {}
        mock_chute_call.side_effect = mock_chute
        
        # Run prediction
        dialogues = await predict_with_utterance_engine(
            utterance_engine_url="http://test:8000",
            chute_slug="test-chute"
        )
        
        # Verify
        assert "dlg-001" in dialogues, "Dialogue should be present"
        assert len(dialogues["dlg-001"]) > 0, "Should have utterance steps"
        
        last_step = dialogues["dlg-001"][-1]
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
    
    async def mock_chute(*args, **kwargs):
        from babelbit.chute_template.schemas import BBPredictedUtterance
        
        result = MagicMock()
        result.success = True
        result.error = None
        result.utterance = BBPredictedUtterance(
            index="test-456",
            step=0,
            prefix="",
            prediction="test",
            context=""
        )
        return result
    
    with patch('babelbit.utils.predict_utterances.get_async_client', new_callable=AsyncMock) as mock_client, \
         patch('babelbit.utils.predict_utterances.get_auth_headers', new_callable=AsyncMock) as mock_headers, \
         patch('babelbit.utils.predict_engine.call_miner_model_on_chutes', new_callable=AsyncMock) as mock_chute_call:
        
        mock_client.return_value = mock_session
        mock_headers.return_value = {}
        mock_chute_call.side_effect = mock_chute
        
        dialogues = await predict_with_utterance_engine(
            utterance_engine_url="http://test:8000",
            chute_slug="test-chute"
        )
        
        assert "dlg-002" in dialogues
        assert len(dialogues["dlg-002"]) > 0
        
        last_step = dialogues["dlg-002"][-1]
        assert last_step.ground_truth == "Hello world", \
            f"Expected 'Hello world', got '{last_step.ground_truth}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
