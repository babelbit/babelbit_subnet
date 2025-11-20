"""
Test ground truth persistence when session ends early due to all miners being deactivated.

This test verifies the fix for the bug where ground_truth was empty when miners were
deactivated before completing an utterance (before EOF token).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from babelbit.utils.predict_utterances import predict_with_utterance_engine_multi_miner
from babelbit.chute_template.schemas import BBPredictedUtterance


@pytest.mark.asyncio
async def test_ground_truth_saved_on_early_termination():
    """
    Test that ground_truth is properly saved when all miners are deactivated
    before reaching EOF token.
    
    Simulates production scenario:
    1. Session starts with tokens: "In", "my", "practice,", "I", "often"
    2. All miners timeout/error out after 5 consecutive steps
    3. Session ends early with "All miners deactivated, stopping session early"
    4. Ground truth should be set to "In my practice, I often" for all steps
    """
    
    # Mock miner
    mock_miner = MagicMock()
    mock_miner.slug = "test-miner"
    mock_miner.uid = 123
    mock_miner.hotkey = "test-hotkey"
    
    miners = [mock_miner]
    
    # Mock /start response
    start_response = {
        "session_id": "test-session-123",
        "done": False,
        "word": "In",
        "dialogue_uid": "test-dialogue-1",
        "utterance_index": 0,
        "challenge_uid": "ch-001"
    }
    
    # Mock /next responses - tokens until all miners deactivated
    next_responses = [
        {"done": False, "word": "my", "dialogue_uid": "test-dialogue-1", "utterance_index": 0},
        {"done": False, "word": "practice,", "dialogue_uid": "test-dialogue-1", "utterance_index": 0},
        {"done": False, "word": "I", "dialogue_uid": "test-dialogue-1", "utterance_index": 0},
        {"done": False, "word": "often", "dialogue_uid": "test-dialogue-1", "utterance_index": 0},
        {"done": False, "word": "admit", "dialogue_uid": "test-dialogue-1", "utterance_index": 0},
        # Session ends early before getting more tokens
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
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(next_responses):
            response.json = AsyncMock(return_value=next_responses[idx])
        else:
            # Don't return done=True, just keep returning last token
            response.json = AsyncMock(return_value=next_responses[-1])
        
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=response)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx
    
    mock_session = MagicMock()
    mock_session.get = mock_get
    mock_session.post = mock_post
    
    # Mock prediction callback that always times out (simulating miner failures)
    prediction_error_count = {"count": 0}
    
    async def mock_prediction_callback(miner, payload, slug):
        """Simulate miner timing out on every prediction"""
        prediction_error_count["count"] += 1
        # Raise timeout error to trigger consecutive error counting
        raise Exception("Prediction timeout after 10.8s")
    
    # Mock subtensor
    mock_subtensor = MagicMock()
    mock_subtensor.block = 6917476
    
    # Patch HTTP calls and block sync
    with patch('babelbit.utils.predict_utterances.get_async_client', new_callable=AsyncMock) as mock_client, \
         patch('babelbit.utils.predict_utterances.get_auth_headers', new_callable=AsyncMock) as mock_headers, \
         patch('babelbit.utils.predict_utterances.wait_until_block_modulo', new_callable=AsyncMock) as mock_block_sync, \
         patch('babelbit.utils.predict_utterances.logger') as mock_logger:
        
        # Configure mocks
        mock_client.return_value = mock_session
        mock_headers.return_value = {}
        mock_block_sync.return_value = None
        
        # Run the function - it should handle all miners deactivating
        result = await predict_with_utterance_engine_multi_miner(
            utterance_engine_url="http://test-engine:8000",
            miners=miners,
            prediction_callback=mock_prediction_callback,
            timeout=10.0,
            max_prediction_errors=5,  # Allow 5 errors before deactivation
            subtensor=mock_subtensor,
            step_block_modulo=1
        )
        
        # Verify results
        assert "test-hotkey" in result, "Miner should have results even after early termination"
        miner_dialogues = result["test-hotkey"]
        assert "test-dialogue-1" in miner_dialogues, "Dialogue should exist"
        
        steps = miner_dialogues["test-dialogue-1"]
        assert len(steps) > 0, "Should have at least one step recorded"
        
        # Check the last step has ground_truth set
        last_step = steps[-1]
        assert last_step.ground_truth is not None, "Ground truth should not be None"
        assert last_step.ground_truth != "", "Ground truth should not be empty"
        
        # Verify ground_truth contains the tokens collected
        # Should be "In my practice, I often admit" (6 tokens before all miners deactivated at step 6)
        expected_ground_truth = "In my practice, I often admit"
        assert last_step.ground_truth == expected_ground_truth, \
            f"Expected ground_truth='{expected_ground_truth}', got '{last_step.ground_truth}'"
        
        # Verify the early termination warning was logged
        warning_calls = [call for call in mock_logger.warning.call_args_list 
                        if "All miners deactivated, stopping session early" in str(call)]
        assert len(warning_calls) > 0, "Should log early termination warning"
        
        # Verify ground truth save was logged
        info_calls = [call for call in mock_logger.info.call_args_list 
                     if "Saving ground truth for incomplete utterance" in str(call)]
        assert len(info_calls) > 0, "Should log ground truth save for incomplete utterance"


@pytest.mark.asyncio
async def test_ground_truth_normal_completion():
    """
    Test that ground_truth is still set correctly during normal EOF completion.
    This ensures our fix doesn't break the normal path.
    """
    
    # Mock miner
    mock_miner = MagicMock()
    mock_miner.slug = "test-miner-normal"
    mock_miner.uid = 456
    mock_miner.hotkey = "test-hotkey-2"
    
    miners = [mock_miner]
    
    # Mock /start response
    start_response = {
        "session_id": "test-session-456",
        "done": False,
        "word": "Hello",
        "dialogue_uid": "test-dialogue-2",
        "utterance_index": 0,
        "challenge_uid": "ch-002"
    }
    
    # Mock /next responses - complete utterance with EOF
    next_responses = [
        {"done": False, "word": "world", "dialogue_uid": "test-dialogue-2", "utterance_index": 0},
        {"done": False, "word": "EOF", "dialogue_uid": "test-dialogue-2", "utterance_index": 0},
        {"done": True, "word": "EOF EOF", "dialogue_uid": "test-dialogue-2", "utterance_index": 0},
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
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(next_responses):
            response.json = AsyncMock(return_value=next_responses[idx])
        else:
            response.json = AsyncMock(return_value={"done": True})
        
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=response)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx
    
    mock_session = MagicMock()
    mock_session.get = mock_get
    mock_session.post = mock_post
    
    # Mock prediction callback that succeeds
    async def mock_prediction_callback(miner, payload, slug):
        """Simulate successful prediction"""
        return f"prediction_{payload.step}"
    
    # Mock subtensor
    mock_subtensor = MagicMock()
    mock_subtensor.block = 6917476
    
    # Patch HTTP calls
    with patch('babelbit.utils.predict_utterances.get_async_client', new_callable=AsyncMock) as mock_client, \
         patch('babelbit.utils.predict_utterances.get_auth_headers', new_callable=AsyncMock) as mock_headers, \
         patch('babelbit.utils.predict_utterances.wait_until_block_modulo', new_callable=AsyncMock) as mock_block_sync:
        
        # Configure mocks
        mock_client.return_value = mock_session
        mock_headers.return_value = {}
        mock_block_sync.return_value = None
        
        # Run the function with normal completion
        result = await predict_with_utterance_engine_multi_miner(
            utterance_engine_url="http://test-engine:8000",
            miners=miners,
            prediction_callback=mock_prediction_callback,
            timeout=10.0,
            max_prediction_errors=5,
            subtensor=mock_subtensor,
            step_block_modulo=1
        )
        
        # Verify results (using hotkey as key)
        assert "test-hotkey-2" in result
        miner_dialogues = result["test-hotkey-2"]
        assert "test-dialogue-2" in miner_dialogues
        
        steps = miner_dialogues["test-dialogue-2"]
        
        # Find the step that should have ground_truth (the one marked done=True)
        done_steps = [s for s in steps if s.done]
        assert len(done_steps) > 0, "Should have at least one done step"
        
        last_done_step = done_steps[-1]
        assert last_done_step.ground_truth == "Hello world", \
            f"Expected ground_truth='Hello world', got '{last_done_step.ground_truth}'"
        assert last_done_step.done is True, "Step should be marked as done"
