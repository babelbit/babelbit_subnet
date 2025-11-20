#!/usr/bin/env python3
"""
Test suite for the runner pipeline
"""
import pytest
import asyncio
import tempfile
import os
import json
import shutil
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List
from pathlib import Path

from babelbit.cli.runner import runner
from babelbit.chute_template.schemas import BBPredictedUtterance, BBUtteranceEvaluation
from babelbit.utils.miner_registry import Miner


@pytest.fixture
def temp_scores_dir():
    """Provide a persistent test scores directory (./test_scores) for inspection.

    Does not delete files after tests so developers can sanity check outputs.
    Files are git-ignored via root .gitignore.
    """
    scores_dir = "./test_scores"
    os.makedirs(scores_dir, exist_ok=True)
    # Clean directory to prevent cross-test contamination that affects filtering logic
    for f in os.listdir(scores_dir):
        try:
            os.remove(os.path.join(scores_dir, f))
        except IsADirectoryError:
            pass
        except Exception:
            continue
    return scores_dir


@pytest.fixture
def temp_logs_dir():
    """Provide a persistent test logs directory (./test_logs) for inspection.

    No cleanup so developers can open the JSONL after a test run. Git-ignored.
    """
    logs_dir = "./test_logs"
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


@pytest.fixture
def mock_settings():
    """Mock settings configuration"""
    settings_mock = Mock()
    settings_mock.BABELBIT_NETUID = 42
    return settings_mock


@pytest.fixture
def sample_miners():
    """Sample miners for testing"""
    return {
        1: Miner(uid=1, hotkey="hotkey1", model="test/model1", revision="main", slug="test-miner-1", chute_id="chute1", block=100),
        2: Miner(uid=2, hotkey="hotkey2", model="test/model2", revision="main", slug="test-miner-2", chute_id="chute2", block=101),
        3: Miner(uid=3, hotkey="hotkey3", model="test/model3", revision="main", slug="test-miner-3", chute_id="chute3", block=102),
    }


@pytest.fixture
def sample_dialogue_utterances():
    """Sample dialogue utterances that would be returned by predict_with_utterance_engine_multi_miner"""
    # Returns dict mapping miner_slug -> {dialogue_uid -> [utterances]}
    single_dialogue = {
        "dialogue-123": [
            BBPredictedUtterance(
                index="utterance-1",
                step=0,
                prefix="Hello",
                prediction="world",
                done=True,
                ground_truth="Hello world EOF"
            ),
            BBPredictedUtterance(
                index="utterance-2", 
                step=1,
                prefix="How are",
                prediction="you",
                done=True,
                ground_truth="How are you EOF"
            ),
            BBPredictedUtterance(
                index="utterance-3",
                step=2, 
                prefix="I'm doing",
                prediction="well",
                done=True,
                ground_truth="I'm doing well thanks EOF"
            )
        ]
    }
    # Return format for multi_miner: {miner_hotkey: dialogues_dict}
    return {
        "hotkey1": single_dialogue,
        "hotkey2": single_dialogue,
        "hotkey3": single_dialogue,
    }


@pytest.fixture
def evaluated_utterances(sample_dialogue_utterances):
    """Sample utterances with evaluation results and final score"""
    # Get the first (and only) dialogue from the sample
    dialogue_utterances = list(sample_dialogue_utterances.values())[0]
    
    # Create evaluated utterances with mock evaluation results
    evaluated_utterance_list = []
    for utterance in dialogue_utterances:
        evaluated_utterance = BBPredictedUtterance(
            index=utterance.index,
            step=utterance.step,
            prefix=utterance.prefix,
            prediction=utterance.prediction,
            context=utterance.context,
            done=utterance.done,
            ground_truth=utterance.ground_truth,
            evaluation=BBUtteranceEvaluation(
                lexical_similarity=1.0,
                semantic_similarity=1.0,
                earliness=1.0,
                u_step=1.0
            )
        )
        evaluated_utterance_list.append(evaluated_utterance)
    
    # Legacy DialogueScore object removed in current pipeline; return just list
    return evaluated_utterance_list, None


class TestRunner:
    """Test suite for the runner function"""

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'BB_MAX_MINERS_PER_RUN': '3'})
    async def test_runner_success_full_pipeline(
        self,
        mock_settings,
        sample_miners,
        sample_dialogue_utterances,
        temp_scores_dir,
        temp_logs_dir,
    ):
        """Test successful execution of the full runner pipeline"""
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123") as mock_get_challenge, \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock) as mock_get_miners, \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock) as mock_predict, \
             patch('babelbit.cli.runner.close_http_clients') as mock_close_clients, \
             patch('babelbit.cli.runner.init_utterance_auth') as mock_init_auth, \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock) as mock_auth:
             
            # Challenge logger not used in file scorer mode
            
            # Setup mocks
            mock_get_miners.return_value = sample_miners
            mock_predict.return_value = sample_dialogue_utterances  # Dict of miner_slug -> dialogues
            # evaluate removed in file scorer mode
            
            # Run the function
            with patch.dict('os.environ', {'BB_OUTPUT_LOGS_DIR': temp_logs_dir}):
                await runner(utterance_engine_url="http://localhost:8000", output_dir=temp_scores_dir)
            
            # Verify calls
            mock_get_miners.assert_called_once_with(42, subtensor=None)  # NETUID + subtensor param
            assert mock_predict.call_count == 1  # Called once for all miners
            
            # Verify predict_with_utterance_engine_multi_miner was called with correct args
            predict_call_kwargs = mock_predict.call_args[1]
            assert predict_call_kwargs['utterance_engine_url'] == "http://localhost:8000"
            assert len(predict_call_kwargs['miners']) == 3
            mock_close_clients.assert_called_once()
            
            # No evaluate path anymore; ensure event JSONLs created per miner per dialogue
            raw_log_files = [f for f in os.listdir(temp_logs_dir) if f.startswith('dialogue_run_') and f.endswith('.jsonl')]
            assert len(raw_log_files) >= 3, f"Expected >=3 raw dialogue logs; found {raw_log_files}"
            # Scored dialogue JSON files should exist
            scored_files = [f for f in os.listdir(temp_scores_dir) if f.endswith('-score.json')]
            assert len(scored_files) >= 3, f"Expected >=3 scored dialogue JSON files; found {scored_files}"
            
            # Verify files were written to temp directory
            # Since we're using mocks, verify the expected calls were made
            # No ChallengeLogger usage in file scorer mode
            
            # Note: utterance completion events are now logged within predict_with_utterance_engine,
            # so when that function is mocked, the logger methods won't be called.
            # The real logging happens in the predict function, not in the runner.
            
            # Verify save functions were called
            # No save_* calls in file scorer mode

    @pytest.mark.asyncio
    async def test_runner_no_miners_found(self, mock_settings, temp_logs_dir, temp_scores_dir):
        """Test behavior when no miners are found"""
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123") as mock_get_challenge, \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock) as mock_get_miners, \
             patch('babelbit.cli.runner.init_utterance_auth') as mock_init_auth, \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock) as mock_auth, \
             patch('babelbit.cli.runner.close_http_clients') as mock_close_clients:
            
            # Setup: no miners returned
            mock_get_miners.return_value = {}
            
            # Run the function
            with patch.dict('os.environ', {'BB_OUTPUT_LOGS_DIR': temp_logs_dir}):
                await runner(output_dir=temp_scores_dir)
            
            # Verify behavior
            mock_get_miners.assert_called_once_with(42, subtensor=None)
            mock_close_clients.assert_called_once()

    @pytest.mark.asyncio
    async def test_runner_miner_prediction_failure(
        self,
        mock_settings,
        sample_miners,
        temp_logs_dir,
        temp_scores_dir,
    ):
        """Test handling of miner prediction failures"""
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123") as mock_get_challenge, \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock) as mock_get_miners, \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock) as mock_predict, \
             patch('babelbit.cli.runner.init_utterance_auth') as mock_init_auth, \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock) as mock_auth, \
             patch('babelbit.cli.runner.close_http_clients') as mock_close_clients:
            
            # Setup mocks
            mock_get_miners.return_value = sample_miners
            mock_predict.side_effect = Exception("Prediction failed")
            
            # Run the function (should not crash)
            with patch.dict('os.environ', {'BB_OUTPUT_LOGS_DIR': temp_logs_dir}):
                await runner(output_dir=temp_scores_dir)
            
            # Verify calls
            mock_get_miners.assert_called_once_with(42, subtensor=None)
            assert mock_predict.call_count == 1  # Called once for all miners
            # evaluate removed
            mock_close_clients.assert_called_once()

    @pytest.mark.asyncio
    async def test_runner_evaluation_failure(
        self,
        mock_settings,
        sample_miners,
        sample_dialogue_utterances,
        temp_logs_dir,
        temp_scores_dir,
    ):
        """Test handling of evaluation failures"""
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123") as mock_get_challenge, \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock) as mock_get_miners, \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock) as mock_predict, \
             patch('babelbit.cli.runner.init_utterance_auth') as mock_init_auth, \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock) as mock_auth, \
             patch('babelbit.cli.runner.close_http_clients') as mock_close_clients:
            
            # Setup mocks
            mock_get_miners.return_value = sample_miners
            mock_predict.return_value = sample_dialogue_utterances
            
            # Run the function 
            with patch.dict('os.environ', {'BB_OUTPUT_LOGS_DIR': temp_logs_dir}):
                await runner(output_dir=temp_scores_dir)
            
            # Verify calls
            mock_get_miners.assert_called_once_with(42, subtensor=None)
            assert mock_predict.call_count == 1  # Called once for all miners
            # no evaluation assertions
            mock_close_clients.assert_called_once()

    @pytest.mark.asyncio 
    async def test_runner_no_ground_truth_extraction(
        self,
        mock_settings,
        sample_miners,
        temp_logs_dir,
        temp_scores_dir,
    ):
        """Test handling when ground truth cannot be extracted from dialogue"""
        
        # Create utterances without ground_truth - multi_miner format (using hotkey)
        utterances_no_gt = {
            "hotkey1": {
                "dialogue-456": [
                    BBPredictedUtterance(
                        index="utterance-1",
                        step=0,
                        prefix="Hello",
                        prediction="world",
                        done=True,
                        ground_truth=None  # No ground truth
                    )
                ]
            }
        }
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123") as mock_get_challenge, \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock) as mock_get_miners, \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock) as mock_predict, \
             patch('babelbit.cli.runner.init_utterance_auth') as mock_init_auth, \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock) as mock_auth, \
             patch('babelbit.cli.runner.close_http_clients') as mock_close_clients:
            
            # Setup mocks
            mock_get_miners.return_value = {1: sample_miners[1]}  # Single miner
            mock_predict.return_value = utterances_no_gt
            
            # Run the function
            with patch.dict('os.environ', {'BB_OUTPUT_LOGS_DIR': temp_logs_dir}):
                await runner(output_dir=temp_scores_dir)
            
            # Verify predict was called
            mock_predict.assert_called_once()

    @pytest.mark.asyncio
    @patch.dict('os.environ', {'BB_MAX_MINERS_PER_RUN': '2'})
    async def test_runner_max_miners_limit(
        self,
        mock_settings,
        sample_miners,
        sample_dialogue_utterances,
        temp_scores_dir,
        temp_logs_dir,
    ):
        """Test that runner respects MAX_MINERS limit"""
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123") as mock_get_challenge, \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock) as mock_get_miners, \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock) as mock_predict, \
             patch('babelbit.cli.runner.init_utterance_auth') as mock_init_auth, \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock) as mock_auth, \
             patch('babelbit.cli.runner.close_http_clients') as mock_close_clients:
            # Challenge logger not used in file scorer mode
            
            # Setup mocks
            mock_get_miners.return_value = sample_miners  # 3 miners available
            mock_predict.return_value = sample_dialogue_utterances
            
            # Run the function
            with patch.dict('os.environ', {'BB_OUTPUT_LOGS_DIR': temp_logs_dir}):
                await runner(output_dir=temp_scores_dir)
            
            # Verify the multi_miner function was called once, but with only 2 miners
            assert mock_predict.call_count == 1
            call_kwargs = mock_predict.call_args[1]
            assert len(call_kwargs['miners']) == 2  # Only 2 miners passed due to MAX_MINERS=2
            # no evaluation assertions


@pytest.mark.asyncio
async def test_runner_integration(mock_settings, temp_logs_dir, temp_scores_dir):
    """Integration test with minimal mocking (ensures runner executes with patches)."""
    mock_miner = Miner(uid=1, hotkey="test_hotkey", model="test/model", revision="main", slug="test-slug", chute_id="test-chute", block=100)
    # Multi-miner format: {miner_slug: {dialogue_uid: [utterances]}}
    mock_dialogues = {
        "test-slug": {
            "dialogue-int": [
                BBPredictedUtterance(
                    index="utt-1",
                    step=0,
                    prefix="Hello",
                    prediction="world",
                    done=True,
                    ground_truth="Hello world EOF"
                )
            ]
        }
    }

    with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
         patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-int") as mock_get_challenge, \
         patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={1: mock_miner}) as mock_get_miners, \
         patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock, return_value=mock_dialogues) as mock_predict, \
         patch('babelbit.cli.runner.init_utterance_auth') as mock_init_auth, \
         patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock) as mock_auth, \
         patch('babelbit.cli.runner.close_http_clients'):
        with patch.dict('os.environ', {'BB_OUTPUT_LOGS_DIR': temp_logs_dir}):
            await runner(output_dir=temp_scores_dir)
        mock_get_miners.assert_called_once()
        mock_predict.assert_called_once()


@pytest.mark.asyncio
async def test_runner_score_jsonl_unavailable(mock_settings, temp_logs_dir, temp_scores_dir):
    """Test runner behavior when score_jsonl module is unavailable"""
    mock_miner = Miner(uid=1, hotkey="test_hotkey", model="test/model", revision="main", slug="test-slug", chute_id="test-chute", block=100)
    mock_dialogues = {
        "test-slug": {
            "dialogue-score-unavail": [
                BBPredictedUtterance(
                    index="utt-1",
                    step=0,
                    prefix="Hello",
                    prediction="world",
                    done=True,
                    ground_truth="Hello world EOF"
                )
            ]
        }
    }

    with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
         patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-no-scorer") as mock_get_challenge, \
         patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={1: mock_miner}) as mock_get_miners, \
         patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock, return_value=mock_dialogues) as mock_predict, \
         patch('babelbit.cli.runner.init_utterance_auth') as mock_init_auth, \
         patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock) as mock_auth, \
         patch('babelbit.cli.runner.score_jsonl', None), \
         patch('babelbit.cli.runner.close_http_clients'):
        
        with patch.dict('os.environ', {'BB_OUTPUT_LOGS_DIR': temp_logs_dir}):
            await runner(output_dir=temp_scores_dir)
        
        # Should complete without errors but skip scoring
        mock_predict.assert_called_once()
        
        # Verify JSONL was written for our specific challenge
        logs_dir = Path(temp_logs_dir)
        our_jsonl_files = list(logs_dir.glob("dialogue_run_challenge-no-scorer_*.jsonl"))
        assert len(our_jsonl_files) == 1, f"Expected 1 JSONL file for challenge-no-scorer, found {len(our_jsonl_files)}"
        
        # No score files should be created when score_jsonl is None
        scores_dir = Path(temp_scores_dir)
        our_score_files = list(scores_dir.glob("dialogue_run_challenge-no-scorer_*-score.json"))
        assert len(our_score_files) == 0, f"Expected 0 score files when scorer unavailable, found {len(our_score_files)}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
