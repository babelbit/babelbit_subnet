#!/usr/bin/env python3
"""
Test suite for multi-step prediction fixes.
Tests both:
1. Multi-step predictions are captured (not just final step)
2. Miner identifiers are properly included in scoring_submissions
"""
import pytest
from babelbit.cli import runner as runner_mod

# Semantic scoring disabled: if score_jsonl is unavailable, skip scoring-dependent tests
if getattr(runner_mod, "score_jsonl", None) is None:  # pragma: no cover
    pytest.skip("score_jsonl unavailable (semantic scoring reverted)", allow_module_level=True)

import json
import os
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from babelbit.cli.runner import runner, group_steps_into_utterances
from babelbit.chute_template.schemas import BBPredictedUtterance
from babelbit.utils.miner_registry import Miner


def print_score_summary(score_data: dict):
    """Print score data in the same format as the runner output"""
    print("\n" + "=" * 80)
    print(f"Score Summary for dialogue: {score_data.get('dialogue_uid', 'unknown')}")
    print("=" * 80 + "\n")
    
    for utt in score_data.get('utterances', []):
        utt_num = utt.get('utterance_number', 0)
        gt = utt.get('ground_truth', '')
        print(f"[utt {utt_num}] ground_truth: {gt}")
        print(f"{'step':<5}{'lex':<8}{'sem':<8}{'earli':<8}{'U_step':<8}prediction")
        
        for step in utt.get('steps', []):
            step_num = step.get('step', 0)
            lex = step.get('lexical_similarity', 0.0)
            sem = step.get('semantic_similarity', 0.0)
            earli = step.get('earliness', 0.0)
            u_step = step.get('U_step', 0.0)
            pred = step.get('prediction', '')
            print(f"{step_num:<5}{lex:<8.4f}{sem:<8.4f}{earli:<8.4f}{u_step:<8.4f}{pred}")
        
        best_step = utt.get('best_step', 0)
        u_best = utt.get('U_best', 0.0)
        total_steps = utt.get('total_steps', 0)
        print(f"[utt {utt_num}] BEST step={best_step}  U_best={u_best:.4f}  total_steps={total_steps}\n")
    
    dialogue_avg = score_data.get('dialogue_summary', {}).get('average_U_best_early', 0.0)
    print(f"Dialogue average U (best-early): {dialogue_avg:.4f}\n")
    print("=" * 80)


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create temporary directories for test outputs"""
    scores_dir = tmp_path / "scores"
    logs_dir = tmp_path / "logs"
    scores_dir.mkdir()
    logs_dir.mkdir()
    return {"scores": str(scores_dir), "logs": str(logs_dir), "base": tmp_path}


@pytest.fixture
def mock_settings():
    """Mock settings configuration"""
    settings_mock = Mock()
    settings_mock.BABELBIT_NETUID = 42
    settings_mock.CHUTES_TIMEOUT_SEC = 10.0
    return settings_mock


@pytest.fixture
def test_miner():
    """Create a test miner with specific uid and hotkey"""
    return Miner(
        uid=1,
        hotkey="5EWYcjAe8rL8HoGJRZtZwK8s9vaKCWAfc9rSjNNydSva3Syc",
        model="test/model",
        revision="main",
        slug="test-miner-1",
        chute_id="chute-1",
        block=100
    )


@pytest.fixture
def multi_step_dialogue_utterances():
    """
    Sample dialogue with multiple prediction steps per utterance.
    Simulates what predict_with_utterance_engine should return after the fix.
    """
    return {
        "dialogue-multi-step": [
            # First utterance with 3 steps
            BBPredictedUtterance(
                index="session-1",
                step=0,
                prefix="Hello",
                prediction="world",
                done=False,
            ),
            BBPredictedUtterance(
                index="session-1",
                step=1,
                prefix="Hello there",
                prediction="friend",
                done=False,
            ),
            BBPredictedUtterance(
                index="session-1",
                step=2,
                prefix="Hello there world",
                prediction="!",
                done=True,
                ground_truth="Hello there world"
            ),
            # Second utterance with 2 steps
            BBPredictedUtterance(
                index="session-1",
                step=0,
                prefix="How",
                prediction="are you",
                done=False,
            ),
            BBPredictedUtterance(
                index="session-1",
                step=1,
                prefix="How are",
                prediction="you doing",
                done=True,
                ground_truth="How are you"
            ),
        ]
    }


class TestMultiStepPredictions:
    """Test that multi-step predictions are properly captured"""

    def test_group_steps_into_utterances(self, multi_step_dialogue_utterances):
        """Test that steps are correctly grouped into utterances by done=True"""
        steps = multi_step_dialogue_utterances["dialogue-multi-step"]
        grouped = group_steps_into_utterances(steps)
        
        # Should have 2 utterances
        assert len(grouped) == 2, f"Expected 2 utterances, got {len(grouped)}"
        
        # First utterance should have 3 steps
        assert len(grouped[0]) == 3, f"First utterance should have 3 steps, got {len(grouped[0])}"
        assert grouped[0][0].step == 0
        assert grouped[0][1].step == 1
        assert grouped[0][2].step == 2
        assert grouped[0][2].done is True
        assert grouped[0][2].ground_truth == "Hello there world"
        
        # Second utterance should have 2 steps
        assert len(grouped[1]) == 2, f"Second utterance should have 2 steps, got {len(grouped[1])}"
        assert grouped[1][0].step == 0
        assert grouped[1][1].step == 1
        assert grouped[1][1].done is True
        assert grouped[1][1].ground_truth == "How are you"

    @pytest.mark.asyncio
    async def test_runner_saves_multi_step_predictions(
        self,
        mock_settings,
        test_miner,
        multi_step_dialogue_utterances,
        temp_test_dir,
    ):
        "Test that runner saves all prediction steps to JSONL, not just the final one"
        
        # Convert to multi_miner format (using hotkey as key)
        multi_miner_result = {
            test_miner.hotkey: multi_step_dialogue_utterances
        }
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-multi") as mock_get_challenge, \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={1: test_miner}), \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock, return_value=multi_miner_result), \
             patch('babelbit.cli.runner.close_http_clients'), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock):
            
            # Run the function
            with patch.dict('os.environ', {'BB_OUTPUT_LOGS_DIR': temp_test_dir['logs']}):
                await runner(output_dir=temp_test_dir['scores'])
            
            # Find the dialogue JSONL file
            logs_dir = Path(temp_test_dir['logs'])
            jsonl_files = list(logs_dir.glob("dialogue_run_*.jsonl"))
            assert len(jsonl_files) == 1, f"Expected 1 JSONL file, found {len(jsonl_files)}"
            
            # Read and verify the JSONL content
            with open(jsonl_files[0], 'r') as f:
                events = [json.loads(line) for line in f if line.strip()]
            
            # Count predicted events (should match number of steps)
            predicted_events = [e for e in events if e['event'] == 'predicted']
            complete_events = [e for e in events if e['event'] == 'utterance_complete']
            
            # Should have 5 predicted events (3 for first utterance, 2 for second)
            assert len(predicted_events) == 5, f"Expected 5 predicted events, got {len(predicted_events)}"
            
            # Should have 2 utterance_complete events
            assert len(complete_events) == 2, f"Expected 2 complete events, got {len(complete_events)}"
            
            # Verify first utterance has steps 0, 1, 2
            utt0_events = [e for e in predicted_events if e['utterance_index'] == 0]
            assert len(utt0_events) == 3, f"First utterance should have 3 steps, got {len(utt0_events)}"
            steps = sorted([e['step'] for e in utt0_events])
            assert steps == [0, 1, 2], f"First utterance steps should be [0,1,2], got {steps}"
            
            # Verify second utterance has steps 0, 1
            utt1_events = [e for e in predicted_events if e['utterance_index'] == 1]
            assert len(utt1_events) == 2, f"Second utterance should have 2 steps, got {len(utt1_events)}"
            steps = sorted([e['step'] for e in utt1_events])
            assert steps == [0, 1], f"Second utterance steps should be [0,1], got {steps}"

    @pytest.mark.asyncio
    async def test_scored_file_contains_multiple_steps(
        self,
        mock_settings,
        test_miner,
        multi_step_dialogue_utterances,
        temp_test_dir,
    ):
        """Test that scored JSON files contain multiple steps per utterance"""
        
        # Convert to multi_miner format (using hotkey as key)
        multi_miner_result = {
            test_miner.hotkey: multi_step_dialogue_utterances
        }
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-score") as mock_get_challenge, \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={1: test_miner}), \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock, return_value=multi_miner_result), \
             patch('babelbit.cli.runner.close_http_clients'), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock):
            
            # Run the function
            with patch.dict('os.environ', {'BB_OUTPUT_LOGS_DIR': temp_test_dir['logs']}):
                await runner(output_dir=temp_test_dir['scores'])
            
            # Find the dialogue score JSON file
            scores_dir = Path(temp_test_dir['scores'])
            score_files = list(scores_dir.glob("dialogue_run_*-score.json"))
            assert len(score_files) == 1, f"Expected 1 score file, found {len(score_files)}"
            
            # Read and verify the score file
            with open(score_files[0], 'r') as f:
                score_data = json.load(f)
            
            # Print formatted score summary
            print_score_summary(score_data)
            
            # Verify structure
            assert 'utterances' in score_data
            assert len(score_data['utterances']) == 2
            
            # First utterance should have 3 steps
            utt0 = score_data['utterances'][0]
            assert 'steps' in utt0
            assert len(utt0['steps']) == 3, f"First utterance should have 3 steps, got {len(utt0['steps'])}"
            
            # Second utterance should have 2 steps
            utt1 = score_data['utterances'][1]
            assert 'steps' in utt1
            assert len(utt1['steps']) == 2, f"Second utterance should have 2 steps, got {len(utt1['steps'])}"
            
            # Verify each step has the required scoring fields
            for step in utt0['steps']:
                assert 'step' in step
                assert 'lexical_similarity' in step
                assert 'semantic_similarity' in step
                assert 'earliness' in step
                assert 'U_step' in step


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
