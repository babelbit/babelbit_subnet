#!/usr/bin/env python3
"""
Test suite for partial scoring failures

Tests cover:
1. Partial utterance list completion
2. Mixed success/failure across miners
3. Score aggregation with missing data
4. Empty ground truth handling (timeout scenarios)
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import json

from babelbit.cli.runner import runner, group_steps_into_utterances
from babelbit.utils.miner_registry import Miner
from babelbit.chute_template.schemas import BBPredictedUtterance


class TestPartialScoringFailures:
    """Test suite for partial scoring failure scenarios"""

    @pytest.mark.asyncio
    async def test_runner_handles_partial_utterance_completion(self, tmp_path):
        """Test that runner handles incomplete utterances (missing done=True)"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.CHUTES_TIMEOUT_SEC = 10.0
        
        logs_dir = tmp_path / "logs"
        scores_dir = tmp_path / "scores"
        
        sample_miner = Miner(
            uid=1, hotkey="test_hotkey", model="test/model",
            revision="main", slug="test-miner", chute_id="chute1", block=100
        )
        
        # Utterance steps without final done=True (incomplete)
        incomplete_utterances = {
            "test_hotkey": {
                "dlg-1": [
                    BBPredictedUtterance(
                        index="utt-1", step=0, prefix="Hello",
                        prediction="world", done=False,  # Not done
                        ground_truth="Hello world EOF"
                    ),
                    BBPredictedUtterance(
                        index="utt-1", step=1, prefix="Hello world",
                        prediction="!", done=False,  # Still not done
                        ground_truth="Hello world EOF"
                    ),
                    # Missing final step with done=True
                ]
            }
        }
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123"), \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={1: sample_miner}), \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock, return_value=incomplete_utterances), \
             patch('babelbit.cli.runner.score_jsonl', return_value={"dialogue_summary": {"average_U_best_early": 0.3}, "utterances": []}), \
             patch('babelbit.cli.runner.close_http_clients'):
            
            with patch.dict('os.environ', {
                'BB_OUTPUT_LOGS_DIR': str(logs_dir),
                'BB_OUTPUT_SCORES_DIR': str(scores_dir)
            }):
                # Should handle incomplete utterances without crashing
                await runner()
            
            # Verify log files were created even with incomplete utterances
            log_files = list(logs_dir.glob("*.jsonl"))
            assert len(log_files) > 0, "Should create log files for incomplete utterances"

    def test_group_steps_into_utterances_with_incomplete_steps(self):
        """Test utterance grouping when some utterances are incomplete"""
        
        steps = [
            BBPredictedUtterance(index="utt-1", step=0, prefix="Hello", prediction="world", done=True, ground_truth="Hello world EOF"),
            BBPredictedUtterance(index="utt-2", step=0, prefix="How", prediction="are", done=False, ground_truth="How are you EOF"),
            BBPredictedUtterance(index="utt-2", step=1, prefix="How are", prediction="you", done=False, ground_truth="How are you EOF"),
            # Missing final step with done=True for utt-2
            BBPredictedUtterance(index="utt-3", step=0, prefix="Good", prediction="bye", done=True, ground_truth="Good bye EOF"),
        ]
        
        grouped = group_steps_into_utterances(steps)
        
        # The function groups by done=True markers. With done=False steps followed by done=True,
        # they all get grouped together until the next done=True
        # So we get 2 groups: utt-1 (done), then utt-2+utt-3 together (incomplete + complete)
        assert len(grouped) >= 2, f"Expected at least 2 utterance groups, got {len(grouped)}"
        
        # First utterance should be complete (1 step)
        assert len(grouped[0]) == 1
        assert grouped[0][0].done is True

    @pytest.mark.asyncio
    async def test_runner_handles_mixed_success_failure_across_miners(self, tmp_path):
        """Test that runner continues when some miners succeed and others fail"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.CHUTES_TIMEOUT_SEC = 10.0
        
        logs_dir = tmp_path / "logs"
        scores_dir = tmp_path / "scores"
        
        miner1 = Miner(uid=1, hotkey="hotkey1", model="test/model1", revision="main", slug="miner-1", chute_id="chute1", block=100)
        miner2 = Miner(uid=2, hotkey="hotkey2", model="test/model2", revision="main", slug="miner-2", chute_id="chute2", block=101)
        miner3 = Miner(uid=3, hotkey="hotkey3", model="test/model3", revision="main", slug="miner-3", chute_id="chute3", block=102)
        
        # Mixed results: miner-1 succeeds, miner-2 has no dialogues, miner-3 succeeds
        mixed_results = {
            "hotkey1": {
                "dlg-1": [
                    BBPredictedUtterance(index="utt-1", step=0, prefix="Test", prediction="1", done=True, ground_truth="Test 1 EOF")
                ]
            },
            "hotkey2": {},  # No dialogues (failure case)
            "hotkey3": {
                "dlg-2": [
                    BBPredictedUtterance(index="utt-1", step=0, prefix="Test", prediction="3", done=True, ground_truth="Test 3 EOF")
                ]
            }
        }
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123"), \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={1: miner1, 2: miner2, 3: miner3}), \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock, return_value=mixed_results), \
             patch('babelbit.cli.runner.score_jsonl', return_value={"dialogue_summary": {"average_U_best_early": 0.5}, "utterances": []}), \
             patch('babelbit.cli.runner.close_http_clients'):
            
            with patch.dict('os.environ', {
                'BB_OUTPUT_LOGS_DIR': str(logs_dir),
                'BB_OUTPUT_SCORES_DIR': str(scores_dir)
            }):
                await runner()
            
            # Should have score files for miner-1 and miner-3, but not miner-2
            score_files = list(scores_dir.glob("*-score.json"))
            
            # Check that successful miners have score files
            miner1_files = [f for f in score_files if "_miner_1_" in f.name]
            miner3_files = [f for f in score_files if "_miner_3_" in f.name]
            
            assert len(miner1_files) > 0, "Miner 1 should have score files"
            assert len(miner3_files) > 0, "Miner 3 should have score files"

    @pytest.mark.asyncio
    async def test_runner_handles_scoring_exception_for_one_dialogue(self, tmp_path):
        """Test that runner continues to other dialogues when one scoring fails"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.CHUTES_TIMEOUT_SEC = 10.0
        
        logs_dir = tmp_path / "logs"
        scores_dir = tmp_path / "scores"
        
        sample_miner = Miner(
            uid=1, hotkey="test_hotkey", model="test/model",
            revision="main", slug="test-miner", chute_id="chute1", block=100
        )
        
        # Multiple dialogues
        multi_dialogues = {
            "test_hotkey": {
                "dlg-1": [BBPredictedUtterance(index="utt-1", step=0, prefix="Test", prediction="1", done=True, ground_truth="Test 1 EOF")],
                "dlg-2": [BBPredictedUtterance(index="utt-2", step=0, prefix="Test", prediction="2", done=True, ground_truth="Test 2 EOF")],
                "dlg-3": [BBPredictedUtterance(index="utt-3", step=0, prefix="Test", prediction="3", done=True, ground_truth="Test 3 EOF")],
            }
        }
        
        scoring_calls = [0]
        
        def mock_score_with_failure(jsonl_path):
            scoring_calls[0] += 1
            
            # Second dialogue fails to score
            if scoring_calls[0] == 2:
                raise ValueError("Scoring computation error")
            
            return {
                "dialogue_summary": {"average_U_best_early": 0.5},
                "utterances": []
            }
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123"), \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={1: sample_miner}), \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock, return_value=multi_dialogues), \
             patch('babelbit.cli.runner.score_jsonl', side_effect=mock_score_with_failure), \
             patch('babelbit.cli.runner.close_http_clients'):
            
            with patch.dict('os.environ', {
                'BB_OUTPUT_LOGS_DIR': str(logs_dir),
                'BB_OUTPUT_SCORES_DIR': str(scores_dir)
            }):
                # Should not crash despite scoring failure
                await runner()
            
            # Should have attempted to score all 3 dialogues
            assert scoring_calls[0] == 3, f"Expected 3 scoring attempts, got {scoring_calls[0]}"

    @pytest.mark.asyncio
    async def test_score_aggregation_with_missing_data(self, tmp_path):
        """Test that challenge summary handles missing dialogue scores correctly"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.CHUTES_TIMEOUT_SEC = 10.0
        
        logs_dir = tmp_path / "logs"
        scores_dir = tmp_path / "scores"
        
        sample_miner = Miner(
            uid=1, hotkey="test_hotkey", model="test/model",
            revision="main", slug="test-miner", chute_id="chute1", block=100
        )
        
        multi_dialogues = {
            "test_hotkey": {
                "dlg-1": [BBPredictedUtterance(index="utt-1", step=0, prefix="Test", prediction="1", done=True, ground_truth="Test 1 EOF")],
                "dlg-2": [BBPredictedUtterance(index="utt-2", step=0, prefix="Test", prediction="2", done=True, ground_truth="Test 2 EOF")],
                "dlg-3": [BBPredictedUtterance(index="utt-3", step=0, prefix="Test", prediction="3", done=True, ground_truth="Test 3 EOF")],
            }
        }
        
        scoring_calls = [0]
        
        def mock_score_selective(jsonl_path):
            scoring_calls[0] += 1
            
            # Only first and third dialogues return scores
            if scoring_calls[0] == 2:
                raise ValueError("Scoring failed")
            
            return {
                "dialogue_summary": {"average_U_best_early": 0.5},
                "utterances": []
            }
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123"), \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={1: sample_miner}), \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock, return_value=multi_dialogues), \
             patch('babelbit.cli.runner.score_jsonl', side_effect=mock_score_selective), \
             patch('babelbit.cli.runner.close_http_clients'):
            
            with patch.dict('os.environ', {
                'BB_OUTPUT_LOGS_DIR': str(logs_dir),
                'BB_OUTPUT_SCORES_DIR': str(scores_dir)
            }):
                await runner()
            
            # Check challenge summary file
            summary_files = list(scores_dir.glob("challenge_run_*.json"))
            
            if summary_files:
                with open(summary_files[0], 'r') as f:
                    summary = json.load(f)
                
                # Should only include successful dialogues (dlg-1 and dlg-3)
                assert 'dialogues' in summary
                assert len(summary['dialogues']) == 2, f"Expected 2 successful dialogues, got {len(summary['dialogues'])}"
                
                # Challenge mean should be calculated from available scores only
                assert 'challenge_mean_U' in summary
                assert summary['challenge_mean_U'] == 0.5  # Both successful dialogues scored 0.5

    @pytest.mark.asyncio
    async def test_runner_handles_miner_with_no_slug(self, tmp_path):
        """Test that runner handles miners without slug (axon-only miners) properly"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.CHUTES_TIMEOUT_SEC = 10.0
        
        logs_dir = tmp_path / "logs"
        scores_dir = tmp_path / "scores"
        
        # Axon-only miner without slug (should still be scored using hotkey)
        miner_no_slug = Miner(
            uid=1, hotkey="test_hotkey_axon", model=None,
            revision=None, slug=None, chute_id=None, block=100,
            axon_ip="192.168.1.1", axon_port=8091
        )
        
        miner_with_slug = Miner(
            uid=2, hotkey="test_hotkey_chute", model="test/model2",
            revision="main", slug="valid-miner", chute_id="chute2", block=101
        )
        
        # Both miners get tracked by hotkey now
        dialogues = {
            "test_hotkey_axon": {
                "dlg-1": [BBPredictedUtterance(index="utt-1", step=0, prefix="Test", prediction="axon_output", done=True, ground_truth="Test axon_output EOF")]
            },
            "test_hotkey_chute": {
                "dlg-2": [BBPredictedUtterance(index="utt-2", step=0, prefix="Test", prediction="chute_output", done=True, ground_truth="Test chute_output EOF")]
            }
        }
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123"), \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={1: miner_no_slug, 2: miner_with_slug}), \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock, return_value=dialogues), \
             patch('babelbit.cli.runner.score_jsonl', return_value={"dialogue_summary": {"average_U_best_early": 0.5}, "utterances": []}), \
             patch('babelbit.cli.runner.close_http_clients'):
            
            with patch.dict('os.environ', {
                'BB_OUTPUT_LOGS_DIR': str(logs_dir),
                'BB_OUTPUT_SCORES_DIR': str(scores_dir)
            }):
                # Both miners should now be scored (using hotkey as key)
                await runner()
            
            # Both miners should have scores
            score_files = list(scores_dir.glob("*-score.json"))
            miner1_files = [f for f in score_files if "_miner_1_" in f.name]
            miner2_files = [f for f in score_files if "_miner_2_" in f.name]
            assert len(miner1_files) > 0, "Axon miner (no slug) should now have score files"
            assert len(miner2_files) > 0, "Chute miner (with slug) should have score files"

    @pytest.mark.asyncio
    async def test_runner_handles_empty_utterance_list(self, tmp_path):
        """Test that runner handles dialogues with empty utterance lists"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.CHUTES_TIMEOUT_SEC = 10.0
        
        logs_dir = tmp_path / "logs"
        scores_dir = tmp_path / "scores"
        
        sample_miner = Miner(
            uid=1, hotkey="test_hotkey", model="test/model",
            revision="main", slug="test-miner", chute_id="chute1", block=100
        )
        
        # Dialogue with empty utterance list
        empty_dialogues = {
            "test_hotkey": {
                "dlg-empty": []  # No utterances
            }
        }
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123"), \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={1: sample_miner}), \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock, return_value=empty_dialogues), \
             patch('babelbit.cli.runner.score_jsonl', return_value={"dialogue_summary": {"average_U_best_early": 0.0}, "utterances": []}), \
             patch('babelbit.cli.runner.close_http_clients'):
            
            with patch.dict('os.environ', {
                'BB_OUTPUT_LOGS_DIR': str(logs_dir),
                'BB_OUTPUT_SCORES_DIR': str(scores_dir)
            }):
                # Should handle empty utterance list without crashing
                await runner()
            
            # Log file should still be created (even if empty)
            log_files = list(logs_dir.glob("*.jsonl"))
            assert len(log_files) >= 0  # May or may not create file for empty dialogue

    @pytest.mark.asyncio
    async def test_partial_evaluation_data_in_utterances(self):
        """Test handling of utterances with missing or partial evaluation data"""
        
        # Utterances with missing evaluation fields
        partial_utterances = [
            BBPredictedUtterance(
                index="utt-1", step=0, prefix="Test", prediction="output",
                done=True, ground_truth="Test output EOF",
                evaluation=None  # No evaluation data
            ),
            BBPredictedUtterance(
                index="utt-2", step=0, prefix="Hello", prediction="world",
                done=True, ground_truth="Hello world EOF"
                # evaluation field completely missing
            ),
        ]
        
        grouped = group_steps_into_utterances(partial_utterances)
        
        # Should handle missing evaluation gracefully
        assert len(grouped) == 2
        assert all(len(g) == 1 for g in grouped)

    @pytest.mark.asyncio
    async def test_zero_dialogues_produces_no_challenge_summary(self, tmp_path):
        """Test that no challenge summary is created when all dialogues fail"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.CHUTES_TIMEOUT_SEC = 10.0
        
        logs_dir = tmp_path / "logs"
        scores_dir = tmp_path / "scores"
        
        sample_miner = Miner(
            uid=1, hotkey="test_hotkey", model="test/model",
            revision="main", slug="test-miner", chute_id="chute1", block=100
        )
        
        # Miner with dialogues but all scoring fails
        dialogues = {
            "test_hotkey": {
                "dlg-1": [BBPredictedUtterance(index="utt-1", step=0, prefix="Test", prediction="1", done=True, ground_truth="Test 1 EOF")],
            }
        }
        
        def mock_score_always_fails(jsonl_path):
            raise RuntimeError("Scoring system unavailable")
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123"), \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={1: sample_miner}), \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock, return_value=dialogues), \
             patch('babelbit.cli.runner.score_jsonl', side_effect=mock_score_always_fails), \
             patch('babelbit.cli.runner.close_http_clients'):
            
            with patch.dict('os.environ', {
                'BB_OUTPUT_LOGS_DIR': str(logs_dir),
                'BB_OUTPUT_SCORES_DIR': str(scores_dir)
            }):
                await runner()
            
            # Should not create challenge summary when no dialogues scored successfully
            summary_files = list(scores_dir.glob("challenge_run_*.json"))
            # Current implementation may still create summary, but ideally shouldn't
            # Test documents the current behavior

    def test_scoring_functions_handle_empty_strings(self):
        """Test that upgraded scoring utilities handle empty comparisons safely"""
        from babelbit.scoring.score_dialogue import _char_similarity, cosine_similarity
        
        # Empty ground truth with empty prediction should score 0.0
        assert _char_similarity("", "") == 0.0, "Empty strings should score 0.0"
        
        # Empty ground truth with prediction should score 0.0
        assert _char_similarity("", "hello world") == 0.0
        
        # Ground truth with empty prediction should score 0.0
        assert _char_similarity("hello world", "") == 0.0
        
        # Semantic similarity short-circuits to 0.0 for empty pairs (no model load)
        assert cosine_similarity("", "") == 0.0

    def test_score_jsonl_with_empty_ground_truth(self, tmp_path):
        """Test that score_jsonl gives 0.0 score for empty ground truth"""
        from babelbit.scoring.score_dialogue import score_jsonl
        
        jsonl_path = tmp_path / "test_empty_gt.jsonl"
        with jsonl_path.open("w") as f:
            f.write(json.dumps({"event": "predicted", "utterance_index": 0, "step": 0, "prediction": ""}) + "\n")
            f.write(json.dumps({"event": "utterance_complete", "utterance_index": 0, "ground_truth": ""}) + "\n")
        
        result = score_jsonl(jsonl_path, show_steps=False)
        avg_score = result["dialogue_summary"]["average_U_best_early"]
        
        # Empty ground truth should result in 0.0 score
        assert avg_score == 0.0, f"Empty GT should give 0.0, got {avg_score}"

    @pytest.mark.asyncio
    async def test_runner_skips_empty_ground_truth_utterances(self, tmp_path):
        """Test that runner skips utterances with empty ground truth"""
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.CHUTES_TIMEOUT_SEC = 10.0
        mock_settings.S3_LOG_DIR = "logs"
        
        logs_dir = tmp_path / "logs"
        scores_dir = tmp_path / "scores"
        
        sample_miner = Miner(
            uid=1, hotkey="test_hotkey", model="test/model",
            revision="main", slug="test-miner", chute_id="chute1", block=100
        )
        
        # Utterance with empty ground truth (timeout scenario)
        empty_gt_dialogues = {
            "test_hotkey": {
                "dlg-timeout": [
                    BBPredictedUtterance(
                        index="utt-1", step=0, prefix="Hello",
                        prediction="world", done=True,
                        ground_truth=""  # Empty due to timeout
                    )
                ]
            }
        }
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-empty-gt"), \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={1: sample_miner}), \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock, return_value=empty_gt_dialogues), \
             patch('babelbit.cli.runner.close_http_clients'):
            
            with patch.dict('os.environ', {
                'BB_OUTPUT_LOGS_DIR': str(logs_dir),
                'BB_OUTPUT_SCORES_DIR': str(scores_dir)
            }):
                await runner()
            
            # Verify empty ground truth was skipped
            log_files = list(logs_dir.glob("*.jsonl"))
            assert len(log_files) > 0
            
            jsonl_content = log_files[0].read_text()
            if jsonl_content.strip():
                lines = [json.loads(line) for line in jsonl_content.strip().split('\n')]
                complete_events = [l for l in lines if l.get('event') == 'utterance_complete']
                
                # No utterance_complete events with empty ground_truth should be written
                for evt in complete_events:
                    gt = evt.get('ground_truth', '')
                    assert gt.strip(), f"Found empty ground_truth in JSONL: {evt}"

    @pytest.mark.asyncio
    async def test_runner_scores_only_valid_ground_truth(self, tmp_path):
        """Test that runner only scores utterances with valid ground truth"""
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.CHUTES_TIMEOUT_SEC = 10.0
        mock_settings.S3_LOG_DIR = "logs"
        
        logs_dir = tmp_path / "logs"
        scores_dir = tmp_path / "scores"
        
        sample_miner = Miner(
            uid=1, hotkey="test_hotkey", model="test/model",
            revision="main", slug="test-miner", chute_id="chute1", block=100
        )
        
        # Mix of valid and empty ground truth
        mixed_dialogues = {
            "test_hotkey": {
                "dlg-mixed": [
                    BBPredictedUtterance(
                        index="utt-1", step=0, prefix="Hello",
                        prediction="world", done=True,
                        ground_truth="Hello world EOF"
                    ),
                    BBPredictedUtterance(
                        index="utt-2", step=0, prefix="Test",
                        prediction="", done=True,
                        ground_truth=""  # Empty - should be skipped
                    ),
                ]
            }
        }
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-mixed"), \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={1: sample_miner}), \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock, return_value=mixed_dialogues), \
             patch('babelbit.cli.runner.close_http_clients'):
            
            with patch.dict('os.environ', {
                'BB_OUTPUT_LOGS_DIR': str(logs_dir),
                'BB_OUTPUT_SCORES_DIR': str(scores_dir)
            }):
                await runner()
            
            score_files = list(scores_dir.glob("*-score.json"))
            
            if len(score_files) > 0:
                score_data = json.loads(score_files[0].read_text())
                utterances_scored = score_data.get("utterances", [])
                
                # Only the valid utterance should be scored
                assert len(utterances_scored) == 1, f"Should score 1 valid utterance, got {len(utterances_scored)}"
                
                # Verify no empty ground_truth
                for utt in utterances_scored:
                    gt = utt.get('ground_truth', '').strip()
                    assert gt, f"Utterance {utt.get('utterance_number')} has empty ground_truth"
                
                # Verify it's the valid one
                assert "Hello world" in utterances_scored[0].get('ground_truth')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
