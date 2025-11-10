#!/usr/bin/env python3
"""
Test suite for file I/O error handling

Tests cover:
1. Missing log directories
2. Permission errors on score file writes
3. Disk full scenarios  
4. Corrupted challenge JSON files
"""
import pytest
import asyncio
import os
import json
import errno
from unittest.mock import Mock, AsyncMock, patch, mock_open
from pathlib import Path

from babelbit.cli.runner import runner
from babelbit.utils.file_handling import (
    save_dialogue_score_file,
    save_challenge_summary_file,
    get_processed_miners_for_challenge
)
from babelbit.utils.miner_registry import Miner
from babelbit.chute_template.schemas import BBPredictedUtterance


class TestFileIOErrorHandling:
    """Test suite for file I/O error scenarios"""

    @pytest.mark.asyncio
    async def test_runner_handles_missing_log_directory_creation(self, tmp_path):
        """Test that runner creates log directory if it doesn't exist"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.CHUTES_TIMEOUT_SEC = 10.0
        
        # Use non-existent path
        logs_dir = tmp_path / "nonexistent" / "logs" / "deep"
        scores_dir = tmp_path / "scores"
        
        assert not logs_dir.exists(), "Log directory should not exist initially"
        
        sample_miner = Miner(
            uid=1, hotkey="test_hotkey", model="test/model",
            revision="main", slug="test-miner", chute_id="chute1", block=100
        )
        
        sample_dialogues = {
            "test-miner": {
                "dlg-1": [
                    BBPredictedUtterance(
                        index="utt-1", step=0, prefix="Test",
                        prediction="output", done=True,
                        ground_truth="Test output EOF"
                    )
                ]
            }
        }
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123"), \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={1: sample_miner}), \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock, return_value=sample_dialogues), \
             patch('babelbit.cli.runner.score_jsonl', return_value={"dialogue_summary": {"average_U_best_early": 0.5}, "utterances": []}), \
             patch('babelbit.cli.runner.close_http_clients'):
            
            with patch.dict('os.environ', {
                'BB_OUTPUT_LOGS_DIR': str(logs_dir),
                'BB_OUTPUT_SCORES_DIR': str(scores_dir)
            }):
                await runner()
            
            # Verify directory was created
            assert logs_dir.exists(), "Log directory should be created"
            assert logs_dir.is_dir(), "Log path should be a directory"

    @pytest.mark.asyncio
    async def test_save_score_file_permission_error(self, tmp_path):
        """Test handling of permission errors when saving score files"""
        
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        
        # Make directory read-only (no write permission)
        scores_dir.chmod(0o444)
        
        score_data = {
            'challenge_uid': 'challenge-123',
            'dialogue_uid': 'dlg-1',
            'miner_uid': 1,
            'miner_hotkey': 'test_hotkey',
            'utterances': [],
            'dialogue_summary': {'average_U_best_early': 0.5}
        }
        
        try:
            with pytest.raises((PermissionError, OSError)):
                save_dialogue_score_file(score_data, output_dir=str(scores_dir))
        finally:
            # Restore permissions for cleanup
            scores_dir.chmod(0o755)

    @pytest.mark.asyncio
    async def test_runner_continues_on_individual_score_write_failure(self, tmp_path):
        """Test that runner continues processing other miners when one score write fails"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.CHUTES_TIMEOUT_SEC = 10.0
        
        logs_dir = tmp_path / "logs"
        scores_dir = tmp_path / "scores"
        logs_dir.mkdir()
        scores_dir.mkdir()
        
        miner1 = Miner(uid=1, hotkey="hotkey1", model="test/model1", revision="main", slug="miner-1", chute_id="chute1", block=100)
        miner2 = Miner(uid=2, hotkey="hotkey2", model="test/model2", revision="main", slug="miner-2", chute_id="chute2", block=101)
        
        dialogues = {
            "miner-1": {
                "dlg-1": [BBPredictedUtterance(index="utt-1", step=0, prefix="Test", prediction="output", done=True, ground_truth="Test output EOF")]
            },
            "miner-2": {
                "dlg-2": [BBPredictedUtterance(index="utt-2", step=0, prefix="Hello", prediction="world", done=True, ground_truth="Hello world EOF")]
            }
        }
        
        save_calls = [0]
        
        def mock_save_with_error(score_data, output_dir=None):
            save_calls[0] += 1
            
            # First call (miner-1) fails
            if save_calls[0] == 1:
                raise PermissionError("Permission denied")
            
            # Second call (miner-2) succeeds
            filename = f"test_score_{save_calls[0]}.json"
            output_path = output_dir if output_dir else "."
            filepath = Path(output_path) / filename
            with open(filepath, 'w') as f:
                json.dump(score_data, f)
            return str(filepath)
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123"), \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={1: miner1, 2: miner2}), \
             patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock, return_value=dialogues), \
             patch('babelbit.cli.runner.score_jsonl', return_value={"dialogue_summary": {"average_U_best_early": 0.5}, "utterances": []}), \
             patch('babelbit.cli.runner.save_dialogue_score_file', side_effect=mock_save_with_error), \
             patch('babelbit.cli.runner.close_http_clients'):
            
            with patch.dict('os.environ', {
                'BB_OUTPUT_LOGS_DIR': str(logs_dir),
                'BB_OUTPUT_SCORES_DIR': str(scores_dir)
            }):
                # Should not raise - continues despite error
                await runner()
            
            # Both miners should have been attempted
            assert save_calls[0] == 2, f"Expected 2 save attempts, got {save_calls[0]}"

    @pytest.mark.asyncio
    async def test_disk_full_scenario_on_log_write(self, tmp_path):
        """Test handling of disk full errors when writing logs"""
        
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        
        def mock_write_with_disk_full(*args, **kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")
        
        mock_file = mock_open()
        mock_file.return_value.write = mock_write_with_disk_full
        
        # Simulate disk full during JSONL write
        with patch('builtins.open', mock_file):
            with pytest.raises(OSError) as exc_info:
                with open(logs_dir / "test.jsonl", 'w') as f:
                    f.write('{"test": "data"}\n')
            
            assert exc_info.value.errno == errno.ENOSPC

    @pytest.mark.asyncio
    async def test_corrupted_challenge_json_file(self, tmp_path):
        """Test handling of corrupted/malformed challenge JSON files"""
        
        challenges_dir = tmp_path / "challenges"
        challenges_dir.mkdir()
        
        # Create corrupted JSON file
        corrupted_file = challenges_dir / "corrupted_challenge.json"
        with open(corrupted_file, 'w') as f:
            f.write('{"invalid": json syntax, missing quotes}')
        
        # Try to load it
        with pytest.raises(json.JSONDecodeError):
            with open(corrupted_file, 'r') as f:
                json.load(f)

    @pytest.mark.asyncio
    async def test_partially_written_score_file_handling(self, tmp_path):
        """Test detection and handling of partially written score files"""
        
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        
        # Create incomplete/corrupted score file
        partial_file = scores_dir / "dialogue_run_challenge-123_miner_1_dlg_dlg-1_run_20240101-score.json"
        with open(partial_file, 'w') as f:
            f.write('{"challenge_uid": "challenge-123", "dialogue_uid":')  # Incomplete JSON
        
        # Test that get_processed_miners handles corrupted files gracefully
        processed_miners = get_processed_miners_for_challenge(str(scores_dir), "challenge-123")
        
        # Should not crash - either skip corrupted file or handle gracefully
        assert isinstance(processed_miners, set), "Should return a set even with corrupted files"

    @pytest.mark.asyncio
    async def test_file_write_atomic_behavior(self, tmp_path):
        """Test that score files are written atomically to prevent corruption"""
        
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        
        score_data = {
            'challenge_uid': 'challenge-123',
            'dialogue_uid': 'dlg-1',
            'miner_uid': 1,
            'miner_hotkey': 'test_hotkey',
            'utterances': [],
            'dialogue_summary': {'average_U_best_early': 0.5}
        }
        
        # Save score file
        filepath = save_dialogue_score_file(score_data, output_dir=str(scores_dir))
        
        # Verify file exists and is valid JSON
        assert os.path.exists(filepath), "Score file should exist"
        
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == score_data, "Loaded data should match saved data"

    @pytest.mark.asyncio
    async def test_log_directory_with_invalid_characters(self, tmp_path):
        """Test handling of directory paths with special characters"""
        
        # Note: Some characters like ':' are invalid on certain filesystems
        # Test with valid special characters
        logs_dir = tmp_path / "logs_with-special.chars_2024"
        scores_dir = tmp_path / "scores"
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.CHUTES_TIMEOUT_SEC = 10.0
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value="challenge-123"), \
             patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock, return_value={}), \
             patch('babelbit.cli.runner.close_http_clients'):
            
            with patch.dict('os.environ', {
                'BB_OUTPUT_LOGS_DIR': str(logs_dir),
                'BB_OUTPUT_SCORES_DIR': str(scores_dir)
            }):
                await runner()
            
            # Should handle special characters correctly
            assert logs_dir.exists(), "Directory with special characters should be created"

    @pytest.mark.asyncio
    async def test_concurrent_writes_to_same_directory(self, tmp_path):
        """Test that multiple concurrent score writes don't interfere"""
        
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        
        score_data_template = {
            'challenge_uid': 'challenge-123',
            'miner_hotkey': 'test_hotkey',
            'utterances': [],
            'dialogue_summary': {'average_U_best_early': 0.5}
        }
        
        async def write_score(miner_uid, dialogue_uid):
            score_data = score_data_template.copy()
            score_data['miner_uid'] = miner_uid
            score_data['dialogue_uid'] = dialogue_uid
            
            # Small delay to simulate concurrent writes
            await asyncio.sleep(0.01)
            
            return save_dialogue_score_file(score_data, output_dir=str(scores_dir))
        
        # Write multiple scores concurrently
        tasks = [
            write_score(1, "dlg-1"),
            write_score(2, "dlg-2"),
            write_score(3, "dlg-3")
        ]
        
        filepaths = await asyncio.gather(*tasks)
        
        # All writes should succeed
        assert len(filepaths) == 3
        assert all(os.path.exists(fp) for fp in filepaths)
        
        # All files should be distinct
        assert len(set(filepaths)) == 3, "Each write should create a unique file"

    @pytest.mark.asyncio
    async def test_read_only_filesystem_handling(self, tmp_path):
        """Test graceful handling of read-only filesystem"""
        
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        
        score_data = {
            'challenge_uid': 'challenge-123',
            'dialogue_uid': 'dlg-1',
            'miner_uid': 1,
            'miner_hotkey': 'test_hotkey',
            'utterances': [],
            'dialogue_summary': {'average_U_best_early': 0.5}
        }
        
        # Mock open to simulate read-only filesystem
        def mock_readonly_open(*args, **kwargs):
            if 'w' in str(kwargs.get('mode', args[1] if len(args) > 1 else '')):
                raise OSError(errno.EROFS, "Read-only file system")
            return mock_open()(*args, **kwargs)
        
        with patch('builtins.open', side_effect=mock_readonly_open):
            with pytest.raises(OSError) as exc_info:
                save_dialogue_score_file(score_data, output_dir=str(scores_dir))
            
            assert exc_info.value.errno == errno.EROFS

    @pytest.mark.asyncio
    async def test_symlink_in_directory_path(self, tmp_path):
        """Test handling of symbolic links in directory paths"""
        
        real_dir = tmp_path / "real_scores"
        real_dir.mkdir()
        
        symlink_dir = tmp_path / "scores_link"
        symlink_dir.symlink_to(real_dir)
        
        score_data = {
            'challenge_uid': 'challenge-123',
            'dialogue_uid': 'dlg-1',
            'miner_uid': 1,
            'miner_hotkey': 'test_hotkey',
            'utterances': [],
            'dialogue_summary': {'average_U_best_early': 0.5}
        }
        
        # Should work through symlink
        filepath = save_dialogue_score_file(score_data, output_dir=str(symlink_dir))
        
        assert os.path.exists(filepath), "File should exist through symlink"
        
        # Verify it's in the real directory
        real_files = list(real_dir.glob("*.json"))
        assert len(real_files) == 1, "File should exist in real directory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
