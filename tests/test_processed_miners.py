import os
import json
import pytest
from unittest.mock import AsyncMock, patch, Mock

from babelbit.cli.runner import runner
from babelbit.utils.miner_registry import Miner

@pytest.mark.asyncio
async def test_runner_skips_processed_miners(tmp_path):
    """Test that runner skips entire challenge if any miners are already processed"""
    # Prepare output directory and existing score file
    output_dir = tmp_path / 'scores'
    output_dir.mkdir()
    challenge_uid = 'challenge-xyz'

    # Existing processed miner
    processed_miner = Miner(uid=1, hotkey='hotkey1', model='m1', revision='main', slug='miner-1', chute_id='c1', block=1)

    existing_score = {
        'challenge_uid': challenge_uid,
        'dialogue_uid': 'dlg-1',
        'miner_uid': processed_miner.uid,
        'miner_hotkey': processed_miner.hotkey,
        'utterances': [],
        'dialogue_summary': {'average_U_best_early': 0.5}
    }
    with open(output_dir / f"dialogue_run_{challenge_uid}_miner_{processed_miner.uid}_dlg_dlg-1_run_20240101-score.json", 'w') as f:
        json.dump(existing_score, f)

    # Another miner that should NOT be processed because challenge already has scores
    new_miner = Miner(uid=2, hotkey='hotkey2', model='m2', revision='main', slug='miner-2', chute_id='c2', block=2)

    with patch('babelbit.cli.runner.get_settings') as mock_settings, \
        patch('babelbit.cli.runner.get_current_challenge_uid', new_callable=AsyncMock, return_value=challenge_uid), \
        patch('babelbit.cli.runner.get_miners_from_registry', new_callable=AsyncMock) as mock_get_miners, \
        patch('babelbit.cli.runner.predict_with_utterance_engine_multi_miner', new_callable=AsyncMock) as mock_predict, \
        patch('babelbit.cli.runner.close_http_clients') as mock_close_clients, \
        patch('babelbit.cli.runner.init_utterance_auth') as mock_init_auth, \
        patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock) as mock_auth:

        # Settings mock
        settings_obj = Mock()
        settings_obj.BABELBIT_NETUID = 42
        mock_settings.return_value = settings_obj

        # Miner registry returns both miners
        mock_get_miners.return_value = {processed_miner.uid: processed_miner, new_miner.uid: new_miner}

        await runner(utterance_engine_url='http://localhost:8000', output_dir=str(output_dir))

        # Runner should skip entire challenge since processed miners exist
        # So predict should NOT be called at all
        assert mock_predict.call_count == 0, "Expected runner to skip challenge with existing scores"

        # Note: close_http_clients is NOT called when runner returns early (before try-finally block)
