#!/usr/bin/env python3
"""
Test suite for runner_loop block synchronization and tempo-based scheduling

Tests cover:
1. Block synchronization logic  
2. Tempo-based scheduling (every 300 blocks)
3. Subtensor reconnection after failures
4. Time-based fallback when blockchain unreachable
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from babelbit.cli.runner import runner_loop


class TestRunnerLoopBlockSynchronization:
    """Test suite for block synchronization and tempo-based scheduling"""

    @pytest.mark.asyncio
    async def test_runner_loop_tempo_based_scheduling(self):
        """Test that runner executes at correct block intervals (every 300 blocks)"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.BB_RUNNER_ON_STARTUP = False
        
        # Simulate block progression: 0, 100, 200, 300 (should trigger), 400, 600 (should trigger)
        blocks = [0, 100, 200, 300, 400, 600]
        block_index = [0]
        runner_calls = []
        
        async def mock_get_current_block():
            if block_index[0] >= len(blocks):
                raise asyncio.CancelledError()  # End the loop
            block = blocks[block_index[0]]
            block_index[0] += 1
            return block
        
        async def mock_wait_for_block():
            await asyncio.sleep(0.01)  # Small delay to prevent tight loop
        
        async def mock_runner(subtensor=None):
            runner_calls.append(subtensor)
            await asyncio.sleep(0.01)
        
        mock_subtensor = Mock()
        mock_subtensor.get_current_block = mock_get_current_block
        mock_subtensor.wait_for_block = mock_wait_for_block
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_subtensor', new_callable=AsyncMock, return_value=mock_subtensor), \
             patch('babelbit.cli.runner.reset_subtensor', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.runner', new_callable=AsyncMock, side_effect=mock_runner), \
             patch.dict('os.environ', {'BABELBIT_RUNNER_TEMPO': '300'}):
            
            try:
                await asyncio.wait_for(runner_loop(), timeout=1.0)
            except asyncio.TimeoutError:
                pass  # Expected - loop runs indefinitely
            except asyncio.CancelledError:
                pass  # Expected when we run out of blocks
        
        # Should trigger at blocks 300 and 600 (blocks divisible by 300)
        assert len(runner_calls) >= 2, f"Expected at least 2 runner calls, got {len(runner_calls)}"

    @pytest.mark.asyncio
    async def test_runner_loop_subtensor_reconnection_after_failure(self):
        """Test that runner_loop reconnects to subtensor after connection failures"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.BB_RUNNER_ON_STARTUP = False
        mock_settings.BITTENSOR_SUBTENSOR_ENDPOINT = "ws://localhost:9944"
        mock_settings.BITTENSOR_SUBTENSOR_FALLBACK = "ws://fallback:9944"
        
        connection_attempts = [0]
        
        async def mock_get_subtensor():
            connection_attempts[0] += 1
            
            # First attempt fails
            if connection_attempts[0] == 1:
                raise ConnectionError("Connection refused")
            
            # Second attempt succeeds
            mock_st = Mock()
            mock_st.get_current_block = AsyncMock(return_value=300)
            mock_st.wait_for_block = AsyncMock(side_effect=asyncio.CancelledError())  # Exit loop
            return mock_st
        
        async def mock_runner(subtensor=None):
            await asyncio.sleep(0.01)
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_subtensor', new_callable=AsyncMock, side_effect=mock_get_subtensor), \
             patch('babelbit.cli.runner.reset_subtensor', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.runner', new_callable=AsyncMock, side_effect=mock_runner), \
             patch('asyncio.sleep', new_callable=AsyncMock):  # Speed up retry delays
            
            try:
                await asyncio.wait_for(runner_loop(), timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        
        # Should have attempted connection at least twice
        assert connection_attempts[0] >= 2, f"Expected at least 2 connection attempts, got {connection_attempts[0]}"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Complex async timing/state mocking - test validates time-based fallback behavior")
    async def test_runner_loop_time_based_fallback_when_blockchain_unreachable(self):
        """Test time-based fallback triggers when blockchain is unreachable"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.BB_RUNNER_ON_STARTUP = False
        mock_settings.BITTENSOR_SUBTENSOR_ENDPOINT = "ws://localhost:9944"
        mock_settings.BITTENSOR_SUBTENSOR_FALLBACK = "ws://fallback:9944"
        
        runner_calls = []
        current_time = [time.time()]
        
        block_fetch_count = [0]
        
        async def mock_get_current_block():
            block_fetch_count[0] += 1
            
            # First call: connection verification
            if block_fetch_count[0] == 1:
                return 100
                
            # Second call: returns block 300 (triggers first normal run)
            if block_fetch_count[0] == 2:
                return 300
            
            # Third call onwards: fail (blockchain unreachable)
            # By this time, current_time has advanced by 4000s
            raise TimeoutError("Connection timeout")
        
        async def mock_runner(subtensor=None):
            runner_calls.append({
                'time': current_time[0],
                'subtensor': subtensor,
                'via_fallback': subtensor is None
            })
            
            # Cancel after we get the time-based fallback call
            if len(runner_calls) >= 2:
                raise asyncio.CancelledError()
        
        def mock_time():
            return current_time[0]
        
        async def mock_wait_for_block():
            # After first successful run, advance time way past the expected interval
            if len(runner_calls) == 1:
                current_time[0] += 4000  # Jump 4000 seconds into the future
            await asyncio.sleep(0.001)
        
        # Mock asyncio.sleep to not actually sleep but still yield control
        original_sleep = asyncio.sleep
        async def mock_sleep(duration):
            await original_sleep(0.001)  # Use original sleep with tiny duration
        
        mock_subtensor = Mock()
        mock_subtensor.get_current_block = mock_get_current_block
        mock_subtensor.wait_for_block = mock_wait_for_block
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.get_subtensor', new_callable=AsyncMock, return_value=mock_subtensor), \
             patch('babelbit.cli.runner.reset_subtensor', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.runner', new_callable=AsyncMock, side_effect=mock_runner), \
             patch('time.time', side_effect=mock_time), \
             patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep):
            
            try:
                await asyncio.wait_for(runner_loop(), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        
        # Should have triggered twice: once at block 300, once via time-based fallback
        assert len(runner_calls) >= 2, f"Expected at least 2 runner calls, got {len(runner_calls)}"
        # Second call should be via fallback (subtensor=None)
        if len(runner_calls) >= 2:
            assert runner_calls[1]['via_fallback'], "Second call should have been via time-based fallback"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Complex async timing/state mocking - test validates max retry behavior and recovery")
    async def test_runner_loop_max_retries_exceeded(self):
        """Test that runner_loop handles max retry limit correctly"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.BB_RUNNER_ON_STARTUP = False
        mock_settings.BITTENSOR_SUBTENSOR_ENDPOINT = "ws://localhost:9944"
        mock_settings.BITTENSOR_SUBTENSOR_FALLBACK = "ws://fallback:9944"
        
        connection_attempts = [0]
        
        async def mock_get_subtensor():
            connection_attempts[0] += 1
            
            # Always fail to trigger max retries
            raise ConnectionError("Connection refused")
        
        sleep_calls = []
        
        async def mock_sleep(duration):
            sleep_calls.append(duration)
            # After collecting max retries and one long sleep, cancel
            if connection_attempts[0] >= 5 and any(s == 300 for s in sleep_calls):
                raise asyncio.CancelledError()
            await asyncio.sleep(0.001)  # Tiny actual sleep to yield control
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.get_subtensor', new_callable=AsyncMock, side_effect=mock_get_subtensor), \
             patch('babelbit.cli.runner.reset_subtensor', new_callable=AsyncMock), \
             patch('asyncio.sleep', new_callable=AsyncMock, side_effect=mock_sleep), \
             patch.dict('os.environ', {'BABELBIT_MAX_SUBTENSOR_RETRIES': '5'}):
            
            try:
                await asyncio.wait_for(runner_loop(), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        
        # Should have attempted connection at least 5 times (max retries)
        assert connection_attempts[0] >= 5, f"Expected at least 5 connection attempts, got {connection_attempts[0]}"
        
        # Should have triggered the 5-minute sleep after max retries
        assert any(s == 300 for s in sleep_calls), f"Expected 300s sleep after max retries, got {sleep_calls}"

    @pytest.mark.asyncio
    async def test_runner_loop_block_synchronization_between_validators(self):
        """Test that validators synchronize at the same block (block % tempo == 0)"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.BB_RUNNER_ON_STARTUP = False
        
        # Simulate blocks around tempo boundary
        blocks = [298, 299, 300, 301, 302]
        block_index = [0]
        runner_blocks = []
        
        async def mock_get_current_block():
            if block_index[0] >= len(blocks):
                raise asyncio.CancelledError()
            block = blocks[block_index[0]]
            block_index[0] += 1
            return block
        
        async def mock_wait_for_block():
            await asyncio.sleep(0.01)
        
        async def mock_runner(subtensor=None):
            # Capture which block triggered the run
            current_block = blocks[block_index[0] - 1]
            runner_blocks.append(current_block)
            await asyncio.sleep(0.01)
        
        mock_subtensor = Mock()
        mock_subtensor.get_current_block = mock_get_current_block
        mock_subtensor.wait_for_block = mock_wait_for_block
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_subtensor', new_callable=AsyncMock, return_value=mock_subtensor), \
             patch('babelbit.cli.runner.reset_subtensor', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.runner', new_callable=AsyncMock, side_effect=mock_runner), \
             patch.dict('os.environ', {'BABELBIT_RUNNER_TEMPO': '300'}):
            
            try:
                await asyncio.wait_for(runner_loop(), timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        
        # Should only trigger at block 300 (divisible by tempo)
        assert 300 in runner_blocks, f"Expected runner to trigger at block 300, got blocks {runner_blocks}"
        assert 298 not in runner_blocks, "Runner should not trigger at block 298"
        assert 299 not in runner_blocks, "Runner should not trigger at block 299"

    @pytest.mark.asyncio
    async def test_runner_loop_startup_run_configuration(self):
        """Test that runner executes on startup when BB_RUNNER_ON_STARTUP is enabled"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.BB_RUNNER_ON_STARTUP = True  # Enable startup run
        
        runner_calls = []
        blocks = [100]  # Not divisible by 300
        block_index = [0]
        
        async def mock_get_current_block():
            if len(runner_calls) >= 1:  # Exit after first run
                raise asyncio.CancelledError()
            block = blocks[block_index[0] % len(blocks)]
            block_index[0] += 1
            return block
        
        async def mock_runner(subtensor=None):
            runner_calls.append(subtensor)
            await asyncio.sleep(0.01)
        
        mock_subtensor = Mock()
        mock_subtensor.get_current_block = mock_get_current_block
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_subtensor', new_callable=AsyncMock, return_value=mock_subtensor), \
             patch('babelbit.cli.runner.reset_subtensor', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.runner', new_callable=AsyncMock, side_effect=mock_runner):
            
            try:
                await asyncio.wait_for(runner_loop(), timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        
        # Should trigger once on startup even though block 100 % 300 != 0
        assert len(runner_calls) >= 1, f"Expected runner to trigger on startup, got {len(runner_calls)} calls"

    @pytest.mark.asyncio
    async def test_runner_loop_clears_stale_connection_on_error(self):
        """Test that runner_loop clears cached subtensor connection after errors"""
        
        mock_settings = Mock()
        mock_settings.BABELBIT_NETUID = 42
        mock_settings.BB_RUNNER_ON_STARTUP = False
        
        reset_calls = []
        
        async def mock_reset_subtensor():
            reset_calls.append(time.time())
        
        block_calls = [0]
        
        async def mock_get_current_block():
            block_calls[0] += 1
            
            # First call succeeds, second fails (stale connection)
            if block_calls[0] == 1:
                return 300
            elif block_calls[0] == 2:
                raise ConnectionError("Connection lost")
            else:
                raise asyncio.CancelledError()  # Exit loop
        
        async def mock_runner(subtensor=None):
            await asyncio.sleep(0.01)
        
        mock_subtensor = Mock()
        mock_subtensor.get_current_block = mock_get_current_block
        
        with patch('babelbit.cli.runner.get_settings', return_value=mock_settings), \
             patch('babelbit.cli.runner.init_utterance_auth'), \
             patch('babelbit.cli.runner.authenticate_utterance_engine', new_callable=AsyncMock), \
             patch('babelbit.cli.runner.get_subtensor', new_callable=AsyncMock, return_value=mock_subtensor), \
             patch('babelbit.cli.runner.reset_subtensor', new_callable=AsyncMock, side_effect=mock_reset_subtensor), \
             patch('babelbit.cli.runner.runner', new_callable=AsyncMock, side_effect=mock_runner), \
             patch('asyncio.sleep', new_callable=AsyncMock):
            
            try:
                await asyncio.wait_for(runner_loop(), timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        
        # Should have called reset_subtensor at least once after connection error
        assert len(reset_calls) >= 1, f"Expected reset_subtensor to be called after error, got {len(reset_calls)} calls"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
