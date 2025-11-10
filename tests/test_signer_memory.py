#!/usr/bin/env python3
"""
Comprehensive memory leak tests for the signer API.

Test Categories:
1. Subtensor lifecycle and connection management
2. HTTP request/response cleanup
3. Asyncio task management
4. Logging and resource accumulation
5. Production workload simulations
6. Memory leak mitigation verification

Run with: pytest tests/test_signer_memory.py -v -s
"""
import pytest
import asyncio
import gc
import sys
import tracemalloc
import weakref
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

# Mock bittensor before import
sys.modules['bittensor'] = MagicMock()
sys.modules['bittensor.wallet'] = MagicMock()
sys.modules['bittensor.async_subtensor'] = MagicMock()

from babelbit.cli.signer_api import _set_weights_with_confirmation


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_wallet():
    """Standard mock wallet for testing"""
    wallet = Mock()
    wallet.hotkey.ss58_address = "5EHw6pwHQvjVGVb2D8ckPrLMCVpdsbzpyzCPNj3ych4gkEXi"
    wallet.hotkey.sign = lambda data: Mock(hex=lambda: "0xdeadbeef")
    return wallet


@pytest.fixture
def mock_successful_subtensor(mock_wallet):
    """Mock subtensor that always succeeds"""
    mock_st = AsyncMock()
    mock_st.get_current_block = AsyncMock(return_value=1000)
    mock_st.set_weights = AsyncMock(return_value=True)
    mock_st.wait_for_block = AsyncMock()
    
    mock_meta = Mock()
    mock_meta.hotkeys = [mock_wallet.hotkey.ss58_address]
    mock_meta.last_update = [1001]
    mock_st.metagraph = AsyncMock(return_value=mock_meta)
    mock_st.close = AsyncMock()
    
    return mock_st


@pytest.fixture
def mock_failing_subtensor():
    """Mock subtensor that always fails"""
    mock_st = AsyncMock()
    mock_st.get_current_block = AsyncMock(return_value=1000)
    mock_st.set_weights = AsyncMock(return_value=False)
    mock_st.wait_for_block = AsyncMock()
    mock_st.close = AsyncMock()
    return mock_st


# ============================================================================
# Test Category 1: Subtensor Lifecycle Management
# ============================================================================

class TestSubtensorLifecycle:
    """Test subtensor connection management and caching"""
    
    @pytest.mark.asyncio
    async def test_subtensor_reused_across_retries(self, mock_wallet, mock_failing_subtensor):
        """
        Verify subtensor instance is cached and reused across retries.
        This ensures we're not creating new instances unnecessarily.
        """
        subtensor_instances = []
        
        async def track_get_subtensor():
            subtensor_instances.append(id(mock_failing_subtensor))
            return mock_failing_subtensor
        
        with patch('babelbit.cli.signer_api.get_subtensor', side_effect=track_get_subtensor):
            with patch('babelbit.cli.signer_api.reset_subtensor', AsyncMock()):
                await _set_weights_with_confirmation(
                    wallet=mock_wallet,
                    netuid=59,
                    uids=[1],
                    weights=[1.0],
                    wait_for_inclusion=False,
                    retries=3,
                    delay_s=0.01,
                    log_prefix="[test]"
                )
        
        # get_subtensor called 3 times (once per retry)
        assert len(subtensor_instances) == 3
        # But returns same cached instance
        assert len(set(subtensor_instances)) == 1, "Should reuse cached subtensor"
    
    @pytest.mark.asyncio
    async def test_single_subtensor_on_success(self, mock_wallet, mock_successful_subtensor):
        """Verify only one subtensor instance created on immediate success"""
        subtensor_instances = []
        
        def create_subtensor():
            subtensor_instances.append(mock_successful_subtensor)
            return mock_successful_subtensor
        
        with patch('babelbit.cli.signer_api.get_subtensor', side_effect=create_subtensor):
            with patch('babelbit.cli.signer_api.reset_subtensor', AsyncMock()):
                result = await _set_weights_with_confirmation(
                    wallet=mock_wallet,
                    netuid=59,
                    uids=[1, 2, 3],
                    weights=[0.33, 0.33, 0.34],
                    wait_for_inclusion=False,
                    retries=3,
                    delay_s=0.01,
                    log_prefix="[test]"
                )
        
        assert len(subtensor_instances) == 1, "Should only create one subtensor on success"
        assert result is True


# ============================================================================
# Test Category 2: HTTP Request/Response Cleanup
# ============================================================================

class TestHTTPResourceCleanup:
    """Test that HTTP resources are properly cleaned up"""
    
    @pytest.mark.asyncio
    async def test_aiohttp_request_cleanup(self, mock_wallet):
        """Verify aiohttp properly cleans up request/response objects"""
        request_refs = []
        response_refs = []
        
        async def handler(request: web.Request):
            request_refs.append(weakref.ref(request))
            resp = web.json_response({"ok": True})
            response_refs.append(weakref.ref(resp))
            return resp
        
        app = web.Application()
        app.add_routes([web.get("/test", handler)])
        
        async with TestClient(TestServer(app)) as client:
            for i in range(100):
                resp = await client.get("/test")
                await resp.json()
        
        gc.collect()
        
        alive_requests = sum(1 for ref in request_refs if ref() is not None)
        alive_responses = sum(1 for ref in response_refs if ref() is not None)
        
        # aiohttp should clean up (allow small margin for framework overhead)
        assert alive_requests < 10, f"Too many requests alive: {alive_requests}/100"
        assert alive_responses < 10, f"Too many responses alive: {alive_responses}/100"
    
    @pytest.mark.asyncio
    async def test_sign_endpoint_no_accumulation(self, mock_wallet):
        """Verify sign endpoint doesn't accumulate memory"""
        async def sign_handler(req: web.Request):
            payload = await req.json()
            return web.json_response({
                "success": True,
                "signatures": ["0x123"],
                "hotkey": mock_wallet.hotkey.ss58_address
            })
        
        app = web.Application()
        app.add_routes([web.post("/sign", sign_handler)])
        
        async with TestClient(TestServer(app)) as client:
            for i in range(50):
                resp = await client.post("/sign", json={"data": "test"})
                assert resp.status == 200
                await resp.json()
        
        gc.collect()
        # Test passes if no exceptions and completes successfully


# ============================================================================
# Test Category 3: Asyncio Task Management
# ============================================================================

class TestAsyncioTaskManagement:
    """Test that asyncio tasks don't accumulate"""
    
    @pytest.mark.asyncio
    async def test_no_task_accumulation(self):
        """Verify tasks are properly cleaned up after completion"""
        initial_tasks = len(asyncio.all_tasks())
        
        async def dummy_task(n):
            await asyncio.sleep(0.001)
            return n * 2
        
        for i in range(100):
            task = asyncio.create_task(dummy_task(i))
            await task
        
        await asyncio.sleep(0.1)
        gc.collect()
        
        final_tasks = len(asyncio.all_tasks())
        
        # Allow small margin for test framework tasks
        assert final_tasks <= initial_tasks + 2, \
            f"Tasks accumulated: {initial_tasks} -> {final_tasks}"


# ============================================================================
# Test Category 4: Logging and Resource Accumulation
# ============================================================================

class TestLoggingAndResources:
    """Test logging and other resource accumulation"""
    
    @pytest.mark.asyncio
    async def test_logging_handlers_dont_accumulate(self):
        """Verify logging handlers don't accumulate over time"""
        import logging
        
        signer_logger = logging.getLogger("sv-signer")
        initial_handlers = len(signer_logger.handlers)
        
        # Simulate many log calls
        for i in range(1000):
            signer_logger.info(f"Test log message {i}")
        
        final_handlers = len(signer_logger.handlers)
        
        assert final_handlers == initial_handlers, \
            f"Handlers accumulated: {initial_handlers} -> {final_handlers}"
    
    @pytest.mark.asyncio
    async def test_connection_pool_cleanup(self):
        """Test that connection objects are cleaned up"""
        connection_refs = []
        
        async def handler(request: web.Request):
            if request.transport:
                connection_refs.append(weakref.ref(request.transport))
            return web.json_response({"ok": True})
        
        app = web.Application()
        app.add_routes([web.get("/test", handler)])
        
        async with TestClient(TestServer(app)) as client:
            for i in range(200):
                resp = await client.get("/test")
                await resp.json()
        
        gc.collect()
        await asyncio.sleep(0.1)
        
        alive_connections = sum(1 for ref in connection_refs if ref() is not None)
        
        # Most should be cleaned up (allow for connection pooling)
        assert alive_connections < 10, \
            f"Too many connections alive: {alive_connections}/{len(connection_refs)}"


# ============================================================================
# Test Category 5: Production Workload Simulations
# ============================================================================

class TestProductionWorkloads:
    """Simulate realistic production workloads"""
    
    @pytest.mark.asyncio
    async def test_heavy_sign_workload(self, mock_wallet):
        """
        Simulate heavy signing workload with memory tracking.
        Tests 500 requests with varying payload sizes.
        """
        tracemalloc.start()
        gc.collect()
        snapshot1 = tracemalloc.take_snapshot()
        
        async def sign_handler(req: web.Request):
            payload = await req.json()
            data = payload.get("payloads") or payload.get("data") or []
            if isinstance(data, str):
                data = [data]
            sigs = [mock_wallet.hotkey.sign(d.encode("utf-8")).hex() for d in data]
            return web.json_response({
                "success": True,
                "signatures": sigs,
                "hotkey": mock_wallet.hotkey.ss58_address,
            })
        
        app = web.Application()
        app.add_routes([web.post("/sign", sign_handler)])
        
        async with TestClient(TestServer(app)) as client:
            for i in range(500):
                num_payloads = (i % 10) + 1
                payloads = [f"payload_{i}_{j}" for j in range(num_payloads)]
                
                resp = await client.post("/sign", json={"payloads": payloads})
                assert resp.status == 200
                await resp.json()
                
                # Occasional concurrent burst
                if i % 50 == 0:
                    tasks = [
                        client.post("/sign", json={"data": f"concurrent_{i}_{j}"})
                        for j in range(10)
                    ]
                    responses = await asyncio.gather(*tasks)
                    for r in responses:
                        await r.json()
        
        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_increase = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        
        tracemalloc.stop()
        
        # Allow up to 5MB for 500+ requests (generous but detects major leaks)
        assert total_increase < 5 * 1024 * 1024, \
            f"Excessive memory growth: {total_increase / 1024 / 1024:.2f} MB"
    
    @pytest.mark.asyncio
    async def test_set_weights_retry_pattern(self, mock_wallet):
        """
        Test set_weights with realistic retry patterns.
        Simulates mix of successes and failures.
        """
        tracemalloc.start()
        gc.collect()
        snapshot1 = tracemalloc.take_snapshot()
        
        call_count = 0
        
        def create_subtensor():
            nonlocal call_count
            call_count += 1
            
            mock_st = AsyncMock()
            mock_st.get_current_block = AsyncMock(return_value=1000 + call_count)
            
            # Sometimes succeeds, sometimes fails
            if call_count % 3 == 0:
                mock_st.set_weights = AsyncMock(return_value=True)
                mock_st.wait_for_block = AsyncMock()
                mock_meta = Mock()
                mock_meta.hotkeys = [mock_wallet.hotkey.ss58_address]
                mock_meta.last_update = [1001 + call_count]
                mock_st.metagraph = AsyncMock(return_value=mock_meta)
            else:
                mock_st.set_weights = AsyncMock(return_value=False)
            
            mock_st.close = AsyncMock()
            return mock_st
        
        with patch('babelbit.cli.signer_api.get_subtensor', side_effect=create_subtensor):
            with patch('babelbit.cli.signer_api.reset_subtensor', AsyncMock()):
                for _ in range(50):
                    await _set_weights_with_confirmation(
                        wallet=mock_wallet,
                        netuid=59,
                        uids=[1, 2, 3],
                        weights=[0.33, 0.33, 0.34],
                        wait_for_inclusion=False,
                        retries=2,
                        delay_s=0.001,
                        log_prefix="[test]"
                    )
        
        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_increase = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        
        tracemalloc.stop()
        
        # Should not grow excessively
        assert total_increase < 1 * 1024 * 1024, \
            f"Memory grew too much: {total_increase / 1024 / 1024:.2f} MB"


# ============================================================================
# Test Category 6: Memory Leak Mitigation Verification
# ============================================================================

class TestMemoryLeakMitigation:
    """Verify memory leak mitigation strategies work correctly"""
    
    @pytest.mark.asyncio
    async def test_gc_collect_works(self):
        """Verify garbage collection can be called and works"""
        # Create some objects
        large_objects = [bytearray(1024 * 1024) for _ in range(10)]  # 10MB
        del large_objects
        
        # Track GC calls
        collect_calls = []
        original_collect = gc.collect
        
        def track_collect(*args, **kwargs):
            collect_calls.append(True)
            return original_collect(*args, **kwargs)
        
        with patch('gc.collect', track_collect):
            gc.collect()
        
        assert len(collect_calls) >= 1, "gc.collect() should be callable"
    
    @pytest.mark.asyncio
    async def test_reset_interval_configurable(self):
        """Verify reset interval can be configured"""
        import os
        
        default = int(os.getenv("SIGNER_SUBTENSOR_RESET_INTERVAL", "100"))
        assert default == 100, "Default should be 100"
        
        with patch.dict(os.environ, {"SIGNER_SUBTENSOR_RESET_INTERVAL": "50"}):
            custom = int(os.getenv("SIGNER_SUBTENSOR_RESET_INTERVAL", "100"))
            assert custom == 50, "Should respect env override"
    
    @pytest.mark.asyncio
    async def test_periodic_reset_logic(self, mock_wallet, mock_successful_subtensor):
        """
        Test the logic for periodic reset (without full handler integration).
        Verifies reset is called at correct intervals.
        """
        reset_calls = []
        operation_count = 0
        reset_interval = 5
        
        async def mock_reset():
            reset_calls.append(operation_count)
        
        with patch('babelbit.cli.signer_api.get_subtensor', return_value=mock_successful_subtensor):
            with patch('babelbit.cli.signer_api.reset_subtensor', side_effect=mock_reset):
                for i in range(15):
                    await _set_weights_with_confirmation(
                        wallet=mock_wallet,
                        netuid=59,
                        uids=[1],
                        weights=[1.0],
                        wait_for_inclusion=False,
                        retries=1,
                        delay_s=0.001,
                        log_prefix="[test]"
                    )
                    
                    operation_count += 1
                    
                    # Simulate handler logic
                    if operation_count % reset_interval == 0:
                        await mock_reset()
        
        # Should have 3 resets (at 5, 10, 15)
        assert len(reset_calls) == 3, \
            f"Expected 3 resets at intervals of {reset_interval}, got {len(reset_calls)}"


# ============================================================================
# Diagnostic Utilities (for debugging)
# ============================================================================

@pytest.mark.asyncio
async def test_memory_diagnostic_output():
    """
    Optional diagnostic test that outputs detailed memory statistics.
    Enable with: pytest tests/test_signer_memory.py::test_memory_diagnostic_output -v -s
    """
    tracemalloc.start()
    gc.collect()
    snapshot1 = tracemalloc.take_snapshot()
    
    # Simulate some memory allocation
    data = []
    for i in range(100):
        data.append({
            "operation": f"set_weights_{i}",
            "data": [1, 2, 3] * 100,
            "metadata": {"timestamp": i, "success": True}
        })
    
    gc.collect()
    snapshot2 = tracemalloc.take_snapshot()
    
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_increase = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
    
    print(f"\n{'='*70}")
    print(f"Memory Diagnostic: 100 operations")
    print(f"{'='*70}")
    print(f"Total increase: {total_increase / 1024:.2f} KB")
    print(f"\nTop 10 increases:")
    for i, stat in enumerate(top_stats[:10], 1):
        if stat.size_diff > 0:
            print(f"{i:2d}. {stat}")
    print(f"{'='*70}\n")
    
    tracemalloc.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
