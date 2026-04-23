import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from babelbit.cli.runner import _build_round2_prediction_callback, _wait_for_round2_routes_health
from babelbit.schemas.prediction import BBPredictedUtterance, BBPredictOutput
from babelbit.utils.managed_container_registry import ManagedRoute
from babelbit.utils.miner_registry import Miner


class _Session:
    def __init__(self):
        self.get_calls = 0

    def get(self, *args, **kwargs):
        self.get_calls += 1
        raise AssertionError("HTTP health probe should not be called for provider routes")


@pytest.mark.asyncio
async def test_wait_for_round2_routes_health_bypasses_gateway_provider_routes():
    route = ManagedRoute(
        miner_hotkey="hk1",
        miner_uid=1,
        endpoint_url="https://scoring.babelbit.ai/runsync",
        provider="gateway",
    )
    settings = type("S", (), {"BB_MINER_PREDICT_ENDPOINT": "predict"})()
    session = _Session()

    with patch("babelbit.cli.runner.get_settings", return_value=settings), \
         patch("babelbit.cli.runner.get_async_client", new_callable=AsyncMock, return_value=session):
        healthy, unresolved = await _wait_for_round2_routes_health(
            routes_by_hotkey={"hk1": route},
            max_wait_seconds=1.0,
            ping_timeout_seconds=0.1,
            ping_interval_seconds=0.0,
        )

    assert healthy == {"hk1"}
    assert unresolved == {}
    assert session.get_calls == 0


@pytest.mark.asyncio
async def test_round2_prediction_callback_retries_first_prediction_until_success():
    route = ManagedRoute(
        miner_hotkey="hk1",
        miner_uid=1,
        endpoint_url="https://scoring.babelbit.ai/runsync",
        provider="gateway",
    )
    miner = Miner(uid=1, hotkey="hk1", block=1)
    payload = BBPredictedUtterance(index="s1", step=0, prefix="Hello", context="", done=False)
    result = BBPredictOutput(
        success=True,
        model="gateway",
        utterance=BBPredictedUtterance(index="s1", step=0, prefix="Hello", prediction="ok"),
        error=None,
        context_used="",
        complete=True,
    )
    timeouts = []
    attempts = {"count": 0}

    async def fake_call_managed_route_endpoint(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        attempts["count"] += 1
        if attempts["count"] == 1:
            return BBPredictOutput(
                success=True,
                model="gateway",
                utterance=BBPredictedUtterance(index="s1", step=0, prefix="Hello", prediction=""),
                error=None,
                context_used="",
                complete=False,
            )
        if attempts["count"] == 2:
            return BBPredictOutput(
                success=False,
                model="gateway",
                utterance=BBPredictedUtterance(index="s1", step=0, prefix="Hello", prediction=""),
                error="504:stream timeout",
                context_used="",
                complete=False,
            )
        return result

    with patch("babelbit.utils.predict_engine.call_managed_route_endpoint", new=AsyncMock(side_effect=fake_call_managed_route_endpoint)):
        callback = _build_round2_prediction_callback(
            routes_by_hotkey={"hk1": route},
            miner_timeout=11.0,
            startup_timeout=300.0,
            startup_request_timeout=60.0,
        )

        first = await callback(miner, payload, "ctx")
        second = await callback(miner, payload, "ctx")

    assert first == "ok"
    assert second == "ok"
    assert attempts["count"] == 4
    assert timeouts == [60.0, 60.0, 60.0, 11.0]


@pytest.mark.asyncio
async def test_round2_prediction_callback_spends_startup_budget_only_once_per_miner():
    route = ManagedRoute(
        miner_hotkey="hk1",
        miner_uid=1,
        endpoint_url="https://scoring.babelbit.ai/runsync",
        provider="gateway",
    )
    miner = Miner(uid=1, hotkey="hk1", block=1)
    payload = BBPredictedUtterance(index="s1", step=0, prefix="Hello", context="", done=False)
    timeouts = []
    attempts = {"count": 0}

    async def fake_call_managed_route_endpoint(*args, **kwargs):
        attempts["count"] += 1
        timeouts.append(kwargs["timeout"])
        await asyncio.sleep(0.02)
        return BBPredictOutput(
            success=False,
            model="gateway",
            utterance=BBPredictedUtterance(index="s1", step=0, prefix="Hello", prediction=""),
            error="504:stream timeout",
            context_used="",
            complete=False,
        )

    with patch("babelbit.utils.predict_engine.call_managed_route_endpoint", new=AsyncMock(side_effect=fake_call_managed_route_endpoint)):
        callback = _build_round2_prediction_callback(
            routes_by_hotkey={"hk1": route},
            miner_timeout=11.0,
            startup_timeout=1.0,
            startup_request_timeout=0.2,
        )

        with pytest.raises(RuntimeError, match="504:stream timeout"):
            await callback(miner, payload, "ctx")

        startup_attempts = attempts["count"]

        with pytest.raises(RuntimeError, match="504:stream timeout"):
            await callback(miner, payload, "ctx")

    assert startup_attempts >= 1
    assert attempts["count"] == startup_attempts + 1
    assert all(timeout == 0.2 for timeout in timeouts[:startup_attempts])
    assert timeouts[-1] == 11.0


@pytest.mark.asyncio
async def test_round2_prediction_callback_cancellation_preserves_startup_budget():
    route = ManagedRoute(
        miner_hotkey="hk1",
        miner_uid=1,
        endpoint_url="https://scoring.babelbit.ai/runsync",
        provider="gateway",
    )
    miner = Miner(uid=1, hotkey="hk1", block=1)
    payload = BBPredictedUtterance(index="s1", step=0, prefix="Hello", context="", done=False)
    timeouts = []
    attempts = {"count": 0}

    async def fake_call_managed_route_endpoint(*args, **kwargs):
        attempts["count"] += 1
        timeouts.append(kwargs["timeout"])
        if attempts["count"] == 1:
            await asyncio.sleep(0.05)
        return BBPredictOutput(
            success=True,
            model="gateway",
            utterance=BBPredictedUtterance(index="s1", step=0, prefix="Hello", prediction="ok"),
            error=None,
            context_used="",
            complete=True,
        )

    with patch("babelbit.utils.predict_engine.call_managed_route_endpoint", new=AsyncMock(side_effect=fake_call_managed_route_endpoint)):
        callback = _build_round2_prediction_callback(
            routes_by_hotkey={"hk1": route},
            miner_timeout=11.0,
            startup_timeout=1.0,
            startup_request_timeout=0.2,
        )

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(callback(miner, payload, "ctx"), timeout=0.01)

        second = await callback(miner, payload, "ctx")

    assert second == "ok"
    assert attempts["count"] == 2
    assert timeouts == [0.2, 0.2]
