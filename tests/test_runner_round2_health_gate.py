from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from babelbit.cli.runner import (
    _build_managed_health_url,
    _wait_for_round2_routes_health,
)
from babelbit.utils.managed_container_registry import ManagedRoute


@dataclass
class _MockSettings:
    BB_MINER_PREDICT_ENDPOINT: str = "predict"


class _MockResponse:
    def __init__(self, status: int, body: str = "ok") -> None:
        self.status = status
        self._body = body

    async def __aenter__(self) -> "_MockResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def text(self) -> str:
        return self._body


class _SequencedSession:
    def __init__(self, by_url: dict[str, list[Any]]) -> None:
        self._by_url = {k: list(v) for k, v in by_url.items()}
        self._last_value = {k: (v[-1] if v else 200) for k, v in self._by_url.items()}
        self.calls: dict[str, int] = {}

    def get(self, url: str, **_kwargs) -> _MockResponse:
        self.calls[url] = self.calls.get(url, 0) + 1
        outcomes = self._by_url.get(url, [200])
        value = outcomes.pop(0) if outcomes else self._last_value.get(url, 200)
        self._by_url[url] = outcomes
        if isinstance(value, Exception):
            raise value
        return _MockResponse(status=int(value))


def test_build_managed_health_url_normalizes_predict_path() -> None:
    assert _build_managed_health_url("http://miner-1:8000/predict", "predict") == "http://miner-1:8000/health"
    assert _build_managed_health_url("https://miner-2/v1/predict", "predict") == "https://miner-2/v1/health"
    assert _build_managed_health_url("https://miner-3", "predict") == "https://miner-3/health"


@pytest.mark.asyncio
async def test_wait_for_round2_routes_health_retries_until_ready() -> None:
    routes = {
        "hk1": ManagedRoute(miner_hotkey="hk1", endpoint_url="http://miner-1:8000/predict"),
        "hk2": ManagedRoute(miner_hotkey="hk2", endpoint_url="http://miner-2:8000/predict"),
    }
    session = _SequencedSession(
        {
            "http://miner-1:8000/health": [503, 200],
            "http://miner-2:8000/health": [200],
        }
    )

    with patch("babelbit.cli.runner.get_settings", return_value=_MockSettings()), \
         patch("babelbit.cli.runner.get_async_client", return_value=session):
        healthy, unresolved = await _wait_for_round2_routes_health(
            routes_by_hotkey=routes,
            max_wait_seconds=3.0,
            ping_timeout_seconds=0.5,
            ping_interval_seconds=0.0,
        )

    assert healthy == {"hk1", "hk2"}
    assert unresolved == {}
    assert session.calls["http://miner-1:8000/health"] >= 2


@pytest.mark.asyncio
async def test_wait_for_round2_routes_health_stops_on_timeout() -> None:
    routes = {
        "hk-timeout": ManagedRoute(miner_hotkey="hk-timeout", endpoint_url="http://miner-timeout:8000/predict"),
    }
    session = _SequencedSession(
        {
            "http://miner-timeout:8000/health": [503] * 1000,
        }
    )

    with patch("babelbit.cli.runner.get_settings", return_value=_MockSettings()), \
         patch("babelbit.cli.runner.get_async_client", return_value=session):
        healthy, unresolved = await _wait_for_round2_routes_health(
            routes_by_hotkey=routes,
            max_wait_seconds=0.05,
            ping_timeout_seconds=0.01,
            ping_interval_seconds=0.0,
        )

    assert healthy == set()
    assert "hk-timeout" in unresolved
