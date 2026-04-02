import pytest
from unittest.mock import AsyncMock, patch

from babelbit.utils.managed_container_registry import fetch_live_containers, resolve_round2_routes
from babelbit.utils.miner_registry import Miner


class _MockResponse:
    def __init__(self, status, payload=None, text_data=""):
        self.status = status
        self._payload = payload
        self._text_data = text_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    async def text(self):
        return self._text_data


class _MockSession:
    def __init__(self, response):
        self._response = response
        self.last_get_kwargs = None

    def get(self, *args, **kwargs):
        self.last_get_kwargs = {"args": args, "kwargs": kwargs}
        return self._response


class _SequenceSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if not self._responses:
            raise AssertionError("Unexpected extra request in _SequenceSession")
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_fetch_live_containers_validates_payload_shape():
    payload = {
        "status": "ok",
        "count": 1,
        "miners": [
            {
                "hotkey": "hk1",
                "uid": 1,
            }
        ],
    }
    session = _MockSession(_MockResponse(200, payload=payload))

    mock_settings = type(
        "S",
        (),
        {
            "BB_SUBMIT_API_URL": "http://containers-api:8080",
            "BB_ARENA_CONTAINERS_API_PATH": "/list_arena_miners",
            "BB_ARENA_CONTAINERS_STATUS": "running",
            "BB_ARENA_CONTAINERS_WINDOW_SECONDS": 300,
            "BB_ARENA_CONTAINERS_TIMEOUT_SEC": 10,
            "BB_ARENA_RUNSYNC_API_PATH": "/runsync",
            "BB_MINER_PREDICT_ENDPOINT": "predict",
        },
    )()

    with patch("babelbit.utils.managed_container_registry.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.managed_container_registry.get_async_client", new_callable=AsyncMock, return_value=session):
        containers = await fetch_live_containers(
            path="/list_arena_miners",
            window_seconds=123,
            timeout=5,
        )

    assert len(containers) == 1
    assert containers[0]["miner_hotkey"] == "hk1"
    assert containers[0]["miner_uid"] == 1
    assert containers[0]["provider"] == "gateway"
    assert containers[0]["endpoint_url"] == "http://containers-api:8080/runsync"
    assert session.last_get_kwargs is not None
    assert session.last_get_kwargs["kwargs"]["params"]["window_seconds"] == 123


@pytest.mark.asyncio
async def test_fetch_live_containers_normalizes_legacy_live_containers_path():
    payload = {
        "status": "ok",
        "count": 1,
        "miners": [
            {
                "hotkey": "hk1",
                "uid": 1,
            }
        ],
    }
    session = _MockSession(_MockResponse(200, payload=payload))

    mock_settings = type(
        "S",
        (),
        {
            "BB_SUBMIT_API_URL": "http://containers-api:8080",
            "BB_ARENA_CONTAINERS_API_PATH": "/live_containers",
            "BB_ARENA_CONTAINERS_STATUS": "running",
            "BB_ARENA_CONTAINERS_WINDOW_SECONDS": 300,
            "BB_ARENA_CONTAINERS_TIMEOUT_SEC": 10,
            "BB_ARENA_RUNSYNC_API_PATH": "/runsync",
            "BB_MINER_PREDICT_ENDPOINT": "predict",
        },
    )()

    with patch("babelbit.utils.managed_container_registry.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.managed_container_registry.get_async_client", new_callable=AsyncMock, return_value=session):
        containers = await fetch_live_containers(timeout=5)

    assert len(containers) == 1
    assert session.last_get_kwargs is not None
    assert session.last_get_kwargs["args"][0] == "http://containers-api:8080/list_arena_miners"


@pytest.mark.asyncio
async def test_fetch_live_containers_uses_round2_gateway_base_url_override():
    payload = {
        "status": "ok",
        "count": 1,
        "miners": [
            {
                "hotkey": "hk1",
                "uid": 1,
            }
        ],
    }
    session = _MockSession(_MockResponse(200, payload=payload))

    mock_settings = type(
        "S",
        (),
        {
            "BB_SUBMIT_API_URL": "http://submit-api:8000",
            "BB_ARENA_GATEWAY_URL": "http://gateway-api:8079",
            "BB_ARENA_CONTAINERS_API_PATH": "/list_arena_miners",
            "BB_ARENA_CONTAINERS_STATUS": "running",
            "BB_ARENA_CONTAINERS_WINDOW_SECONDS": 300,
            "BB_ARENA_CONTAINERS_TIMEOUT_SEC": 10,
            "BB_ARENA_RUNSYNC_API_PATH": "/runsync",
            "BB_MINER_PREDICT_ENDPOINT": "predict",
        },
    )()

    with patch("babelbit.utils.managed_container_registry.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.managed_container_registry.get_async_client", new_callable=AsyncMock, return_value=session):
        containers = await fetch_live_containers(timeout=5)

    assert len(containers) == 1
    assert containers[0]["endpoint_url"] == "http://gateway-api:8079/runsync"
    assert session.last_get_kwargs is not None
    assert session.last_get_kwargs["args"][0] == "http://gateway-api:8079/list_arena_miners"


@pytest.mark.asyncio
async def test_fetch_live_containers_prefers_non_duplicated_v1_path():
    payload = {
        "status": "ok",
        "count": 1,
        "miners": [
            {
                "hotkey": "hk1",
                "uid": 1,
            }
        ],
    }
    session = _SequenceSession([_MockResponse(200, payload=payload)])
    mock_settings = type(
        "S",
        (),
        {
            "BB_SUBMIT_API_URL": "http://containers-api:8080/v1",
            "BB_ARENA_CONTAINERS_API_PATH": "/v1/list_arena_miners",
            "BB_ARENA_CONTAINERS_STATUS": "running",
            "BB_ARENA_CONTAINERS_WINDOW_SECONDS": 300,
            "BB_ARENA_CONTAINERS_TIMEOUT_SEC": 10,
            "BB_ARENA_RUNSYNC_API_PATH": "/runsync",
            "BB_MINER_PREDICT_ENDPOINT": "predict",
        },
    )()

    with patch("babelbit.utils.managed_container_registry.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.managed_container_registry.get_async_client", new_callable=AsyncMock, return_value=session):
        containers = await fetch_live_containers(timeout=5)

    assert len(containers) == 1
    assert len(session.calls) == 1
    assert session.calls[0]["args"][0] == "http://containers-api:8080/v1/list_arena_miners"


@pytest.mark.asyncio
async def test_fetch_live_containers_falls_back_to_v1_path_on_404():
    payload = {
        "status": "ok",
        "count": 1,
        "miners": [
            {
                "hotkey": "hk1",
                "uid": 1,
            }
        ],
    }
    session = _SequenceSession([
        _MockResponse(404, payload={"detail": "Not Found"}, text_data='{"detail":"Not Found"}'),
        _MockResponse(200, payload=payload),
    ])
    mock_settings = type(
        "S",
        (),
        {
            "BB_SUBMIT_API_URL": "http://containers-api:8080",
            "BB_ARENA_CONTAINERS_API_PATH": "/list_arena_miners",
            "BB_ARENA_CONTAINERS_STATUS": "running",
            "BB_ARENA_CONTAINERS_WINDOW_SECONDS": 300,
            "BB_ARENA_CONTAINERS_TIMEOUT_SEC": 10,
            "BB_ARENA_RUNSYNC_API_PATH": "/runsync",
            "BB_MINER_PREDICT_ENDPOINT": "predict",
        },
    )()

    with patch("babelbit.utils.managed_container_registry.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.managed_container_registry.get_async_client", new_callable=AsyncMock, return_value=session):
        containers = await fetch_live_containers(timeout=5)

    assert len(containers) == 1
    assert len(session.calls) == 2
    assert session.calls[0]["args"][0] == "http://containers-api:8080/list_arena_miners"
    assert session.calls[1]["args"][0] == "http://containers-api:8080/v1/list_arena_miners"


@pytest.mark.asyncio
async def test_fetch_live_containers_uses_single_call_for_arena_miners_payload():
    payload = {
        "status": "ok",
        "count": 2,
        "miners": [
            {"hotkey": "hk1", "uid": 1},
            {"hotkey": "hk2", "uid": 2},
        ],
    }

    class _MultiStatusSession:
        def __init__(self):
            self.called_statuses = []

        def get(self, *args, **kwargs):
            status_value = kwargs["params"].get("status", "")
            self.called_statuses.append(status_value)
            return _MockResponse(200, payload=payload)

    session = _MultiStatusSession()
    mock_settings = type(
        "S",
        (),
        {
            "BB_SUBMIT_API_URL": "http://containers-api:8080",
            "BB_ARENA_CONTAINERS_API_PATH": "/list_arena_miners",
            "BB_ARENA_CONTAINERS_STATUS": "running",
            "BB_ARENA_CONTAINERS_WINDOW_SECONDS": 300,
            "BB_ARENA_CONTAINERS_TIMEOUT_SEC": 10,
            "BB_ARENA_RUNSYNC_API_PATH": "/runsync",
            "BB_MINER_PREDICT_ENDPOINT": "predict",
        },
    )()

    with patch("babelbit.utils.managed_container_registry.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.managed_container_registry.get_async_client", new_callable=AsyncMock, return_value=session):
        containers = await fetch_live_containers(status="running,terminated", timeout=5)

    assert len(containers) == 2
    assert len(session.called_statuses) == 1
    assert session.called_statuses[0] == "running"


@pytest.mark.asyncio
async def test_resolve_round2_routes_intersects_with_onchain_miners():
    miners = {
        1: Miner(uid=1, hotkey="hk1", block=100, axon_ip="1.1.1.1", axon_port=8091),
        2: Miner(uid=2, hotkey="hk2", block=101, axon_ip="1.1.1.2", axon_port=8091),
    }

    containers = [
        {
            "miner_hotkey": "hk1",
            "endpoint_url": "http://managed-1:9000/predict",
            "status": "running",
            "container_name": "ctr-hk1",
        },
        {
            "miner_hotkey": "hk3",
            "endpoint_url": "http://managed-3:9000/predict",
            "status": "running",
        },
        {
            # malformed row (no endpoint)
            "miner_hotkey": "hk2",
            "status": "running",
        },
    ]

    with patch("babelbit.utils.managed_container_registry.get_miners_from_registry", new_callable=AsyncMock, return_value=miners):
        round2_miners, routes_by_hotkey = await resolve_round2_routes(netuid=42, containers=containers)

    assert [m.hotkey for m in round2_miners] == ["hk1"]
    assert "hk1" in routes_by_hotkey
    assert routes_by_hotkey["hk1"].endpoint_url == "http://managed-1:9000/predict"
    assert routes_by_hotkey["hk1"].provider == "managed_container"
    assert "hk2" not in routes_by_hotkey


@pytest.mark.asyncio
async def test_resolve_round2_routes_accepts_gateway_routes_from_arena_miners():
    miners = {
        1: Miner(uid=1, hotkey="hk1", block=100, axon_ip="1.1.1.1", axon_port=8091),
        2: Miner(uid=2, hotkey="hk2", block=101, axon_ip="1.1.1.2", axon_port=8091),
    }
    containers = [
        {
            "miner_hotkey": "hk1",
            "miner_uid": 1,
            "provider": "gateway",
            "endpoint_url": "https://scoring.babelbit.ai/runsync",
            "status": "running",
        },
        {
            "miner_hotkey": "hk3",
            "miner_uid": 3,
            "provider": "gateway",
            "endpoint_url": "https://scoring.babelbit.ai/runsync",
            "status": "running",
        },
    ]

    with patch("babelbit.utils.managed_container_registry.get_miners_from_registry", new_callable=AsyncMock, return_value=miners):
        round2_miners, routes_by_hotkey = await resolve_round2_routes(netuid=42, containers=containers)

    assert [m.hotkey for m in round2_miners] == ["hk1"]
    assert routes_by_hotkey["hk1"].provider == "gateway"
    assert routes_by_hotkey["hk1"].miner_uid == 1
    assert routes_by_hotkey["hk1"].endpoint_url == "https://scoring.babelbit.ai/runsync"


@pytest.mark.asyncio
async def test_resolve_round2_routes_canonicalizes_hotkey_from_uid_match():
    miners = {
        1: Miner(uid=1, hotkey="hk1", block=100, axon_ip="1.1.1.1", axon_port=8091),
        2: Miner(uid=2, hotkey="hk2", block=101, axon_ip="1.1.1.2", axon_port=8091),
    }
    containers = [
        {
            "miner_hotkey": "stale-hk2",
            "miner_uid": 2,
            "provider": "gateway",
            "endpoint_url": "https://scoring.babelbit.ai/runsync",
            "status": "running",
        },
    ]

    with patch("babelbit.utils.managed_container_registry.get_miners_from_registry", new_callable=AsyncMock, return_value=miners):
        round2_miners, routes_by_hotkey = await resolve_round2_routes(netuid=42, containers=containers)

    assert [m.hotkey for m in round2_miners] == ["hk2"]
    assert "stale-hk2" not in routes_by_hotkey
    assert routes_by_hotkey["hk2"].provider == "gateway"
    assert routes_by_hotkey["hk2"].miner_uid == 2
    assert routes_by_hotkey["hk2"].endpoint_url == "https://scoring.babelbit.ai/runsync"


@pytest.mark.asyncio
async def test_resolve_round2_routes_accepts_url_field_from_live_containers():
    miners = {
        8: Miner(uid=8, hotkey="hk8", block=100, axon_ip="2.2.2.2", axon_port=8091),
    }
    containers = [
        {
            "miner_hotkey": "hk8",
            "url": "https://serv-u-1.serverless.targon.com",
            "status": "running",
            "container_name": "ctr-hk8",
        }
    ]

    with patch("babelbit.utils.managed_container_registry.get_miners_from_registry", new_callable=AsyncMock, return_value=miners):
        round2_miners, routes_by_hotkey = await resolve_round2_routes(netuid=42, containers=containers)

    assert [m.hotkey for m in round2_miners] == ["hk8"]
    assert routes_by_hotkey["hk8"].endpoint_url == "https://serv-u-1.serverless.targon.com"
    assert routes_by_hotkey["hk8"].provider == "managed_container"

@pytest.mark.asyncio
async def test_resolve_round2_routes_accepts_soft_terminated_endpoint_from_metadata():
    miners = {
        11: Miner(uid=11, hotkey="hk11", block=100, axon_ip="5.5.5.5", axon_port=8091),
    }
    containers = [
        {
            "miner_hotkey": "hk11",
            "status": "terminated",
            "metadata": "{\"url\":\"https://serv-u-11.serverless.targon.com/predict\",\"reason\":\"not_in_list\"}",
        }
    ]

    with patch("babelbit.utils.managed_container_registry.get_miners_from_registry", new_callable=AsyncMock, return_value=miners):
        round2_miners, routes_by_hotkey = await resolve_round2_routes(netuid=42, containers=containers)

    assert [m.hotkey for m in round2_miners] == ["hk11"]
    assert routes_by_hotkey["hk11"].provider == "managed_container"
    assert routes_by_hotkey["hk11"].endpoint_url == "https://serv-u-11.serverless.targon.com/predict"


@pytest.mark.asyncio
async def test_resolve_round2_routes_excludes_hard_terminated_endpoint():
    miners = {
        12: Miner(uid=12, hotkey="hk12", block=100, axon_ip="6.6.6.6", axon_port=8091),
    }
    containers = [
        {
            "miner_hotkey": "hk12",
            "status": "terminated",
            "metadata": "{\"url\":\"https://serv-u-12.serverless.targon.com/predict\",\"reason\":\"worker_crashed\"}",
        }
    ]

    with patch("babelbit.utils.managed_container_registry.get_miners_from_registry", new_callable=AsyncMock, return_value=miners):
        round2_miners, routes_by_hotkey = await resolve_round2_routes(netuid=42, containers=containers)

    assert round2_miners == []
    assert routes_by_hotkey == {}
