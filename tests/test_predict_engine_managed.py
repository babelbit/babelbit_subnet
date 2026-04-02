import asyncio
from unittest.mock import Mock, AsyncMock, patch

import pytest

from babelbit.schemas.prediction import BBPredictedUtterance
from babelbit.utils.managed_container_registry import ManagedRoute
from babelbit.utils.predict_engine import call_managed_container_endpoint, call_managed_route_endpoint


class _Response:
    def __init__(self, status: int, text: str):
        self.status = status
        self._text = text

    async def text(self):
        return self._text


class _PostContext:
    def __init__(self, response=None, enter_exc=None):
        self._response = response
        self._enter_exc = enter_exc

    async def __aenter__(self):
        if self._enter_exc:
            raise self._enter_exc
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Session:
    def __init__(self, response=None, responses=None, enter_exc=None):
        self._response = response
        self._responses = list(responses or [])
        self._enter_exc = enter_exc
        self.last_post_kwargs = None
        self.calls = []

    def post(self, *args, **kwargs):
        self.last_post_kwargs = {"args": args, "kwargs": kwargs}
        self.calls.append(self.last_post_kwargs)
        response = self._responses.pop(0) if self._responses else self._response
        return _PostContext(response=response, enter_exc=self._enter_exc)


@pytest.fixture
def payload():
    return BBPredictedUtterance(
        index="session-1",
        step=0,
        prefix="Hello",
        prediction="",
        context="",
        done=False,
    )


@pytest.fixture
def mock_settings():
    settings = Mock()
    settings.BB_ARENA_MINER_TIMEOUT_SEC = 11
    settings.BB_MINER_TIMEOUT_SEC = 10
    settings.BB_MINER_PREDICT_ENDPOINT = "predict"
    settings.BB_ARENA_RUNSYNC_API_PATH = "/runsync"
    settings.BB_ARENA_GATEWAY_AUTH_API_PATH = "/auth/token"
    return settings


@pytest.mark.asyncio
async def test_call_managed_container_endpoint_success(payload, mock_settings):
    session = _Session(response=_Response(200, '{"prediction": "world"}'))

    with patch("babelbit.utils.predict_engine.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session):
        result = await call_managed_container_endpoint(
            endpoint_url="http://managed:9000/predict",
            payload=payload,
            context_used="ctx",
            miner_hotkey="hk1",
            timeout=5,
        )

    assert result.success is True
    assert result.utterance is not None
    assert result.utterance.prediction == "world"
    assert session.last_post_kwargs is not None
    assert session.last_post_kwargs["args"][0] == "http://managed:9000/predict"


@pytest.mark.asyncio
async def test_call_managed_container_endpoint_non_200(payload, mock_settings):
    session = _Session(response=_Response(503, "upstream unavailable"))

    with patch("babelbit.utils.predict_engine.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session):
        result = await call_managed_container_endpoint(
            endpoint_url="http://managed:9000/predict",
            payload=payload,
            context_used="ctx",
            miner_hotkey="hk1",
            timeout=5,
        )

    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("503:")


@pytest.mark.asyncio
async def test_call_managed_container_endpoint_timeout(payload, mock_settings):
    session = _Session(enter_exc=asyncio.TimeoutError())

    with patch("babelbit.utils.predict_engine.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session):
        result = await call_managed_container_endpoint(
            endpoint_url="http://managed:9000/predict",
            payload=payload,
            context_used="ctx",
            miner_hotkey="hk1",
            timeout=5,
        )

    assert result.success is False
    assert result.error is not None
    assert "timeout" in result.error


@pytest.mark.asyncio
async def test_call_managed_container_endpoint_appends_predict_for_base_url(payload, mock_settings):
    session = _Session(response=_Response(200, '{"prediction": "ok"}'))

    with patch("babelbit.utils.predict_engine.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session):
        result = await call_managed_container_endpoint(
            endpoint_url="https://serv-u-1.serverless.targon.com",
            payload=payload,
            context_used="ctx",
            miner_hotkey="hk1",
            timeout=5,
        )

    assert result.success is True
    assert session.last_post_kwargs is not None
    assert session.last_post_kwargs["args"][0] == "https://serv-u-1.serverless.targon.com/predict"


@pytest.mark.asyncio
async def test_call_managed_container_endpoint_appends_predict_for_non_predict_path(payload, mock_settings):
    session = _Session(response=_Response(200, '{"prediction": "ok"}'))

    with patch("babelbit.utils.predict_engine.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session):
        result = await call_managed_container_endpoint(
            endpoint_url="https://serv-u-1.serverless.targon.com/base",
            payload=payload,
            context_used="ctx",
            miner_hotkey="hk1",
            timeout=5,
        )

    assert result.success is True
    assert session.last_post_kwargs is not None
    assert session.last_post_kwargs["args"][0] == "https://serv-u-1.serverless.targon.com/base/predict"


class _FakeKeypair:
    ss58_address = "5FakeValidatorHotkey"

    def sign(self, _message):
        return b"\x01" * 64


@pytest.mark.asyncio
async def test_call_managed_route_endpoint_gateway_success(payload, mock_settings):
    from babelbit.utils import predict_engine as pe
    pe._GATEWAY_AUTH_TOKEN_CACHE.clear()

    session = _Session(
        responses=[
            _Response(200, '{"status":"ok","auth_token":"gateway-token","expires_in":1800}'),
            _Response(200, '{"status":"COMPLETED","output":{"prediction":"gateway-ok"}}'),
        ]
    )
    route = ManagedRoute(
        miner_hotkey="hk1",
        miner_uid=7,
        endpoint_url="https://scoring.babelbit.ai/runsync",
        provider="gateway",
    )
    validator_identity = {
        "keypair": _FakeKeypair(),
        "hotkey": _FakeKeypair.ss58_address,
        "external_ip": "1.2.3.4",
        "uuid": "validator-uuid",
    }

    with patch("babelbit.utils.predict_engine.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session), \
         patch("babelbit.utils.predict_engine._get_validator_identity", return_value=validator_identity):
        result = await call_managed_route_endpoint(
            route=route,
            payload=payload,
            context_used="ctx",
            miner_hotkey="hk1",
            timeout=5,
        )

    assert result.success is True
    assert result.utterance is not None
    assert result.utterance.prediction == "gateway-ok"
    assert session.last_post_kwargs is not None
    assert session.last_post_kwargs["args"][0] == "https://scoring.babelbit.ai/runsync"
    assert len(session.calls) == 2
    assert session.calls[0]["args"][0] == "https://scoring.babelbit.ai/auth/token"
    assert session.calls[1]["args"][0] == "https://scoring.babelbit.ai/runsync"
    auth_posted_json = session.calls[0]["kwargs"]["json"]
    assert auth_posted_json["hotkey"] == _FakeKeypair.ss58_address


@pytest.mark.asyncio
async def test_call_managed_route_endpoint_gateway_rejects_empty_prediction(payload, mock_settings):
    from babelbit.utils import predict_engine as pe
    pe._GATEWAY_AUTH_TOKEN_CACHE.clear()

    session = _Session(
        responses=[
            _Response(200, '{"status":"ok","auth_token":"gateway-token","expires_in":1800}'),
            _Response(200, '{"status":"COMPLETED","output":{}}'),
        ]
    )
    route = ManagedRoute(
        miner_hotkey="hk1",
        miner_uid=7,
        endpoint_url="https://scoring.babelbit.ai/runsync",
        provider="gateway",
    )
    validator_identity = {
        "keypair": _FakeKeypair(),
        "hotkey": _FakeKeypair.ss58_address,
        "external_ip": "1.2.3.4",
        "uuid": "validator-uuid",
    }

    with patch("babelbit.utils.predict_engine.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session), \
         patch("babelbit.utils.predict_engine._get_validator_identity", return_value=validator_identity):
        result = await call_managed_route_endpoint(
            route=route,
            payload=payload,
            context_used="ctx",
            miner_hotkey="hk1",
            timeout=5,
        )

    assert result.success is False
    assert result.error is not None
    assert result.error.startswith("empty_prediction_from_gateway")
    posted_json = session.last_post_kwargs["kwargs"]["json"]
    assert posted_json["auth_token"] == "gateway-token"
    assert posted_json["uid"] == 7
    assert posted_json["miner_hotkey"] == "hk1"
    assert posted_json["input"]["predict_payload"]["prefix"] == payload.prefix
    assert posted_json["input"]["bt_headers"]["bt_header_axon_hotkey"] == "hk1"
    assert posted_json["input"]["bt_headers"]["bt_header_dendrite_hotkey"] == _FakeKeypair.ss58_address
    assert posted_json["input"]["bt_headers"]["bt_header_dendrite_signature"].startswith("0x")


@pytest.mark.asyncio
async def test_call_managed_route_endpoint_gateway_requires_uid(payload, mock_settings):
    from babelbit.utils import predict_engine as pe
    pe._GATEWAY_AUTH_TOKEN_CACHE.clear()

    route = ManagedRoute(
        miner_hotkey="hk1",
        endpoint_url="https://scoring.babelbit.ai/runsync",
        provider="gateway",
    )
    validator_identity = {
        "keypair": _FakeKeypair(),
        "hotkey": _FakeKeypair.ss58_address,
        "external_ip": "1.2.3.4",
        "uuid": "validator-uuid",
    }
    session = _Session(response=_Response(200, '{"status":"COMPLETED","output":{"prediction":"gateway-ok"}}'))

    with patch("babelbit.utils.predict_engine.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session), \
         patch("babelbit.utils.predict_engine._get_validator_identity", return_value=validator_identity):
        result = await call_managed_route_endpoint(
            route=route,
            payload=payload,
            context_used="ctx",
            miner_hotkey="hk1",
            timeout=5,
        )

    assert result.success is False
    assert result.error == "missing_miner_uid_for_gateway"
    assert session.last_post_kwargs is None


@pytest.mark.asyncio
async def test_call_managed_route_endpoint_gateway_reuses_cached_auth_token(payload, mock_settings):
    from babelbit.utils import predict_engine as pe
    pe._GATEWAY_AUTH_TOKEN_CACHE.clear()

    session = _Session(
        responses=[
            _Response(200, '{"status":"ok","auth_token":"gateway-token","expires_in":1800}'),
            _Response(200, '{"status":"COMPLETED","output":{"prediction":"first"}}'),
            _Response(200, '{"status":"COMPLETED","output":{"prediction":"second"}}'),
        ]
    )
    route = ManagedRoute(
        miner_hotkey="hk1",
        miner_uid=7,
        endpoint_url="https://scoring.babelbit.ai/runsync",
        provider="gateway",
    )
    validator_identity = {
        "keypair": _FakeKeypair(),
        "hotkey": _FakeKeypair.ss58_address,
        "external_ip": "1.2.3.4",
        "uuid": "validator-uuid",
    }

    with patch("babelbit.utils.predict_engine.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session), \
         patch("babelbit.utils.predict_engine._get_validator_identity", return_value=validator_identity):
        result_1 = await call_managed_route_endpoint(
            route=route,
            payload=payload,
            context_used="ctx",
            miner_hotkey="hk1",
            timeout=5,
        )
        result_2 = await call_managed_route_endpoint(
            route=route,
            payload=payload,
            context_used="ctx",
            miner_hotkey="hk1",
            timeout=5,
        )

    assert result_1.success is True
    assert result_2.success is True
    assert len(session.calls) == 3
    assert session.calls[0]["args"][0] == "https://scoring.babelbit.ai/auth/token"
    assert session.calls[1]["args"][0] == "https://scoring.babelbit.ai/runsync"
    assert session.calls[2]["args"][0] == "https://scoring.babelbit.ai/runsync"


@pytest.mark.asyncio
async def test_call_managed_route_endpoint_gateway_auth_retries_legacy_hotkey_on_422(payload, mock_settings):
    from babelbit.utils import predict_engine as pe
    pe._GATEWAY_AUTH_TOKEN_CACHE.clear()

    session = _Session(
        responses=[
            _Response(422, '{"detail":[{"type":"extra_forbidden","loc":["body","miner_hotkey"],"msg":"Extra inputs are not permitted"},{"type":"extra_forbidden","loc":["body","uid"],"msg":"Extra inputs are not permitted"}]}'),
            _Response(200, '{"status":"ok","auth_token":"gateway-token","expires_in":1800}'),
            _Response(200, '{"status":"COMPLETED","output":{"prediction":"gateway-ok"}}'),
        ]
    )
    route = ManagedRoute(
        miner_hotkey="hk1",
        miner_uid=7,
        endpoint_url="https://scoring.babelbit.ai/runsync",
        provider="gateway",
    )
    validator_identity = {
        "keypair": _FakeKeypair(),
        "hotkey": _FakeKeypair.ss58_address,
        "external_ip": "1.2.3.4",
        "uuid": "validator-uuid",
    }

    with patch("babelbit.utils.predict_engine.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session), \
         patch("babelbit.utils.predict_engine._get_validator_identity", return_value=validator_identity):
        result = await call_managed_route_endpoint(
            route=route,
            payload=payload,
            context_used="ctx",
            miner_hotkey="hk1",
            timeout=5,
        )

    assert result.success is True
    assert len(session.calls) == 3
    first_auth_json = session.calls[0]["kwargs"]["json"]
    second_auth_json = session.calls[1]["kwargs"]["json"]
    assert first_auth_json["hotkey"] == _FakeKeypair.ss58_address
    assert first_auth_json["miner_hotkey"] == "hk1"
    assert first_auth_json["uid"] == 7
    assert second_auth_json["hotkey"] == _FakeKeypair.ss58_address
    assert "miner_hotkey" not in second_auth_json
    assert "uid" not in second_auth_json


@pytest.mark.asyncio
async def test_call_managed_route_endpoint_gateway_auth_cache_is_scoped_per_miner(payload, mock_settings):
    from babelbit.utils import predict_engine as pe
    pe._GATEWAY_AUTH_TOKEN_CACHE.clear()

    session = _Session(
        responses=[
            _Response(200, '{"status":"ok","auth_token":"gateway-token-1","expires_in":1800}'),
            _Response(200, '{"status":"COMPLETED","output":{"prediction":"first"}}'),
            _Response(200, '{"status":"ok","auth_token":"gateway-token-2","expires_in":1800}'),
            _Response(200, '{"status":"COMPLETED","output":{"prediction":"second"}}'),
        ]
    )
    route_1 = ManagedRoute(
        miner_hotkey="hk1",
        miner_uid=7,
        endpoint_url="https://scoring.babelbit.ai/runsync",
        provider="gateway",
    )
    route_2 = ManagedRoute(
        miner_hotkey="hk2",
        miner_uid=8,
        endpoint_url="https://scoring.babelbit.ai/runsync",
        provider="gateway",
    )
    validator_identity = {
        "keypair": _FakeKeypair(),
        "hotkey": _FakeKeypair.ss58_address,
        "external_ip": "1.2.3.4",
        "uuid": "validator-uuid",
    }

    with patch("babelbit.utils.predict_engine.get_settings", return_value=mock_settings), \
         patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session), \
         patch("babelbit.utils.predict_engine._get_validator_identity", return_value=validator_identity):
        result_1 = await call_managed_route_endpoint(
            route=route_1,
            payload=payload,
            context_used="ctx",
            miner_hotkey="hk1",
            timeout=5,
        )
        result_2 = await call_managed_route_endpoint(
            route=route_2,
            payload=payload,
            context_used="ctx",
            miner_hotkey="hk2",
            timeout=5,
        )

    assert result_1.success is True
    assert result_2.success is True
    assert len(session.calls) == 4
    assert session.calls[0]["args"][0] == "https://scoring.babelbit.ai/auth/token"
    assert session.calls[1]["args"][0] == "https://scoring.babelbit.ai/runsync"
    assert session.calls[2]["args"][0] == "https://scoring.babelbit.ai/auth/token"
    assert session.calls[3]["args"][0] == "https://scoring.babelbit.ai/runsync"
