import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class FakeGauge:
    def __init__(self):
        self.calls = []
        self._current_uid = None

    def labels(self, uid):
        self._current_uid = uid
        return self

    def set(self, value):
        self.calls.append((self._current_uid, value))


class FakeSingleGauge:
    def __init__(self):
        self.value = None

    def set(self, value):
        self.value = value


class MockResponse:
    def __init__(self, *, status=200, json_data=None, text_data=""):
        self.status = status
        self._json_data = json_data or {}
        self._text_data = text_data

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class MockSession:
    def __init__(self, *, get_response=None, post_response=None, get_exc=None, post_exc=None):
        self.get_response = get_response
        self.post_response = post_response
        self.get_exc = get_exc
        self.post_exc = post_exc
        self.get_calls = []
        self.post_calls = []

    def get(self, url, params=None, **kwargs):
        self.get_calls.append((url, params, kwargs))
        if self.get_exc:
            raise self.get_exc
        return self.get_response

    async def post(self, url, json=None):
        self.post_calls.append((url, json))
        if self.post_exc:
            raise self.post_exc
        return self.post_response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeConnectorError(Exception):
    pass


class FakeTimeout:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_validate_int_env_parser_handles_empty_and_invalid_values(monkeypatch):
    from babelbit.cli import validate as validate_mod

    monkeypatch.setenv("BB_DEFAULT_FALLBACK_UID", "")
    monkeypatch.setenv("BB_MAX_SKIPPED_WEIGHT_EPOCHS", "   ")

    assert validate_mod._get_int_env("BB_DEFAULT_FALLBACK_UID", 248) == 248
    assert validate_mod._get_int_env("BB_MAX_SKIPPED_WEIGHT_EPOCHS", 12) == 12

    monkeypatch.setenv("BB_DEFAULT_FALLBACK_UID", "not-an-int")
    assert validate_mod._get_int_env("BB_DEFAULT_FALLBACK_UID", 248) == 248


def test_reset_no_score_if_challenge_changed():
    from babelbit.cli import validate as validate_mod

    # When challenge changes, counter resets and last uid updates.
    count, last = validate_mod._reset_no_score_if_challenge_changed("chal-2", "chal-1", 5)
    assert count == 0
    assert last == "chal-2"

    # When challenge stays the same, values remain untouched.
    count, last = validate_mod._reset_no_score_if_challenge_changed("chal-2", "chal-2", 3)
    assert count == 3
    assert last == "chal-2"

    # When current is missing, leave counters alone.
    count, last = validate_mod._reset_no_score_if_challenge_changed(None, "chal-2", 4)
    assert count == 4
    assert last == "chal-2"


@pytest.mark.asyncio
async def test_get_weights_selects_winner_and_resets_counter(monkeypatch):
    """Pick the highest score from API results and reset the miss counter."""
    fake_settings = SimpleNamespace(BB_SUBMIT_API_URL="http://api")
    from babelbit.cli import validate as validate_mod

    monkeypatch.setattr(validate_mod, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(
        validate_mod,
        "fetch_scores_from_api",
        AsyncMock(
            return_value=[
                {"miner_hotkey": "hk1", "challenge_mean_score": 0.7},
                {"miner_hotkey": "hk2", "challenge_mean_score": 0.9},
                {"miner_hotkey": "hk2", "challenge_mean_score": 1.1},  # ignored (first occurrence kept)
                {"hotkey": "unknown", "challenge_mean_score": 2.0},  # not in metagraph
            ]
        ),
    )

    fake_meta = SimpleNamespace(hotkeys=["hk1", "hk2"])
    validator_kp = SimpleNamespace(ss58_address="validator-hk")

    uids, weights, no_score_rounds = await validate_mod.get_weights(
        metagraph=fake_meta,
        validator_kp=validator_kp,
        challenge_uid="chal-1",
        last_weights=None,
        no_score_rounds=3,
        max_no_score_rounds=5,
        default_uid=248,
    )

    assert uids == [1, 0]  # hk2 is the winner at index 1, hk1 trails
    assert weights == pytest.approx([0.95, 0.05])
    assert no_score_rounds == 0


@pytest.mark.asyncio
async def test_get_weights_updates_metrics_and_prefers_first_occurrence(monkeypatch):
    """Ensure metrics are updated and the first score per miner is kept."""
    fake_settings = SimpleNamespace(BB_SUBMIT_API_URL="http://api")
    from babelbit.cli import validate as validate_mod

    monkeypatch.setattr(validate_mod, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(
        validate_mod,
        "fetch_scores_from_api",
        AsyncMock(
            return_value=[
                {"hotkey": "hk1", "score": 0.4},
                {"hotkey": "hk2", "score": 0.9},
                {"hotkey": "hk2", "score": 1.2},  # ignored (first score wins)
                {"hotkey": "hk3", "score": None},  # ignored missing score
            ]
        ),
    )

    scores_gauge = FakeGauge()
    winner_gauge = FakeSingleGauge()
    monkeypatch.setattr(validate_mod, "SCORES_BY_UID", scores_gauge)
    monkeypatch.setattr(validate_mod, "CURRENT_WINNER", winner_gauge)

    uids, weights, no_score_rounds = await validate_mod.get_weights(
        metagraph=SimpleNamespace(hotkeys=["hk1", "hk2", "hk3"]),
        validator_kp=SimpleNamespace(ss58_address="validator-hk"),
        challenge_uid="chal-metrics",
        last_weights=None,
        no_score_rounds=0,
        max_no_score_rounds=5,
        default_uid=248,
    )

    assert uids == [1, 0]  # hk2 wins, hk1 trails
    assert weights == pytest.approx([0.95, 0.05])
    assert no_score_rounds == 0
    assert ("0", 0.4) in scores_gauge.calls
    assert ("1", 0.9) in scores_gauge.calls
    assert winner_gauge.value == 1


@pytest.mark.asyncio
async def test_get_weights_splits_main_and_arena_modes(monkeypatch):
    """Split 95% core incentive by mode while preserving 5% trailing."""
    fake_settings = SimpleNamespace(BB_SUBMIT_API_URL="http://api")
    from babelbit.cli import validate as validate_mod

    monkeypatch.setattr(validate_mod, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(
        validate_mod,
        "fetch_scores_from_api",
        AsyncMock(
            return_value=[
                {"miner_hotkey": "hk-main", "challenge_type": "main", "challenge_mean_score": 0.9},
                {"miner_hotkey": "hk-main", "challenge_type": "arena", "challenge_mean_score": 0.3},
                {"miner_hotkey": "hk-arena", "challenge_type": "main", "challenge_mean_score": 0.2},
                {"miner_hotkey": "hk-arena", "challenge_type": "arena", "challenge_mean_score": 0.95},
                {"miner_hotkey": "hk-tail", "challenge_type": "main", "challenge_mean_score": 0.5},
                {"miner_hotkey": "hk-tail", "challenge_type": "arena", "challenge_mean_score": 0.4},
            ]
        ),
    )

    uids, weights, no_score_rounds = await validate_mod.get_weights(
        metagraph=SimpleNamespace(hotkeys=["hk-main", "hk-arena", "hk-tail"]),
        validator_kp=SimpleNamespace(ss58_address="validator-hk"),
        challenge_uid="chal-modes",
        last_weights=None,
        no_score_rounds=0,
        max_no_score_rounds=5,
        default_uid=248,
        arena_incentive_fraction=0.4,
    )

    # Main winner uid=0 gets 57%; arena winner uid=1 gets 38%; trailing uid=2 gets 5%.
    assert uids == [0, 1, 2]
    assert weights == pytest.approx([0.57, 0.38, 0.05])
    assert no_score_rounds == 0


@pytest.mark.asyncio
async def test_get_weights_fetches_mode_specific_scores(monkeypatch):
    """Fetch main/arena mode scores separately and combine winners."""
    fake_settings = SimpleNamespace(BB_SUBMIT_API_URL="http://api")
    from babelbit.cli import validate as validate_mod

    monkeypatch.setattr(validate_mod, "get_settings", lambda: fake_settings)

    async def fake_fetch(*, base_url, validator_kp, challenge_uid, challenge_type=None):
        assert base_url == "http://api"
        assert challenge_uid == "chal-modes"
        if challenge_type == "main":
            return [{"miner_hotkey": "hk-main", "challenge_mean_score": 0.9}]
        if challenge_type == "arena":
            return [{"miner_hotkey": "hk-arena", "challenge_mean_score": 0.95}]
        return []

    fetch_mock = AsyncMock(side_effect=fake_fetch)
    monkeypatch.setattr(validate_mod, "fetch_scores_from_api", fetch_mock)

    uids, weights, no_score_rounds = await validate_mod.get_weights(
        metagraph=SimpleNamespace(hotkeys=["hk-main", "hk-arena"]),
        validator_kp=SimpleNamespace(ss58_address="validator-hk"),
        challenge_uid="chal-modes",
        last_weights=None,
        no_score_rounds=0,
        max_no_score_rounds=5,
        default_uid=248,
        arena_incentive_fraction=0.4,
    )

    assert uids == [0, 1]
    assert weights == pytest.approx([0.6, 0.4])
    assert no_score_rounds == 0
    assert fetch_mock.await_count == 2


@pytest.mark.asyncio
async def test_get_weights_routes_missing_arena_share_to_default_uid(monkeypatch):
    fake_settings = SimpleNamespace(BB_SUBMIT_API_URL="http://api")
    from babelbit.cli import validate as validate_mod

    monkeypatch.setattr(validate_mod, "get_settings", lambda: fake_settings)

    async def fake_fetch(*, base_url, validator_kp, challenge_uid, challenge_type=None):
        assert base_url == "http://api"
        assert challenge_uid == "chal-main-only"
        if challenge_type == "main":
            return [{"miner_hotkey": "hk-main", "challenge_mean_score": 0.9}]
        if challenge_type == "arena":
            return []
        return []

    monkeypatch.setattr(validate_mod, "fetch_scores_from_api", AsyncMock(side_effect=fake_fetch))

    uids, weights, no_score_rounds = await validate_mod.get_weights(
        metagraph=SimpleNamespace(hotkeys=["hk-main", "hk-other"]),
        validator_kp=SimpleNamespace(ss58_address="validator-hk"),
        challenge_uid="chal-main-only",
        last_weights=None,
        no_score_rounds=0,
        max_no_score_rounds=5,
        default_uid=248,
        arena_incentive_fraction=0.4,
    )

    assert uids == [0, 248]
    assert weights == pytest.approx([0.6, 0.4])
    assert no_score_rounds == 0


@pytest.mark.asyncio
async def test_get_weights_accepts_zero_scores(monkeypatch):
    """Zero scores should still count as valid results (not treated as missing)."""
    fake_settings = SimpleNamespace(BB_SUBMIT_API_URL="http://api")
    from babelbit.cli import validate as validate_mod

    monkeypatch.setattr(validate_mod, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(
        validate_mod,
        "fetch_scores_from_api",
        AsyncMock(return_value=[{"miner_hotkey": "hk-zero", "challenge_mean_score": 0.0}]),
    )

    uids, weights, no_score_rounds = await validate_mod.get_weights(
        metagraph=SimpleNamespace(hotkeys=["hk-zero"]),
        validator_kp=SimpleNamespace(ss58_address="validator-hk"),
        challenge_uid="chal-zero",
        last_weights=None,
        no_score_rounds=2,
        max_no_score_rounds=5,
        default_uid=248,
    )

    assert uids == [0]
    assert weights == [1.0]
    assert no_score_rounds == 0


@pytest.mark.asyncio
async def test_get_weights_reuses_last_weights_when_no_scores(monkeypatch):
    """When API returns nothing, reuse the previous weights and increment counter."""
    fake_settings = SimpleNamespace(BB_SUBMIT_API_URL="http://api")
    from babelbit.cli import validate as validate_mod

    monkeypatch.setattr(validate_mod, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(validate_mod, "fetch_scores_from_api", AsyncMock(return_value=[]))

    fake_meta = SimpleNamespace(hotkeys=["hk1", "hk2"])
    last_weights = ([0], [1.0])

    uids, weights, no_score_rounds = await validate_mod.get_weights(
        metagraph=fake_meta,
        validator_kp=SimpleNamespace(ss58_address="validator-hk"),
        challenge_uid=None,
        last_weights=last_weights,
        no_score_rounds=2,
        max_no_score_rounds=5,
        default_uid=248,
    )

    assert (uids, weights) == last_weights
    assert no_score_rounds == 3


@pytest.mark.asyncio
async def test_get_weights_waits_before_defaulting(monkeypatch):
    """Without scores or history, wait for more rounds until the limit is reached."""
    fake_settings = SimpleNamespace(BB_SUBMIT_API_URL="http://api")
    from babelbit.cli import validate as validate_mod

    monkeypatch.setattr(validate_mod, "get_settings", lambda: fake_settings)
    # API returns scores, but none match known hotkeys => treated as no scores
    monkeypatch.setattr(
        validate_mod,
        "fetch_scores_from_api",
        AsyncMock(return_value=[{"miner_hotkey": "unknown", "challenge_mean_score": 0.9}]),
    )

    uids, weights, no_score_rounds = await validate_mod.get_weights(
        metagraph=SimpleNamespace(hotkeys=["hk1"]),
        validator_kp=SimpleNamespace(ss58_address="validator-hk"),
        challenge_uid="chal-2",
        last_weights=None,
        no_score_rounds=0,
        max_no_score_rounds=2,
        default_uid=248,
    )

    assert uids == []
    assert weights == []
    assert no_score_rounds == 1


@pytest.mark.asyncio
async def test_get_weights_falls_back_to_default_after_limit(monkeypatch):
    """After exceeding the max no-score rounds, fall back to the default UID."""
    fake_settings = SimpleNamespace(BB_SUBMIT_API_URL="http://api")
    from babelbit.cli import validate as validate_mod

    monkeypatch.setattr(validate_mod, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(validate_mod, "fetch_scores_from_api", AsyncMock(return_value=[]))

    uids, weights, no_score_rounds = await validate_mod.get_weights(
        metagraph=SimpleNamespace(hotkeys=["hk1"]),
        validator_kp=SimpleNamespace(ss58_address="validator-hk"),
        challenge_uid="chal-3",
        last_weights=None,
        no_score_rounds=3,
        max_no_score_rounds=3,
        default_uid=999,
    )

    assert uids == [999]
    assert weights == [1.0]
    assert no_score_rounds == 4


@pytest.mark.asyncio
async def test_fetch_scores_from_api_success(monkeypatch):
    """Ensure fetch_scores_from_api signs and passes expected params."""
    from babelbit.cli import validate as validate_mod

    fake_session = MockSession(
        get_response=MockResponse(
            status=200,
            json_data={"scores": [{"hotkey": "hk1", "score": 0.8}]},
        )
    )
    canonical_holder = {}

    def fake_sign(kp, canonical):
        canonical_holder["payload"] = canonical
        return "sig"

    validator_kp = SimpleNamespace(ss58_address="validator-hk")
    monkeypatch.setattr(validate_mod, "get_async_client", AsyncMock(return_value=fake_session))
    monkeypatch.setattr(validate_mod, "sign_message", fake_sign)

    scores = await validate_mod.fetch_scores_from_api(
        base_url="http://submit-api",
        validator_kp=validator_kp,
        challenge_uid="chal-xyz",
    )

    assert scores == [{"hotkey": "hk1", "score": 0.8}]
    url, params, kwargs = fake_session.get_calls[0]
    assert url == "http://submit-api/v1/get_scores"
    assert params["signature"] == "sig"
    assert params["hotkey"] == validator_kp.ss58_address
    assert params["challenge_uid"] == "chal-xyz"
    assert isinstance(params["timestamp"], int)
    assert kwargs["timeout"].total == 30.0

    # Canonical payload should match the expected structure
    parsed = json.loads(canonical_holder["payload"])
    assert parsed["hotkey"] == validator_kp.ss58_address
    assert parsed["challenge_id"] == "chal-xyz"
    assert parsed["data"]["challenge_uid"] == "chal-xyz"
    assert isinstance(parsed["timestamp"], int)


@pytest.mark.asyncio
async def test_fetch_scores_from_api_includes_challenge_type(monkeypatch):
    """challenge_type should be sent as mapped query param, not signed payload."""
    from babelbit.cli import validate as validate_mod

    fake_session = MockSession(
        get_response=MockResponse(
            status=200,
            json_data={"scores": [{"hotkey": "hk-arena", "score": 0.95}]},
        )
    )
    canonical_holder = {}

    def fake_sign(kp, canonical):
        canonical_holder["payload"] = canonical
        return "sig"

    validator_kp = SimpleNamespace(ss58_address="validator-hk")
    monkeypatch.setattr(validate_mod, "get_async_client", AsyncMock(return_value=fake_session))
    monkeypatch.setattr(validate_mod, "sign_message", fake_sign)

    scores = await validate_mod.fetch_scores_from_api(
        base_url="http://submit-api",
        validator_kp=validator_kp,
        challenge_uid="chal-xyz",
        challenge_type="main",
    )

    assert scores == [{"hotkey": "hk-arena", "score": 0.95}]
    _, params, kwargs = fake_session.get_calls[0]
    assert params["challenge_type"] == "qualifying"
    assert kwargs["timeout"].total == 30.0

    parsed = json.loads(canonical_holder["payload"])
    assert parsed["data"]["challenge_uid"] == "chal-xyz"
    assert "challenge_type" not in parsed["data"]


@pytest.mark.asyncio
async def test_fetch_scores_from_api_maps_arena_challenge_type(monkeypatch):
    from babelbit.cli import validate as validate_mod

    fake_session = MockSession(
        get_response=MockResponse(
            status=200,
            json_data={"scores": [{"hotkey": "hk-arena", "score": 0.95}]},
        )
    )
    monkeypatch.setattr(validate_mod, "get_async_client", AsyncMock(return_value=fake_session))
    monkeypatch.setattr(validate_mod, "sign_message", lambda *_args, **_kwargs: "sig")

    _ = await validate_mod.fetch_scores_from_api(
        base_url="http://submit-api",
        validator_kp=SimpleNamespace(ss58_address="validator-hk"),
        challenge_uid="chal-xyz",
        challenge_type="arena",
    )

    _, params, kwargs = fake_session.get_calls[0]
    assert params["challenge_type"] == "arena"
    assert kwargs["timeout"].total == 30.0


@pytest.mark.asyncio
async def test_fetch_scores_from_api_handles_errors(monkeypatch):
    """Non-200 or request failures should return an empty list."""
    from babelbit.cli import validate as validate_mod

    validator_kp = SimpleNamespace(ss58_address="validator-hk")
    error_session = MockSession(
        get_response=MockResponse(status=500, json_data={"error": "bad"}, text_data="bad")
    )
    monkeypatch.setattr(validate_mod, "get_async_client", AsyncMock(return_value=error_session))
    monkeypatch.setattr(validate_mod, "sign_message", lambda *_args, **_kwargs: "sig")
    scores = await validate_mod.fetch_scores_from_api("http://api", validator_kp, "chal")
    assert scores == []

    failing_session = MockSession(get_exc=RuntimeError("boom"))
    monkeypatch.setattr(validate_mod, "get_async_client", AsyncMock(return_value=failing_session))
    scores = await validate_mod.fetch_scores_from_api("http://api", validator_kp, "chal")
    assert scores == []


@pytest.mark.asyncio
async def test_fetch_scores_from_api_skips_without_challenge(monkeypatch):
    """If challenge UID is missing, do nothing."""
    from babelbit.cli import validate as validate_mod

    validator_kp = SimpleNamespace(ss58_address="validator-hk")

    async def should_not_call():
        raise AssertionError("get_async_client should not be called when challenge is missing")

    monkeypatch.setattr(validate_mod, "get_async_client", should_not_call)
    scores = await validate_mod.fetch_scores_from_api("http://api", validator_kp, None)
    assert scores == []


@pytest.mark.asyncio
async def test_retry_set_weights_prefers_signer_success(monkeypatch):
    """Return True when signer responds with success and avoid fallback."""
    from babelbit.cli import validate as validate_mod

    fake_settings = SimpleNamespace(BABELBIT_NETUID=11, SIGNER_URL="http://signer")
    monkeypatch.setattr(validate_mod, "get_settings", lambda: fake_settings)

    session = MockSession(post_response=MockResponse(status=200, json_data={"success": True}))

    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda timeout: session,
        ClientConnectorError=FakeConnectorError,
        ClientTimeout=lambda **kwargs: FakeTimeout(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)

    fallback = AsyncMock(return_value=False)
    monkeypatch.setattr(validate_mod, "_set_weights_with_confirmation", fallback)

    ok = await validate_mod.retry_set_weights(wallet="wallet", uids=[1, 2], weights=[0.5, 0.5])

    assert ok is True
    assert session.post_calls[0][0] == "http://signer/set_weights"
    assert fallback.await_count == 0


@pytest.mark.asyncio
async def test_retry_set_weights_falls_back_on_failure(monkeypatch):
    """When signer fails or times out, fallback should be invoked."""
    from babelbit.cli import validate as validate_mod

    fake_settings = SimpleNamespace(BABELBIT_NETUID=7, SIGNER_URL="http://signer")
    monkeypatch.setattr(validate_mod, "get_settings", lambda: fake_settings)
    monkeypatch.setenv("BB_SET_WEIGHTS_RETRIES", "3")
    monkeypatch.setenv("BB_SET_WEIGHTS_RETRY_DELAY", "0.1")

    # First call: signer returns non-200, triggering fallback
    session = MockSession(post_response=MockResponse(status=500, json_data={"success": False}))

    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda timeout: session,
        ClientConnectorError=FakeConnectorError,
        ClientTimeout=lambda **kwargs: FakeTimeout(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)

    fallback = AsyncMock(return_value=True)
    monkeypatch.setattr(validate_mod, "_set_weights_with_confirmation", fallback)

    ok = await validate_mod.retry_set_weights(wallet="wallet", uids=[3], weights=[1.0])
    assert ok is True
    fallback.assert_awaited_once_with(
        wallet="wallet",
        netuid=7,
        uids=[3],
        weights=[1.0],
        retries=3,
        delay_s=0.1,
    )

    # Second call: signer raises connector error, still uses fallback
    error_session = MockSession(post_exc=FakeConnectorError("no network"))
    fake_aiohttp_error = SimpleNamespace(
        ClientSession=lambda timeout: error_session,
        ClientConnectorError=FakeConnectorError,
        ClientTimeout=lambda **kwargs: FakeTimeout(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp_error)
    fallback = AsyncMock(return_value=True)
    monkeypatch.setattr(validate_mod, "_set_weights_with_confirmation", fallback)

    ok = await validate_mod.retry_set_weights(wallet="wallet", uids=[4], weights=[1.0])
    assert ok is True
    assert fallback.await_count == 1

    # Third call: signer raises timeout, still uses fallback
    timeout_session = MockSession(post_exc=asyncio.TimeoutError())
    fake_aiohttp_timeout = SimpleNamespace(
        ClientSession=lambda timeout: timeout_session,
        ClientConnectorError=FakeConnectorError,
        ClientTimeout=lambda **kwargs: FakeTimeout(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp_timeout)
    fallback = AsyncMock(return_value=True)
    monkeypatch.setattr(validate_mod, "_set_weights_with_confirmation", fallback)

    ok = await validate_mod.retry_set_weights(wallet="wallet", uids=[5], weights=[1.0])
    assert ok is True
    assert fallback.await_count == 1
