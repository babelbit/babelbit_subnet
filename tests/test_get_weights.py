import asyncio
import types
from datetime import datetime, timedelta, timezone
import pytest
from unittest.mock import AsyncMock, patch

# We'll import get_weights after patching helpers to avoid hitting real chain

@pytest.mark.asyncio
async def test_get_weights_single_winner(monkeypatch):
    # Mock metagraph with two hotkeys
    class FakeMeta:
        hotkeys = ["hk1", "hk2"]
    
    async def fake_get_subtensor():
        class ST:
            async def get_current_block(self):
                return 100
            async def metagraph(self, netuid):
                return FakeMeta()
        return ST()

    # Mock settings
    class FakeSettings:
        BABELBIT_NETUID = 1
        BITTENSOR_WALLET_COLD = "cold"
        BITTENSOR_WALLET_HOT = "hot"
        SIGNER_URL = "http://signer"
    
    from babelbit.utils import settings as settings_mod
    monkeypatch.setattr(settings_mod, "get_settings", lambda: FakeSettings())

    # Patch get_subtensor used inside validate
    from babelbit.cli import validate as validate_mod
    monkeypatch.setattr(validate_mod, "get_subtensor", fake_get_subtensor)

    # Patch db_pool.init (noop)
    monkeypatch.setattr(validate_mod.db_pool, "init", AsyncMock())

    # Patch _iter_scores_from_db to return scores (hk1 higher)
    # Rows already sorted DESC by time (latest first) for a single challenge 'chal-X'
    monkeypatch.setattr(validate_mod, "_iter_scores_from_db", AsyncMock(return_value=[
        ("hk1", 0.95, "chal-X"),
        ("hk2", 0.70, "chal-X"),
        ("hk1", 0.90, "chal-X"),  # older hk1 score ignored for 'latest per miner'
    ]))

    uids, weights, challenge_uid = await validate_mod.get_weights()
    assert weights == [1.0]
    # hk1 should map to uid 0 (order in FakeMeta.hotkeys)
    assert uids == [0]
    # Should return None when falling back to DB without current challenge
    assert challenge_uid is None

@pytest.mark.asyncio
async def test_get_weights_no_data_defaults_uid0(monkeypatch):
    """Test that with no data, get_weights returns default uid 248"""
    # Avoid hitting utterance engine during tests
    monkeypatch.setenv("BB_UTTERANCE_ENGINE_URL", "")
    # Ensure skip count is well above any configured max to force fallback
    max_skip_trigger = 1_000
    class FakeMeta:
        hotkeys = ["a", "b", "c"]
    async def fake_get_subtensor():
        class ST:
            async def metagraph(self, netuid):
                return FakeMeta()
        return ST()
    class FakeSettings:
        BABELBIT_NETUID = 4
        BITTENSOR_WALLET_COLD = "cold"
        BITTENSOR_WALLET_HOT = "hot"
        SIGNER_URL = "http://signer"
    from babelbit.utils import settings as settings_mod
    monkeypatch.setattr(settings_mod, "get_settings", lambda: FakeSettings())
    from babelbit.cli import validate as validate_mod
    monkeypatch.setattr(validate_mod, "get_subtensor", fake_get_subtensor)
    monkeypatch.setattr(validate_mod.db_pool, "init", AsyncMock())
    monkeypatch.setattr(validate_mod, "_iter_scores_from_db", AsyncMock(return_value=[]))
    # Avoid network call to utterance engine
    from babelbit.utils import predict_utterances as pred_mod
    monkeypatch.setattr(pred_mod, "get_current_challenge_uid", AsyncMock(return_value=None))
    
    # Pass consecutive_skipped_epochs >= MAX to trigger fallback
    uids, weights, challenge_uid = await validate_mod.get_weights(consecutive_skipped_epochs=max_skip_trigger)
    # Code defaults to uid 248 when no data available after max skips
    assert uids == [248]
    assert weights == [1.0]
    assert challenge_uid is None

@pytest.mark.asyncio
async def test_get_weights_multiple_challenges_uses_latest(monkeypatch):
    class FakeMeta:
        hotkeys = ["h1", "h2", "h3"]
    async def fake_get_subtensor():
        class ST:
            async def metagraph(self, netuid):
                return FakeMeta()
        return ST()
    class FakeSettings:
        BABELBIT_NETUID = 10
        BITTENSOR_WALLET_COLD = "cold"
        BITTENSOR_WALLET_HOT = "hot"
        SIGNER_URL = "http://signer"
    from babelbit.utils import settings as settings_mod
    monkeypatch.setattr(settings_mod, "get_settings", lambda: FakeSettings())
    from babelbit.cli import validate as validate_mod
    monkeypatch.setattr(validate_mod, "get_subtensor", fake_get_subtensor)
    monkeypatch.setattr(validate_mod.db_pool, "init", AsyncMock())
    # DESC order: latest challenge 'C2' first
    scores = [
        ("h1", 0.6, "C2"),
        ("h2", 0.9, "C2"),  # winner in latest
        ("h1", 0.55, "C2"),
        ("h3", 0.95, "C1"),  # older challenge should be ignored
        ("h2", 0.50, "C1"),
    ]
    monkeypatch.setattr(validate_mod, "_iter_scores_from_db", AsyncMock(return_value=scores))
    uids, weights, challenge_uid = await validate_mod.get_weights()
    assert uids == [1]  # h2 index
    assert weights == [1.0]
    assert challenge_uid is None  # Fallback to DB without current challenge


@pytest.mark.asyncio
async def test_get_weights_returns_challenge_uid(monkeypatch):
    """Test that get_weights returns the current challenge UID for tracking"""
    class FakeMeta:
        hotkeys = ["h1", "h2"]
    
    async def fake_get_subtensor():
        class ST:
            async def metagraph(self, netuid):
                return FakeMeta()
        return ST()
    
    class FakeSettings:
        BABELBIT_NETUID = 1
        BITTENSOR_WALLET_COLD = "cold"
        BITTENSOR_WALLET_HOT = "hot"
        SIGNER_URL = "http://signer"
    
    from babelbit.utils import settings as settings_mod
    monkeypatch.setattr(settings_mod, "get_settings", lambda: FakeSettings())
    from babelbit.cli import validate as validate_mod
    monkeypatch.setattr(validate_mod, "get_subtensor", fake_get_subtensor)
    monkeypatch.setattr(validate_mod.db_pool, "init", AsyncMock())
    
    # Mock challenge status checks
    monkeypatch.setattr(validate_mod, "is_challenge_processed", lambda uid: True)
    monkeypatch.setattr(validate_mod, "is_challenge_processed_db", AsyncMock(return_value=True))
    
    # Mock get_current_challenge_uid to return a specific challenge
    from babelbit.utils import predict_utterances as pred_mod
    monkeypatch.setattr(pred_mod, "get_current_challenge_uid", AsyncMock(return_value="challenge-abc-123"))
    
    # Mock _iter_scores_for_challenge to return scores
    from babelbit.utils import db_pool as db_pool_mod
    monkeypatch.setattr(db_pool_mod, "_iter_scores_for_challenge", AsyncMock(return_value=[
        ("h1", 0.85),
        ("h2", 0.90),
    ]))
    
    uids, weights, challenge_uid = await validate_mod.get_weights()
    
    # Should return the current challenge UID
    assert challenge_uid == "challenge-abc-123"
    assert uids == [1]  # h2 is winner
    assert weights == [1.0]


@pytest.mark.asyncio
async def test_get_weights_skips_unprocessed_challenge(monkeypatch):
    """Test that get_weights uses last challenge when current is unprocessed"""
    # Use dummy URL so mocked get_current_challenge_uid is invoked
    monkeypatch.setenv("BB_UTTERANCE_ENGINE_URL", "http://test")
    max_skip_trigger = 1_000
    class FakeMeta:
        hotkeys = ["h1", "h2"]
    
    async def fake_get_subtensor():
        class ST:
            async def metagraph(self, netuid):
                return FakeMeta()
        return ST()
    
    class FakeSettings:
        BABELBIT_NETUID = 1
        BITTENSOR_WALLET_COLD = "cold"
        BITTENSOR_WALLET_HOT = "hot"
        SIGNER_URL = "http://signer"
    
    from babelbit.utils import settings as settings_mod
    monkeypatch.setattr(settings_mod, "get_settings", lambda: FakeSettings())
    from babelbit.cli import validate as validate_mod
    monkeypatch.setattr(validate_mod, "get_subtensor", fake_get_subtensor)
    monkeypatch.setattr(validate_mod.db_pool, "init", AsyncMock())
    
    # Mock challenge status checks - current challenge NOT processed
    monkeypatch.setattr(validate_mod, "is_challenge_processed", lambda uid: False)
    monkeypatch.setattr(validate_mod, "is_challenge_processed_db", AsyncMock(return_value=False))
    
    # Mock get_current_challenge_uid
    from babelbit.utils import predict_utterances as pred_mod
    monkeypatch.setattr(pred_mod, "get_current_challenge_uid", AsyncMock(return_value="challenge-unprocessed"))
    # Avoid DB call when we stick with current challenge and max skips
    from babelbit.utils import db_pool as db_pool_mod
    monkeypatch.setattr(db_pool_mod, "_iter_scores_for_challenge", AsyncMock(return_value=[]))
    
    # Mock DB to return scores from last challenge
    monkeypatch.setattr(
        validate_mod, 
        "_iter_scores_from_db", 
        AsyncMock(return_value=[("h1", 0.8, "challenge-old"), ("h2", 0.6, "challenge-old")])
    )
    
    # Call with consecutive_skipped_epochs < MAX (should use last challenge)
    uids, weights, challenge_uid = await validate_mod.get_weights(consecutive_skipped_epochs=2)
    
    # Should return weights from last challenge (h1 wins with 0.8)
    assert uids == [0]  # h1 is at index 0 in hotkeys
    assert weights == [1.0]
    # Challenge UID should be None since we fell back to DB query
    assert challenge_uid is None
    
    # Test with no historical data either - should return empty
    monkeypatch.setattr(validate_mod, "_iter_scores_from_db", AsyncMock(return_value=[]))
    
    uids, weights, challenge_uid = await validate_mod.get_weights(consecutive_skipped_epochs=2)
    
    # Should return empty weights when no historical data
    assert uids == []
    assert weights == []
    assert challenge_uid is None
    
    # Call with consecutive_skipped_epochs >= MAX (should fallback to uid 248)
    uids, weights, challenge_uid = await validate_mod.get_weights(consecutive_skipped_epochs=max_skip_trigger)
    
    # Should fall back to uid 248
    assert uids == [248]
    assert weights == [1.0]
    # When falling back to uid 248 from current unprocessed challenge, we still return the challenge UID
    # This allows the main loop to track which challenge triggered the fallback
    assert challenge_uid == "challenge-unprocessed"


@pytest.mark.asyncio
async def test_get_weights_fallback_stale_challenge_defaults(monkeypatch):
    """If fallback challenge data is older than 12h, default to uid 248 immediately."""
    class FakeMeta:
        hotkeys = ["h1", "h2"]

    async def fake_get_subtensor():
        class ST:
            async def metagraph(self, netuid):
                return FakeMeta()
        return ST()

    class FakeSettings:
        BABELBIT_NETUID = 1
        BITTENSOR_WALLET_COLD = "cold"
        BITTENSOR_WALLET_HOT = "hot"
        SIGNER_URL = "http://signer"

    from babelbit.utils import settings as settings_mod
    monkeypatch.setattr(settings_mod, "get_settings", lambda: FakeSettings())

    from babelbit.cli import validate as validate_mod
    monkeypatch.setattr(validate_mod, "get_subtensor", fake_get_subtensor)
    monkeypatch.setattr(validate_mod.db_pool, "init", AsyncMock())

    # Avoid hitting the network for current challenge lookup
    from babelbit.utils import predict_utterances as pred_mod
    monkeypatch.setattr(pred_mod, "get_current_challenge_uid", AsyncMock(return_value=None))

    stale_ts = datetime.now(timezone.utc) - timedelta(hours=13)
    monkeypatch.setattr(
        validate_mod,
        "_iter_scores_from_db",
        AsyncMock(return_value=[
            ("h1", 0.7, "challenge-old", stale_ts),
            ("h2", 0.8, "challenge-old", stale_ts),
        ]),
    )

    uids, weights, challenge_uid = await validate_mod.get_weights(consecutive_skipped_epochs=2)

    assert uids == [248]
    assert weights == [1.0]
    assert challenge_uid is None


@pytest.mark.asyncio
async def test_get_weights_fallback_stale_challenge_defaults_after_max(monkeypatch):
    """If fallback challenge data is older than 12h and max skips exceeded, default to uid 248."""
    class FakeMeta:
        hotkeys = ["h1", "h2"]

    async def fake_get_subtensor():
        class ST:
            async def metagraph(self, netuid):
                return FakeMeta()
        return ST()

    class FakeSettings:
        BABELBIT_NETUID = 1
        BITTENSOR_WALLET_COLD = "cold"
        BITTENSOR_WALLET_HOT = "hot"
        SIGNER_URL = "http://signer"

    from babelbit.utils import settings as settings_mod
    monkeypatch.setattr(settings_mod, "get_settings", lambda: FakeSettings())

    from babelbit.cli import validate as validate_mod
    monkeypatch.setattr(validate_mod, "get_subtensor", fake_get_subtensor)
    monkeypatch.setattr(validate_mod.db_pool, "init", AsyncMock())

    from babelbit.utils import predict_utterances as pred_mod
    monkeypatch.setattr(pred_mod, "get_current_challenge_uid", AsyncMock(return_value=None))

    stale_ts = datetime.now(timezone.utc) - timedelta(hours=13)
    monkeypatch.setattr(
        validate_mod,
        "_iter_scores_from_db",
        AsyncMock(return_value=[
            ("h1", 0.7, "challenge-old", stale_ts),
            ("h2", 0.8, "challenge-old", stale_ts),
        ]),
    )

    uids, weights, challenge_uid = await validate_mod.get_weights(consecutive_skipped_epochs=6)

    assert uids == [248]
    assert weights == [1.0]
    assert challenge_uid is None
