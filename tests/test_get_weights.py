import asyncio
import types
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

    uids, weights = await validate_mod.get_weights()
    assert weights == [1.0]
    # hk1 should map to uid 0 (order in FakeMeta.hotkeys)
    assert uids == [0]

@pytest.mark.asyncio
async def test_get_weights_no_data_defaults_uid0(monkeypatch):
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
    uids, weights = await validate_mod.get_weights()
    assert uids == [0]
    assert weights == [1.0]

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
    uids, weights = await validate_mod.get_weights()
    assert uids == [1]  # h2 index
    assert weights == [1.0]
