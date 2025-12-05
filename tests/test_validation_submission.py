from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from babelbit.utils.validation_submission import ValidationSubmissionClient


def _make_hotkey():
    hotkey = Mock()
    hotkey.ss58_address = "5abc"
    hotkey.sign = Mock(return_value=b"sigbytes")
    return hotkey


@pytest.mark.asyncio
async def test_submit_validation_file_success(tmp_path, monkeypatch):
    test_file = tmp_path / "dialogue.json"
    test_file.write_text('{"ok":true}', encoding="utf-8")

    # Patch settings and hotkey loader
    with patch("babelbit.utils.validation_submission.get_settings") as mock_settings, \
         patch("babelbit.utils.validation_submission.load_hotkey_keypair", return_value=_make_hotkey()):
        mock_settings.return_value = SimpleNamespace(
            BITTENSOR_WALLET_COLD="cold",
            BITTENSOR_WALLET_HOT="hot",
            BB_SUBMIT_API_URL="http://submit.test",
        )

        # Capture outgoing request
        post_mock = Mock()
        post_mock.return_value = SimpleNamespace(status_code=200, text="ok")
        monkeypatch.setattr("babelbit.utils.validation_submission.requests.post", post_mock)

        client = ValidationSubmissionClient()
        assert client.is_ready

        ok = await client.submit_validation_file(
            file_path=test_file,
            file_type="dialogue_log",
            kind="dialogue_logs",
            challenge_id="ch-123",
            miner_uid=1,
            miner_hotkey="hk",
            dialogue_uid="dlg-1",
            s3_path="s3://bucket/key",
            extra_data={"foo": "bar"},
        )

        assert ok is True
        post_mock.assert_called_once()
        called_kwargs = post_mock.call_args.kwargs
        assert called_kwargs["json"]["data"]["content"] == '{"ok":true}'
        assert called_kwargs["json"]["data"]["file_size"] == test_file.stat().st_size
        assert called_kwargs["json"]["signature"] == b"sigbytes".hex()
        assert called_kwargs["json"]["challenge_id"] == "ch-123"
        assert called_kwargs["json"]["data"]["foo"] == "bar"
        assert called_kwargs["json"]["kind"] == "dialogue_logs"


@pytest.mark.asyncio
async def test_submit_validation_file_rejected(monkeypatch):
    with patch("babelbit.utils.validation_submission.get_settings") as mock_settings, \
         patch("babelbit.utils.validation_submission.load_hotkey_keypair", return_value=_make_hotkey()):
        mock_settings.return_value = SimpleNamespace(
            BITTENSOR_WALLET_COLD="cold",
            BITTENSOR_WALLET_HOT="hot",
            BB_SUBMIT_API_URL="http://submit.test",
        )

        post_mock = Mock(return_value=SimpleNamespace(status_code=403, text="nope"))
        monkeypatch.setattr("babelbit.utils.validation_submission.requests.post", post_mock)

        client = ValidationSubmissionClient()
        ok = await client.submit_validation_file(
            file_path=Path("missing.json"),
            file_type="dialogue_log",
            kind=None,
            challenge_id="ch-123",
            miner_uid=None,
            miner_hotkey=None,
        )

        assert ok is False
        post_mock.assert_called_once()


@pytest.mark.asyncio
async def test_disabled_client_skips_submission(monkeypatch):
    with patch("babelbit.utils.validation_submission.get_settings") as mock_settings:
        mock_settings.return_value = SimpleNamespace(
            BITTENSOR_WALLET_COLD="cold",
            BITTENSOR_WALLET_HOT="hot",
            BB_SUBMIT_API_URL="http://submit.test",
        )

        client = ValidationSubmissionClient(enabled=False)
        assert client.is_ready is False

        post_spy = Mock()
        monkeypatch.setattr("babelbit.utils.validation_submission.requests.post", post_spy)

        ok = await client.submit_validation_file(
            file_path=Path("missing.json"),
            file_type="dialogue_log",
            kind=None,
            challenge_id="ch-123",
            miner_uid=None,
            miner_hotkey=None,
        )

        assert ok is False
        post_spy.assert_not_called()
