import asyncio
import json
import os
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from babelbit.utils.settings import get_settings
from babelbit.utils.bittensor_helpers import load_hotkey_keypair

logger = getLogger(__name__)


class ValidationSubmissionClient:
    """Submit validation artifacts to the API with a signed payload."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        enabled: Optional[bool] = None,
        timeout: int = 10,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.BB_SUBMIT_API_URL).rstrip("/")
        self.submit_url = f"{self.base_url}/v1/submit"
        self.timeout = timeout
        if enabled is None:
            enabled = os.getenv("BB_ENABLE_VALIDATION_SUBMISSIONS", "1").lower() in {"1", "true", "yes"}
        self.enabled = bool(enabled)

        self._hotkey = None
        if self.enabled:
            try:
                self._hotkey = load_hotkey_keypair(settings.BITTENSOR_WALLET_COLD, settings.BITTENSOR_WALLET_HOT)
            except Exception as e:
                logger.warning("Validation submissions disabled; could not load hotkey: %s", e)
                self.enabled = False

    @property
    def is_ready(self) -> bool:
        return self.enabled and self._hotkey is not None

    def _build_signed_payload(self, challenge_id: str, miner_hotkey: str, miner_uid: int, data: Dict[str, Any]) -> Dict[str, Any]:
        if self._hotkey is None:
            raise RuntimeError("Hotkey not loaded; cannot sign submission payload")

        payload = {
            "vali_hotkey": self._hotkey.ss58_address,
            "miner_hotkey": miner_hotkey,
            "miner_uid": miner_uid,
            "timestamp": int(time.time()),
            "challenge_id": challenge_id,
            "data": data,
        }
        canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        payload["signature"] = self._hotkey.sign(canonical.encode()).hex()
        return payload

    async def submit_validation_file(
        self,
        *,
        file_path: Path,
        file_type: str,
        challenge_id: str,
        miner_uid: Optional[int],
        miner_hotkey: Optional[str],
        dialogue_uid: Optional[str] = None,
        s3_path: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        kind: Optional[str] = None,
    ) -> bool:
        """Read the file, sign the payload, and POST to the validation API."""
        if not self.is_ready:
            return False

        data: Dict[str, Any] = {
            "file_type": file_type,
            "file_name": file_path.name,
            "challenge_uid": challenge_id,
            "miner_uid": miner_uid,
            "miner_hotkey": miner_hotkey,
            "dialogue_uid": dialogue_uid,
            "s3_path": s3_path,
            "file_size": None,
        }

        try:
            if file_path.exists():
                data["file_size"] = file_path.stat().st_size
                data["content"] = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.debug("Unable to read file %s for submission: %s", file_path, e)

        if extra_data:
            data.update(extra_data)

        payload = self._build_signed_payload(challenge_id or "", miner_hotkey or "", miner_uid or 0, data)
        outbound_kind = kind or None
        if outbound_kind == "dialogue_run":
            outbound_kind = "dialogue_logs"
        elif outbound_kind is None and file_type == "dialogue_run":
            outbound_kind = "dialogue_logs"

        if outbound_kind:
            allowed_kinds = {"challenge_scores", "dialogue_scores", "dialogue_logs"}
            if outbound_kind not in allowed_kinds:
                logger.warning("Validation submit rejected client-side due to invalid kind: %s", outbound_kind)
                return False
            payload["kind"] = outbound_kind

        try:
            response = await asyncio.to_thread(
                requests.post,
                self.submit_url,
                json=payload,
                timeout=self.timeout,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Validation submit failed for %s (%s): %s", file_path.name, file_type, e)
            return False

        if response.status_code != 200:
            logger.warning(
                "Validation submit rejected for %s (%s): %s %s",
                file_path.name,
                file_type,
                response.status_code,
                response.text,
            )
            return False

        logger.info(
            "Validation submit accepted for %s (%s) via %s",
            file_path.name,
            file_type,
            self.submit_url,
        )
        return True
