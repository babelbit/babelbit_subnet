from json import loads

from aiohttp import ClientTimeout, ClientSession
from substrateinterface import Keypair

from babelbit.utils.settings import get_settings


async def _sign_batch(payloads: list[str]) -> tuple[str, list[str]]:
    """
    Retourne (hotkey_hex, signatures_hex[]).
    1) essaie SIGNER_URL /sign
    2) fallback: SIGN_SEED local
    """
    settings = get_settings()
    if settings.SIGNER_URL:
        try:
            timeout = ClientTimeout(connect=2, total=30)
            async with ClientSession(timeout=timeout) as sess:
                r = await sess.post(
                    f"{settings.SIGNER_URL}/sign", json={"payloads": payloads}
                )
                txt = await r.text()
                if r.status == 200:
                    data = loads(txt)
                    sigs = data.get("signatures") or []
                    hk = data.get("hotkey") or ""
                    if len(sigs) == len(payloads) and hk:
                        return hk, sigs
        except Exception as e:
            raise Exception(f"signer unavailable, fallback to local: {e}")
    raise ValueError("No Signer URL set")


def sign_message(keypair: Keypair, message: str | None) -> str | None:
    """Sign an arbitrary message with the provided keypair, returning a raw hex string.

    Historically this function returned the signature prefixed with '0x'. The
    utterance engine authentication path (and typical on-chain style
    verification using bytes.fromhex) expects a pure hexadecimal string
    without any prefix. The '0x' prefix caused downstream failures like:

        ValueError: non-hexadecimal number found in fromhex() arg at position 1

    because 'x' is not a valid hex digit. We now return the bare lowercase hex
    so consumers can directly feed it to bytes.fromhex(). If callers stored or
    compared older '0x' formatted signatures they should strip the prefix
    before comparison.

    Args:
        keypair: Substrate Keypair used to sign.
        message: String message to sign. If None, returns None.

    Returns:
        Hex string of the signature (no 0x prefix) or None if message is None.
    """
    if message is None:
        return None
    # Bittensor / substrate Keypair.sign accepts str or bytes; ensure bytes for clarity.
    if isinstance(message, str):
        msg_bytes = message.encode("utf-8")
    else:  # pragma: no cover - defensive, current callers pass str
        msg_bytes = message
    sig_hex = keypair.sign(msg_bytes).hex()
    # Ensure no accidental 0x prefix (shouldn't be present from .hex()) but be safe.
    if sig_hex.startswith("0x"):
        sig_hex = sig_hex[2:]
    return sig_hex.lower()
