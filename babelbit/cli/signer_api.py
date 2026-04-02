import os
import time
import socket
import asyncio
import logging

from aiohttp import web
import bittensor as bt

from babelbit.utils.settings import get_settings
from babelbit.utils.subtensor_gateway_client import SubtensorGatewayClient

logger = logging.getLogger("sv-signer")

NETUID = int(os.getenv("BABELBIT_NETUID", "59"))


async def get_subtensor():
    """Deprecated runtime path retained for compatibility with older tests."""
    return None


async def reset_subtensor():
    """Deprecated runtime path retained for compatibility with older tests."""
    return None


async def _set_weights_with_confirmation(
    wallet: "bt.wallet",
    netuid: int,
    uids: list[int],
    weights: list[float],
    wait_for_inclusion: bool,
    retries: int = 20,
    delay_s: float = 2.0,
    log_prefix: str = "[signer]",
) -> bool:
    """Set weights via subtensor gateway and wait for confirmation."""
    gateway = SubtensorGatewayClient()
    try:
        ok = await gateway.set_weights_and_confirm(
            netuid=netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=wait_for_inclusion,
            retries=retries,
            delay_s=delay_s,
            wallet_hotkey=wallet.hotkey.ss58_address,
        )
        if ok:
            logger.info("%s confirmation OK via gateway", log_prefix)
        else:
            logger.warning("%s confirmation failed via gateway", log_prefix)
        return ok
    except Exception as e:
        logger.warning(
            "%s gateway set_weights failed: %s: %s",
            log_prefix,
            type(e).__name__,
            e,
        )
        return False


async def run_signer() -> None:
    settings = get_settings()
    host = settings.SIGNER_HOST
    port = settings.SIGNER_PORT

    # Wallet Bittensor
    cold = settings.BITTENSOR_WALLET_COLD
    hot = settings.BITTENSOR_WALLET_HOT
    wallet = bt.wallet(name=cold, hotkey=hot)

    @web.middleware
    async def access_log(request: web.Request, handler):
        t0 = time.monotonic()
        try:
            resp = await handler(request)
            return resp
        finally:
            dt = (time.monotonic() - t0) * 1000
            logger.info(
                "[signer] %s %s -> %s %.1fms",
                request.method,
                request.path,
                getattr(getattr(request, "response", None), "status", "?"),
                dt,
            )

    async def health(_req: web.Request):
        return web.json_response({"ok": True})

    async def sign_handler(req: web.Request):
        try:
            payload = await req.json()
            data = payload.get("payloads") or payload.get("data") or []
            if isinstance(data, str):
                data = [data]
            sigs = [(wallet.hotkey.sign(data=d.encode("utf-8"))).hex() for d in data]
            return web.json_response(
                {
                    "success": True,
                    "signatures": sigs,
                    "hotkey": wallet.hotkey.ss58_address,
                }
            )
        except Exception as e:
            logger.error("[sign] error: %s", e)
            return web.json_response({"success": False, "error": str(e)}, status=500)

    async def set_weights_handler(req: web.Request):
        try:
            payload = await req.json()
            netuid = int(payload.get("netuid", NETUID))
            uids = payload.get("uids") or []
            wgts = payload.get("weights") or []
            wfi = bool(payload.get("wait_for_inclusion", False))

            if isinstance(uids, int):
                uids = [uids]
            if isinstance(wgts, (int, float, str)):
                try:
                    wgts = [float(wgts)]
                except Exception:
                    wgts = [0.0]
            if not isinstance(uids, list):
                uids = list(uids)
            if not isinstance(wgts, list):
                wgts = list(wgts)
            try:
                uids = [int(u) for u in uids]
            except Exception:
                uids = []
            try:
                wgts = [float(w) for w in wgts]
            except Exception:
                wgts = []

            if len(uids) != len(wgts) or not uids:
                return web.json_response(
                    {"success": False, "error": "uids/weights mismatch or empty"},
                    status=400,
                )

            ok = await _set_weights_with_confirmation(
                wallet,
                netuid,
                uids,
                wgts,
                wfi,
                retries=int(os.getenv("SIGNER_RETRIES", "20")),
                delay_s=float(os.getenv("SIGNER_RETRY_DELAY", "2")),
                log_prefix="[signer]",
            )
            return web.json_response(
                (
                    {"success": True}
                    if ok
                    else {"success": False, "error": "confirmation failed"}
                ),
                status=200 if ok else 500,
            )
        except Exception as e:
            logger.error("[set_weights] error: %s", e)
            return web.json_response({"success": False, "error": str(e)}, status=500)

    app = web.Application(middlewares=[access_log])
    app.add_routes(
        [
            web.get("/healthz", health),
            web.post("/sign", sign_handler),
            web.post("/set_weights", set_weights_handler),
        ]
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()

    try:
        hn = socket.gethostname()
        ip = socket.gethostbyname(hn)
    except Exception:
        hn, ip = ("?", "?")
    logger.info(
        "Signer listening on http://%s:%s hostname=%s ip=%s", host, port, hn, ip
    )

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Signer received shutdown signal")
    finally:
        logger.info("Shutting down signer...")
        await runner.cleanup()
        logger.info("Signer shutdown complete")
