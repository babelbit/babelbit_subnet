import os, time, socket, asyncio, logging
from aiohttp import web
import bittensor as bt

from babelbit.utils.settings import get_settings

logger = logging.getLogger("sv-signer")

NETUID = int(os.getenv("BABELBIT_NETUID", "59"))

# Local subtensor management for the signer service
_signer_subtensor = None
_signer_subtensor_lock = asyncio.Lock()
_signer_subtensor_created_at = None


async def _get_local_subtensor():
    """Get or create a subtensor connection local to this signer service."""
    global _signer_subtensor, _signer_subtensor_created_at
    async with _signer_subtensor_lock:
        if _signer_subtensor is None:
            settings = get_settings()
            logger.info("[signer] 🔗 Initializing local subtensor: %s", settings.BITTENSOR_SUBTENSOR_ENDPOINT)
            _signer_subtensor = bt.async_subtensor(settings.BITTENSOR_SUBTENSOR_ENDPOINT)
            try:
                await _signer_subtensor.initialize()
                _signer_subtensor_created_at = time.monotonic()
                logger.info("[signer] ✅ Subtensor initialized: %s", settings.BITTENSOR_SUBTENSOR_ENDPOINT)
            except Exception as e:
                logger.error("[signer] ❌ Primary connection failed: %s", e)
                logger.info("[signer] 🔄 Trying fallback: %s", settings.BITTENSOR_SUBTENSOR_FALLBACK)
                _signer_subtensor = bt.async_subtensor(settings.BITTENSOR_SUBTENSOR_FALLBACK)
                await _signer_subtensor.initialize()
                _signer_subtensor_created_at = time.monotonic()
                logger.info("[signer] ✅ Fallback subtensor initialized: %s", settings.BITTENSOR_SUBTENSOR_FALLBACK)
        return _signer_subtensor


async def _reset_local_subtensor():
    """Close and reset the local subtensor connection."""
    global _signer_subtensor, _signer_subtensor_created_at
    async with _signer_subtensor_lock:
        if _signer_subtensor is not None:
            try:
                await _signer_subtensor.close()
                logger.info("[signer] 🔄 Subtensor connection closed")
            except Exception as e:
                logger.warning("[signer] ⚠️  Error closing subtensor: %s", e)
            _signer_subtensor = None
            _signer_subtensor_created_at = None
            logger.info("[signer] 🔄 Subtensor connection reset")


async def _should_reset_subtensor(operation_count: dict) -> bool:
    """
    Determine if subtensor should be reset based on operation count or age.
    Returns True if reset is needed.
    """
    global _signer_subtensor_created_at
    
    # Check operation count
    reset_interval = int(os.getenv("SIGNER_SUBTENSOR_RESET_INTERVAL", "50"))
    if operation_count["set_weights"] >= reset_interval:
        return True
    
    # Check age of connection (default: reset every 30 minutes)
    max_age_seconds = int(os.getenv("SIGNER_SUBTENSOR_MAX_AGE_SECONDS", "1800"))
    if _signer_subtensor_created_at is not None:
        age = time.monotonic() - _signer_subtensor_created_at
        if age >= max_age_seconds:
            logger.info(
                "[signer] Subtensor connection age: %.1f seconds (max: %d)",
                age,
                max_age_seconds
            )
            return True
    
    return False


async def _set_weights_with_confirmation(
    wallet: "bt.wallet",
    netuid: int,
    uids: list[int],
    weights: list[float],
    wait_for_inclusion: bool,
    retries: int = 10,
    delay_s: float = 2.0,
    log_prefix: str = "[signer]",
) -> bool:
    """Set weights with confirmation, using local subtensor instance."""
    for attempt in range(retries):
        st = None
        try:
            st = await _get_local_subtensor()
            ref_block = await st.get_current_block()

            # extrinsic
            success = await st.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=uids,
                weights=weights,
                wait_for_inclusion=wait_for_inclusion,
            )

            if not success:
                logger.warning("%s set_weights returned False, retry...", log_prefix)
                await asyncio.sleep(delay_s)
                continue

            logger.info(
                "%s extrinsic submitted; waiting next block … (ref=%d)",
                log_prefix,
                ref_block,
            )

            await st.wait_for_block()
            meta = await st.metagraph(netuid)
            try:
                i = meta.hotkeys.index(wallet.hotkey.ss58_address)
                lu = meta.last_update[i]
                if lu >= ref_block:
                    logger.info(
                        "%s confirmation OK (last_update=%d >= ref=%d)",
                        log_prefix,
                        lu,
                        ref_block,
                    )
                    return True
                logger.warning(
                    "%s not yet included (last_update=%d < ref=%d), retry…",
                    log_prefix,
                    lu,
                    ref_block,
                )
            except ValueError:
                logger.warning(
                    "%s wallet hotkey not found in metagraph; retry…", log_prefix
                )
        except Exception as e:
            logger.warning(
                "%s attempt %d error: %s: %s",
                log_prefix,
                attempt + 1,
                type(e).__name__,
                e,
            )
            # Reset stale connection on error
            await _reset_local_subtensor()
        await asyncio.sleep(delay_s)
    return False


async def run_signer() -> None:
    settings = get_settings()
    host = settings.SIGNER_HOST
    port = settings.SIGNER_PORT

    # Wallet Bittensor
    cold = settings.BITTENSOR_WALLET_COLD
    hot = settings.BITTENSOR_WALLET_HOT
    wallet = bt.wallet(name=cold, hotkey=hot)
    
    # Track operations for periodic cleanup (memory leak mitigation)
    operation_count = {"set_weights": 0, "total": 0}

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
        """"""
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
        """"""
        nonlocal operation_count
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
                except:
                    wgts = [0.0]
            if not isinstance(uids, list):
                uids = list(uids)
            if not isinstance(wgts, list):
                wgts = list(wgts)
            try:
                uids = [int(u) for u in uids]
            except:
                uids = []
            try:
                wgts = [float(w) for w in wgts]
            except:
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
                retries=int(os.getenv("SIGNER_RETRIES", "10")),
                delay_s=float(os.getenv("SIGNER_RETRY_DELAY", "2")),
                log_prefix="[signer]",
            )
            
            # Memory leak mitigation: periodically reset subtensor connection
            # to prevent accumulation of internal buffers/state in bittensor
            operation_count["set_weights"] += 1
            operation_count["total"] += 1
            
            if await _should_reset_subtensor(operation_count):
                logger.info(
                    "[signer] Resetting subtensor connection after %d set_weights operations (memory leak mitigation)",
                    operation_count["set_weights"]
                )
                await _reset_local_subtensor()
                operation_count["set_weights"] = 0
                # Force garbage collection to clean up accumulated objects
                import gc
                gc.collect()
            
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
        # Cleanup on shutdown
        logger.info("Shutting down signer...")
        await _reset_local_subtensor()  # Close subtensor connection
        await runner.cleanup()
        logger.info("Signer shutdown complete")