from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from typing import Any

import bittensor as bt
from aiohttp import web

from babelbit.utils.settings import get_settings

logger = logging.getLogger("sv-subtensor-gateway")


class _GatewayState:
    def __init__(self):
        self.subtensor = None
        self.subtensor_lock = asyncio.Lock()
        self.meta_cache: dict[tuple[int, bool], Any] = {}
        self.meta_locks: dict[tuple[int, bool], asyncio.Lock] = {}
        self.created_at: float | None = None

    def _meta_lock(self, key: tuple[int, bool]) -> asyncio.Lock:
        lock = self.meta_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self.meta_locks[key] = lock
        return lock

    async def get_subtensor(self):
        async with self.subtensor_lock:
            if self.subtensor is None:
                settings = get_settings()
                primary = settings.BITTENSOR_SUBTENSOR_ENDPOINT
                fallback = settings.BITTENSOR_SUBTENSOR_FALLBACK
                logger.info("[gateway] connecting subtensor primary=%s", primary)
                self.subtensor = bt.async_subtensor(primary)
                try:
                    await self.subtensor.initialize()
                    self.created_at = time.monotonic()
                    logger.info("[gateway] subtensor connected primary=%s", primary)
                except Exception as e:
                    logger.warning("[gateway] primary failed: %s", e)
                    self.subtensor = bt.async_subtensor(fallback)
                    await self.subtensor.initialize()
                    self.created_at = time.monotonic()
                    logger.info("[gateway] subtensor connected fallback=%s", fallback)
            return self.subtensor

    async def reset_subtensor(self):
        async with self.subtensor_lock:
            if self.subtensor is not None:
                try:
                    await self.subtensor.close()
                except Exception:
                    logger.debug("[gateway] subtensor close failed", exc_info=True)
            self.subtensor = None
            self.created_at = None
            self.meta_cache.clear()

    async def metagraph(self, netuid: int, lite: bool):
        key = (int(netuid), bool(lite))
        lock = self._meta_lock(key)
        async with lock:
            st = await self.get_subtensor()
            meta = self.meta_cache.get(key)
            if meta is None:
                meta = await st.metagraph(netuid, lite=lite)
                self.meta_cache[key] = meta
            else:
                await meta.sync(subtensor=st, lite=lite)
            return meta


STATE = _GatewayState()


def _to_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    try:
        return [int(v) for v in values]
    except Exception:
        return []


def _serialize_axons(meta: Any) -> list[dict[str, Any]]:
    out = []
    axons = getattr(meta, "axons", []) or []
    for axon in axons:
        ip = getattr(axon, "ip", None)
        port = getattr(axon, "port", None)
        out.append({"ip": ip, "port": int(port) if port is not None else None})
    return out


def _meta_snapshot(meta: Any) -> dict[str, Any]:
    return {
        "block": int(getattr(meta, "block", 0) or 0),
        "hotkeys": list(getattr(meta, "hotkeys", []) or []),
        "last_update": _to_int_list(getattr(meta, "last_update", [])),
        "axons": _serialize_axons(meta),
    }


def _serialize_commitments(commits: dict[str, Any]) -> dict[str, list[list[Any]]]:
    out: dict[str, list[list[Any]]] = {}
    for hotkey, arr in (commits or {}).items():
        rows: list[list[Any]] = []
        for item in arr or []:
            try:
                block, data = item
            except Exception:
                continue
            if isinstance(data, bytes):
                try:
                    data = data.decode("utf-8")
                except Exception:
                    data = data.hex()
            rows.append([int(block), str(data)])
        out[str(hotkey)] = rows
    return out


async def run_subtensor_gateway() -> None:
    settings = get_settings()
    host = settings.SUBTENSOR_GATEWAY_HOST
    port = settings.SUBTENSOR_GATEWAY_PORT
    wallet = bt.wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )

    @web.middleware
    async def access_log(request: web.Request, handler):
        t0 = time.monotonic()
        try:
            resp = await handler(request)
            return resp
        finally:
            dt = (time.monotonic() - t0) * 1000
            logger.info(
                "[gateway] %s %s -> %s %.1fms",
                request.method,
                request.path,
                getattr(getattr(request, "response", None), "status", "?"),
                dt,
            )

    async def health(_req: web.Request):
        return web.json_response({"ok": True})

    async def block_current_handler(_req: web.Request):
        st = await STATE.get_subtensor()
        block = await st.get_current_block()
        return web.json_response({"block": int(block)})

    async def block_wait_handler(req: web.Request):
        payload = await req.json()
        timeout_s = payload.get("timeout_s")
        st = await STATE.get_subtensor()
        if timeout_s is None:
            await st.wait_for_block()
        else:
            await asyncio.wait_for(st.wait_for_block(), timeout=float(timeout_s))
        block = await st.get_current_block()
        return web.json_response({"block": int(block)})

    async def metagraph_snapshot_handler(req: web.Request):
        payload = await req.json()
        netuid = int(payload.get("netuid", os.getenv("BABELBIT_NETUID", "59")))
        lite = bool(payload.get("lite", False))
        meta = await STATE.metagraph(netuid=netuid, lite=lite)
        return web.json_response(_meta_snapshot(meta))

    async def registry_context_handler(req: web.Request):
        payload = await req.json()
        netuid = int(payload.get("netuid", os.getenv("BABELBIT_NETUID", "59")))
        lite = bool(payload.get("lite", False))
        st = await STATE.get_subtensor()
        meta = await STATE.metagraph(netuid=netuid, lite=lite)
        commits = await st.get_all_revealed_commitments(netuid)
        return web.json_response(
            {
                "metagraph": _meta_snapshot(meta),
                "commitments": _serialize_commitments(commits),
            }
        )

    async def set_weights_and_confirm_handler(req: web.Request):
        payload = await req.json()
        netuid = int(payload.get("netuid", os.getenv("BABELBIT_NETUID", "59")))
        uids = [int(u) for u in (payload.get("uids") or [])]
        weights = [float(w) for w in (payload.get("weights") or [])]
        wfi = bool(payload.get("wait_for_inclusion", False))
        retries = int(payload.get("retries", 20))
        delay_s = float(payload.get("delay_s", 2.0))
        wallet_hotkey = payload.get("wallet_hotkey") or wallet.hotkey.ss58_address

        if not uids or len(uids) != len(weights):
            return web.json_response(
                {"success": False, "error": "uids/weights mismatch or empty"}, status=400
            )

        for attempt in range(retries):
            try:
                st = await STATE.get_subtensor()
                ref = await st.get_current_block()
                success = await st.set_weights(
                    wallet=wallet,
                    netuid=netuid,
                    uids=uids,
                    weights=weights,
                    wait_for_inclusion=wfi,
                )
                if not success:
                    await asyncio.sleep(delay_s)
                    continue
                await st.wait_for_block()
                meta = await STATE.metagraph(netuid=netuid, lite=True)
                try:
                    idx = meta.hotkeys.index(wallet_hotkey)
                except ValueError:
                    await asyncio.sleep(delay_s)
                    continue
                lu = int(meta.last_update[idx])
                if lu >= int(ref):
                    return web.json_response(
                        {
                            "success": True,
                            "details": {
                                "last_update": lu,
                                "ref_block": int(ref),
                                "attempt": attempt + 1,
                            },
                        }
                    )
            except Exception as e:
                logger.warning(
                    "[gateway] set_weights attempt %d/%d failed: %s: %s",
                    attempt + 1,
                    retries,
                    type(e).__name__,
                    e,
                )
                await STATE.reset_subtensor()
            await asyncio.sleep(delay_s)

        return web.json_response(
            {"success": False, "error": "confirmation failed"}, status=500
        )

    app = web.Application(middlewares=[access_log])
    app.add_routes(
        [
            web.get("/healthz", health),
            web.post("/v1/block/current", block_current_handler),
            web.post("/v1/block/wait", block_wait_handler),
            web.post("/v1/metagraph/snapshot", metagraph_snapshot_handler),
            web.post("/v1/registry/context", registry_context_handler),
            web.post("/v1/weights/set_and_confirm", set_weights_and_confirm_handler),
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
        "Subtensor gateway listening on http://%s:%s hostname=%s ip=%s",
        host,
        port,
        hn,
        ip,
    )

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Gateway received shutdown signal")
    finally:
        await STATE.reset_subtensor()
        await runner.cleanup()
