from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import aiohttp
import requests
from huggingface_hub import HfApi

from babelbit.utils.bittensor_helpers import get_subtensor
from babelbit.utils.subtensor_gateway_client import SubtensorGatewayClient


@dataclass
class Miner:
    uid: int
    hotkey: str
    block: int
    model: Optional[str] = None
    revision: Optional[str] = None
    slug: Optional[str] = None
    chute_id: Optional[str] = None
    axon_ip: Optional[str] = None
    axon_port: Optional[int] = None


_HF_MODEL_GATING_CACHE: Dict[str, Tuple[bool, float]] = {}
_HF_GATING_TTL = 300  # seconds


def _is_valid_axon(ip: Optional[str], port: Optional[int]) -> bool:
    if not ip or not port:
        return False
    ip_str = str(ip).strip()
    if not ip_str or ip_str in {"0.0.0.0", "0", "None"}:
        return False
    try:
        return int(port) > 0
    except Exception:
        return False


def _hf_is_gated(model_id: str) -> Optional[bool]:
    try:
        resp = requests.get(f"https://huggingface.co/api/models/{model_id}", timeout=5)
        if resp.status_code == 200:
            return bool(resp.json().get("gated", False))
    except Exception:
        pass
    return None


def _hf_revision_accessible(model_id: str, revision: Optional[str]) -> bool:
    if not revision:
        return True
    try:
        token = os.getenv("HUGGINGFACE_API_KEY")
        api = HfApi(token=token) if token else HfApi()
        api.repo_info(repo_id=model_id, repo_type="model", revision=revision)
        return True
    except Exception:
        return False


def _hf_gated_or_inaccessible(model_id: Optional[str], revision: Optional[str]) -> Optional[bool]:
    if not model_id:
        return True
    now = time.time()
    cached = _HF_MODEL_GATING_CACHE.get(model_id)
    gated = None
    if cached and (now - cached[1]) < _HF_GATING_TTL:
        gated = cached[0]
    else:
        gated = _hf_is_gated(model_id)
        _HF_MODEL_GATING_CACHE[model_id] = (bool(gated) if gated is not None else False, now)
    if gated is True:
        return True
    if not _hf_revision_accessible(model_id, revision):
        return True
    return False


async def _chutes_get_json(url: str, headers: Dict[str, str]) -> Optional[dict]:
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                return None
            try:
                return await response.json()
            except Exception:
                return None


async def fetch_chute_info(chute_id: str) -> Optional[dict]:
    token = os.getenv("CHUTES_API_KEY", "")
    if not token or not chute_id:
        return None
    return await _chutes_get_json(
        f"https://api.chutes.ai/chutes/{chute_id}",
        headers={"Authorization": token},
    )


def _extract_axon(meta: Any, uid: int) -> tuple[Optional[str], Optional[int]]:
    axons = getattr(meta, "axons", []) or []
    if uid >= len(axons):
        return None, None

    axon = axons[uid]
    if isinstance(axon, dict):
        ip = axon.get("ip")
        port = axon.get("port")
    else:
        ip = getattr(axon, "ip", None)
        port = getattr(axon, "port", None)

    ip_str = str(ip).strip() if ip is not None else None
    port_int: Optional[int]
    try:
        port_int = int(port) if port is not None else None
    except Exception:
        port_int = None
    return ip_str, port_int


def _snapshot_to_meta(snapshot: dict[str, Any]) -> Any:
    return SimpleNamespace(
        hotkeys=snapshot.get("hotkeys", []),
        last_update=snapshot.get("last_update", []),
        axons=snapshot.get("axons", []),
        block=snapshot.get("block"),
    )


async def _load_registry_context(netuid: int, subtensor=None, logger: Optional[logging.Logger] = None) -> dict[str, Any]:
    if logger is None:
        logger = logging.getLogger(__name__)

    if subtensor is None:
        try:
            gateway = SubtensorGatewayClient()
            return await gateway.registry_context(netuid=netuid, lite=False)
        except Exception as exc:
            logger.warning(
                "Gateway registry context failed; falling back to subtensor: %s: %s",
                type(exc).__name__,
                exc,
            )
            st = await get_subtensor()
            meta = await st.metagraph(netuid)
            commits = await st.get_all_revealed_commitments(netuid)
            return {"metagraph": {"hotkeys": getattr(meta, "hotkeys", []), "axons": getattr(meta, "axons", []), "block": getattr(meta, "block", 0)}, "commitments": commits}

    if hasattr(subtensor, "registry_context"):
        return await subtensor.registry_context(netuid=netuid, lite=False)

    if hasattr(subtensor, "metagraph_object"):
        meta = await subtensor.metagraph_object(netuid=netuid, lite=False)
        commits = {}
        if hasattr(subtensor, "get_all_revealed_commitments"):
            try:
                commits = await subtensor.get_all_revealed_commitments(netuid)
            except Exception:
                commits = {}
        return {
            "metagraph": {
                "hotkeys": getattr(meta, "hotkeys", []),
                "last_update": getattr(meta, "last_update", []),
                "axons": getattr(meta, "axons", []),
                "block": getattr(meta, "block", 0),
            },
            "commitments": commits,
        }

    if hasattr(subtensor, "metagraph"):
        meta = await subtensor.metagraph(netuid)
        commits = {}
        if hasattr(subtensor, "get_all_revealed_commitments"):
            try:
                commits = await subtensor.get_all_revealed_commitments(netuid)
            except Exception:
                commits = {}
        return {
            "metagraph": {
                "hotkeys": getattr(meta, "hotkeys", []),
                "last_update": getattr(meta, "last_update", []),
                "axons": getattr(meta, "axons", []),
                "block": getattr(meta, "block", 0),
            },
            "commitments": commits,
        }

    raise RuntimeError(f"Unsupported subtensor client type: {type(subtensor).__name__}")


def _normalize_hotkey_commitments(commitments: Any) -> dict[str, list]:
    if not isinstance(commitments, dict):
        return {}
    normalized: dict[str, list] = {}
    for hotkey, rows in commitments.items():
        if isinstance(rows, list):
            normalized[str(hotkey)] = rows
    return normalized


def _commitment_latest_row(rows: list) -> Optional[tuple[int, str]]:
    if not rows:
        return None
    latest = rows[-1]
    if not isinstance(latest, (list, tuple)) or len(latest) < 2:
        return None
    raw_block, raw_data = latest[0], latest[1]
    try:
        block = int(raw_block or 0)
    except Exception:
        block = 0
    if isinstance(raw_data, bytes):
        try:
            data = raw_data.decode("utf-8")
        except Exception:
            data = raw_data.hex()
    else:
        data = str(raw_data or "")
    return block, data


async def get_miners_from_registry(netuid: int, subtensor=None) -> Dict[int, Miner]:
    """
    Resolve miner candidates from commitments + self-hosted axons.
    Main-mode runner uses this registry source.
    """
    logger = logging.getLogger(__name__)
    ctx = await _load_registry_context(netuid=netuid, subtensor=subtensor, logger=logger)
    snapshot = ctx.get("metagraph", {}) if isinstance(ctx, dict) else {}
    meta = _snapshot_to_meta(snapshot if isinstance(snapshot, dict) else {})
    commits = _normalize_hotkey_commitments(ctx.get("commitments", {}) if isinstance(ctx, dict) else {})

    hotkeys = list(getattr(meta, "hotkeys", []) or [])
    logger.info("Checking %d hotkeys for commitments", len(hotkeys))
    logger.info("Available commitments for hotkeys: %s", list(commits.keys()))

    candidates: Dict[int, Miner] = {}
    seen_slugs: Dict[str, tuple[int, int]] = {}

    for uid, hotkey in enumerate(hotkeys):
        commitment_rows = commits.get(hotkey) or []
        axon_ip, axon_port = _extract_axon(meta, uid)

        if not commitment_rows:
            if _is_valid_axon(axon_ip, axon_port):
                candidates[uid] = Miner(
                    uid=uid,
                    hotkey=hotkey,
                    block=0,
                    model=None,
                    revision=None,
                    slug=None,
                    chute_id=None,
                    axon_ip=axon_ip,
                    axon_port=axon_port,
                )
            continue

        latest = _commitment_latest_row(commitment_rows)
        if latest is None:
            continue
        raw_block, raw_data = latest
        try:
            commitment_obj = json.loads(raw_data)
        except Exception:
            continue

        if not isinstance(commitment_obj, dict):
            continue

        model = commitment_obj.get("model")
        revision = commitment_obj.get("revision")
        slug = commitment_obj.get("slug")
        chute_id = commitment_obj.get("chute_id")

        if not slug and not _is_valid_axon(axon_ip, axon_port):
            continue

        block = int(raw_block or 0) if uid != 0 else 0

        if isinstance(slug, str) and slug:
            prev = seen_slugs.get(slug)
            if prev is None or block < prev[1]:
                if prev is not None:
                    candidates.pop(prev[0], None)
                seen_slugs[slug] = (uid, block)
                candidates[uid] = Miner(
                    uid=uid,
                    hotkey=hotkey,
                    block=block,
                    model=model,
                    revision=revision,
                    slug=slug,
                    chute_id=chute_id,
                    axon_ip=axon_ip,
                    axon_port=axon_port,
                )
            continue

        candidates[uid] = Miner(
            uid=uid,
            hotkey=hotkey,
            block=block,
            model=model,
            revision=revision,
            slug=slug,
            chute_id=chute_id,
            axon_ip=axon_ip,
            axon_port=axon_port,
        )

    if not candidates:
        logger.info("Found 0 eligible miners from commitments/axons")
        return {}

    filtered: Dict[int, Miner] = {}
    for uid, miner in candidates.items():
        if not miner.model:
            if _is_valid_axon(miner.axon_ip, miner.axon_port):
                filtered[uid] = miner
            continue

        gated = _hf_gated_or_inaccessible(miner.model, miner.revision)
        if gated is True:
            continue

        keep = True
        if miner.chute_id:
            info = await fetch_chute_info(miner.chute_id)
            if not info:
                keep = False
            else:
                chute_slug = (info.get("slug") or "").strip() if isinstance(info, dict) else ""
                if chute_slug and chute_slug != (miner.slug or ""):
                    keep = False
                chute_revision = info.get("revision") if isinstance(info, dict) else None
                if chute_revision and miner.revision and str(chute_revision) != str(miner.revision):
                    keep = False
        if keep:
            filtered[uid] = miner

    if not filtered:
        logger.info("Found 0 eligible miners after filters")
        return {}

    best_by_model: Dict[str, Tuple[int, int]] = {}
    for uid, miner in filtered.items():
        if not miner.model:
            continue
        blk = miner.block if isinstance(miner.block, int) else int(miner.block or (2**63 - 1))
        prev = best_by_model.get(miner.model)
        if prev is None or blk < prev[0]:
            best_by_model[miner.model] = (blk, uid)

    keep_uids = {uid for _, uid in best_by_model.values()}
    result: Dict[int, Miner] = {uid: filtered[uid] for uid in keep_uids if uid in filtered}
    for uid, miner in filtered.items():
        if not miner.model:
            result[uid] = miner

    logger.info("Found %d eligible miners from registry", len(result))
    return result
