from __future__ import annotations
import os, json, time, asyncio, requests, logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp
from huggingface_hub import HfApi

from babelbit.utils.bittensor_helpers import get_subtensor
from babelbit.utils.settings import get_settings


@dataclass
class Miner:
    uid: int
    hotkey: str
    model: Optional[str]
    revision: Optional[str]
    slug: Optional[str]
    chute_id: Optional[str]
    block: int
    axon_ip: Optional[str] = None
    axon_port: Optional[int] = None


# ------------------------- HF gating & revision checks ------------------------- #
_HF_MODEL_GATING_CACHE: Dict[str, Tuple[bool, float]] = {}
_HF_GATING_TTL = 300  # seconds


def _hf_is_gated(model_id: str) -> Optional[bool]:
    try:
        r = requests.get(f"https://huggingface.co/api/models/{model_id}", timeout=5)
        if r.status_code == 200:
            return bool(r.json().get("gated", False))
    except Exception:
        pass
    return None


def _hf_revision_accessible(model_id: str, revision: Optional[str]) -> bool:
    if not revision:
        return True
    try:
        tok = os.getenv("HUGGINGFACE_API_KEY")
        api = HfApi(token=tok) if tok else HfApi()
        api.repo_info(repo_id=model_id, repo_type="model", revision=revision)
        return True
    except Exception:
        return False


def _hf_gated_or_inaccessible(
    model_id: Optional[str], revision: Optional[str]
) -> Optional[bool]:
    if not model_id:
        return True  # no model id -> treat as not eligible
    now = time.time()
    cached = _HF_MODEL_GATING_CACHE.get(model_id)
    gated = None
    if cached and (now - cached[1]) < _HF_GATING_TTL:
        gated = cached[0]
    else:
        gated = _hf_is_gated(model_id)
        # store something even if None to avoid hammering
        _HF_MODEL_GATING_CACHE[model_id] = (
            bool(gated) if gated is not None else False,
            now,
        )
    if gated is True:
        return True
    if not _hf_revision_accessible(model_id, revision):
        return True
    return False  # either False or None (unknown) -> allow


# ------------------------------ Chutes helpers -------------------------------- #
async def _chutes_get_json(url: str, headers: Dict[str, str]) -> Optional[dict]:
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.get(url, headers=headers) as r:
            if r.status != 200:
                return None
            try:
                return await r.json()
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


# ---------------------------- Miner registry main ----------------------------- #
async def get_miners_from_registry(netuid: int, subtensor=None) -> Dict[int, Miner]:
    """
    Reads on-chain commitments, verifies HF gating/revision and Chutes slug,
    and returns at most one miner per model (earliest block wins).
    
    Args:
        netuid: The subnet ID
        subtensor: Optional existing subtensor connection to reuse
    """
    settings = get_settings()
    st = subtensor if subtensor is not None else await get_subtensor()
    meta = await st.metagraph(netuid)
    commits = await st.get_all_revealed_commitments(netuid)

    # 1) Extract candidates (uid -> Miner)
    candidates: Dict[int, Miner] = {}
    logger = logging.getLogger(__name__)
    logger.info(f"Checking {len(meta.hotkeys)} hotkeys for commitments")
    logger.info(f"Available commitments for hotkeys: {list(commits.keys())}")

    seen_slugs: Dict[str, tuple[int, int]] = {}
    
    for uid, hk in enumerate(meta.hotkeys):
        arr = commits.get(hk)
        logger.info(f"UID {uid} hotkey {hk}: commitment = {arr}")
        
        # Extract axon information from metagraph (for all miners, regardless of commitment)
        axon_ip = None
        axon_port = None
        try:
            if hasattr(meta, 'axons') and uid < len(meta.axons):
                axon = meta.axons[uid]
                if hasattr(axon, 'ip') and hasattr(axon, 'port'):
                    axon_ip = axon.ip
                    axon_port = axon.port
                    if axon_ip and axon_port:
                        logger.info(f"UID {uid} has axon at {axon_ip}:{axon_port}")
        except Exception as e:
            logger.warning(f"UID {uid} failed to extract axon info: {e}")
        
        # If no commitment, check if miner has axon endpoint
        if not arr:
            if axon_ip and axon_port:
                logger.info(f"UID {uid} has no commitment but has axon endpoint, including as self-hosted miner")
                candidates[uid] = Miner(
                    uid=uid,
                    hotkey=hk,
                    model=None,
                    revision=None,
                    slug=None,
                    chute_id=None,
                    block=0,
                    axon_ip=axon_ip,
                    axon_port=axon_port,
                )
            continue
            
        block, data = arr[-1]
        try:
            obj = json.loads(data)
            logger.info(f"UID {uid} parsed commitment: {obj}")
        except Exception as e:
            logger.warning(f"UID {uid} failed to parse commitment JSON: {e}")
            continue

        model = obj.get("model")
        revision = obj.get("revision")
        slug = obj.get("slug")
        chute_id = obj.get("chute_id")

        # If no slug and no axon, skip miner
        if not slug and not (axon_ip and axon_port):
            logger.warning(f"UID {uid} has neither slug nor axon endpoint, skipping")
            continue
        
        block = int(block or 0) if uid != 0 else 0  # mirror special-case for uid 0

        # Slug deduplication: keep earliest block per slug (only if slug exists)
        if slug:
            if slug not in seen_slugs:
                seen_slugs[slug] = (uid, block)
                # Add this miner since it's the first with this slug
                candidates[uid] = Miner(
                    uid=uid,
                    hotkey=hk,
                    model=model,
                    revision=revision,
                    slug=slug,
                    chute_id=chute_id,
                    block=block,
                    axon_ip=axon_ip,
                    axon_port=axon_port,
                )
            elif seen_slugs[slug][1] > block:
                # This miner has an earlier block, replace the previous one
                candidates.pop(seen_slugs[slug][0], None)
                logger.warning(
                    f"Slug deduplication: eliminating UID {seen_slugs[slug][0]} with duplicate slug '{slug}'"
                )
                seen_slugs[slug] = (uid, block)
                candidates[uid] = Miner(
                    uid=uid,
                    hotkey=hk,
                    model=model,
                    revision=revision,
                    slug=slug,
                    chute_id=chute_id,
                    block=block,
                    axon_ip=axon_ip,
                    axon_port=axon_port,
                )
        else:
            # No slug, but has axon - add directly
            candidates[uid] = Miner(
                uid=uid,
                hotkey=hk,
                model=model,
                revision=revision,
                slug=slug,
                chute_id=chute_id,
                block=block,
                axon_ip=axon_ip,
                axon_port=axon_port,
            )

    if not candidates:
        return {}

    # 2) Filter by HF gating/inaccessible + Chutes slug/revision checks
    # Note: allow self-hosted miners (no on-chain model/slug) if they expose an axon endpoint
    logger.info(f"Starting filtering of {len(candidates)} candidates: {list(candidates.keys())}")
    filtered: Dict[int, Miner] = {}
    for uid, m in candidates.items():
        # Self-hosted miners won't have a model id; permit them if they expose an axon endpoint
        if not m.model:
            if m.axon_ip and m.axon_port:
                logger.info(f"UID {uid} is self-hosted with axon {m.axon_ip}:{m.axon_port}, adding to filtered")
                filtered[uid] = m
            else:
                logger.info(f"UID {uid} is self-hosted but missing axon, skipping")
            # skip HF gating/chutes checks for self-hosted entries
            continue

        gated = _hf_gated_or_inaccessible(m.model, m.revision)
        if gated is True:
            continue

        ok = True
        if m.chute_id:
            info = await fetch_chute_info(m.chute_id)
            if not info:
                ok = False
            else:
                # cross-check slug (light-normalize)
                slug_chutes = (info.get("slug") or "").strip()
                if slug_chutes and slug_chutes != (m.slug or ""):
                    ok = False
                # optional: if chutes reports a revision, ensure it matches miner's revision
                ch_rev = info.get("revision")
                if ch_rev and m.revision and str(ch_rev) != str(m.revision):
                    ok = False
        if ok:
            filtered[uid] = m

    if not filtered:
        return {}

    # 3) De-duplicate by model: keep earliest block per model (stable)
    logger.info(f"Starting model deduplication of {len(filtered)} filtered miners: {list(filtered.keys())}")
    best_by_model: Dict[str, Tuple[int, int]] = {}
    for uid, m in filtered.items():
        if not m.model:
            # skip self-hosted entries for model-based de-duplication
            logger.info(f"UID {uid} is self-hosted (no model), skipping deduplication")
            continue
        blk = (
            m.block
            if isinstance(m.block, int)
            else (int(m.block) if m.block is not None else (2**63 - 1))
        )
        prev = best_by_model.get(m.model)
        if prev is None or blk < prev[0]:
            best_by_model[m.model] = (blk, uid)

    keep_uids = {uid for _, uid in best_by_model.values()}
    
    logger.info(f"Model-backed miners to keep: {keep_uids}")

    # Assemble final result: include de-duplicated model-backed miners
    result: Dict[int, Miner] = {uid: filtered[uid] for uid in keep_uids if uid in filtered}
    logger.info(f"After adding model-backed miners, result has {len(result)} miners: {list(result.keys())}")
    
    # Also include any self-hosted miners (model is None) that passed earlier checks
    for uid, m in filtered.items():
        if not m.model:
            logger.info(f"Adding self-hosted UID {uid} to final result")
            result[uid] = m
    
    logger.info(f"Final result has {len(result)} miners: {list(result.keys())}")
    return result
