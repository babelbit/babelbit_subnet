import atexit
import gc
import copy
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import lru_cache
from typing import List, Optional, Dict
from logging import INFO, getLogger
import os
import sys
import subprocess
from pathlib import Path
import json
import random
import asyncio
import time
import traceback
from urllib.parse import urlparse, urlunparse
from aiohttp import ClientTimeout

from babelbit.utils.s3_manager import S3Manager
from babelbit.utils.settings import get_settings

from babelbit.utils.predict_utterances import (
    get_current_challenge_uid, 
    predict_with_utterance_engine_multi_miner,
)
from babelbit.utils.utterance_auth import init_utterance_auth, authenticate_utterance_engine
from babelbit.utils.async_clients import close_http_clients, get_async_client

from babelbit.utils.miner_registry import get_miners_from_registry, Miner
from babelbit.utils.managed_container_registry import ManagedRoute, resolve_round2_routes
from babelbit.utils.subtensor_gateway_client import (
    SubtensorGatewayClient,
    close_gateway_clients,
)
from babelbit.schemas.prediction import BBPredictedUtterance
from babelbit.utils.file_handling import (
    get_processed_miners_for_challenge,
    save_dialogue_score_file,
    save_challenge_summary_file,
)
from babelbit.utils.challenge_status import mark_challenge_processed
from babelbit.utils.validation_submission import ValidationSubmissionClient


from babelbit.scoring.score_dialogue import prime_score_jsonl_events, score_jsonl
from babelbit.scoring.score_dialogue import prewarm_score_dialogue_embedder

logger = getLogger(__name__)

s3_manager: Optional[S3Manager] = None
settings = get_settings()
_DEFER_HTTP_CLIENT_CLOSE = False
_BACKGROUND_TASKS: set[asyncio.Task] = set()
_SCORING_PROCESS_POOL: Optional[ProcessPoolExecutor] = None
_SCORING_PROCESS_POOL_CONFIG: Optional[tuple[int, int]] = None


async def get_subtensor():
    return SubtensorGatewayClient()


async def reset_subtensor():
    await close_gateway_clients()


def _close_http_clients_if_allowed(*, caller: str) -> None:
    """Avoid tearing down shared clients between main/arena phases in runner_loop."""
    if _DEFER_HTTP_CLIENT_CLOSE:
        _stderr_boot(f"{caller} exit: close_http_clients deferred")
        return
    _stderr_boot(f"{caller} exit: close_http_clients")
    close_http_clients()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


@lru_cache(maxsize=1)
def _get_runner_build_info() -> str:
    """Return git branch/commit information for boot logs when available."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        branch = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        state = "dirty" if dirty else "clean"
        return f"branch={branch or 'unknown'} commit={commit or 'unknown'} state={state}"
    except Exception as e:
        return f"branch=unknown commit=unknown state=unavailable({type(e).__name__})"


def _format_runner_startup_context(*, version: Optional[str] = None) -> str:
    parts = [_get_runner_build_info()]
    if version:
        parts.append(f"version={version}")
    return " ".join(parts)


def _stderr_boot(message: str) -> None:
    """Emit startup/debug messages even if logging is misconfigured."""
    try:
        print(f"[runner-boot] {message}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _trace_miner_call(message: str) -> None:
    """Emit per-miner-call traces to stderr when enabled."""
    if _env_flag("BB_TRACE_MINER_CALLS", default=True):
        _stderr_boot(f"miner-call {message}")


def _enforce_runner_logging_level() -> None:
    """Keep runner loggers at INFO+ even if external libs mutate log levels."""
    try:
        for name in (
            "babelbit",
            "babelbit.cli.runner",
            "babelbit.utils.predict_utterances",
            "babelbit.utils.predict_engine",
            "babelbit.utils.managed_container_registry",
        ):
            getLogger(name).setLevel(INFO)
    except Exception:
        pass


def _coerce_timeout_seconds(value: object, default: float = 10.0) -> float:
    """Convert timeout config to float safely (handles mocked settings in tests)."""
    try:
        parsed = float(value)
        if parsed > 0:
            return parsed
    except Exception:
        pass
    return default


def _serialize_prediction_step(step: BBPredictedUtterance) -> dict:
    if hasattr(step, "model_dump"):
        return step.model_dump(mode="json")
    if hasattr(step, "dict"):
        return step.dict()
    return {
        "index": getattr(step, "index", ""),
        "step": getattr(step, "step", 0),
        "prefix": getattr(step, "prefix", ""),
        "prediction": getattr(step, "prediction", ""),
        "context": getattr(step, "context", ""),
        "done": getattr(step, "done", False),
        "ground_truth": getattr(step, "ground_truth", None),
    }


def _resolve_score_parallelism() -> int:
    cpu_count = os.cpu_count() or 1
    try:
        configured = int(os.getenv("BB_SCORE_PARALLELISM", str(min(4, cpu_count))))
    except Exception:
        configured = min(4, cpu_count)
    configured = max(1, configured)
    return min(configured, cpu_count)


def _resolve_score_io_parallelism(score_parallelism: int) -> int:
    try:
        configured = int(os.getenv("BB_SCORE_IO_PARALLELISM", str(max(2, score_parallelism))))
    except Exception:
        configured = max(2, score_parallelism)
    return max(1, configured)


def _resolve_score_worker_threads(score_parallelism: int) -> int:
    cpu_count = os.cpu_count() or 1
    try:
        configured = int(os.getenv("BB_SCORE_TORCH_THREADS", "0"))
    except Exception:
        configured = 0
    if configured > 0:
        return configured
    return max(1, cpu_count // max(1, score_parallelism))


def _resolve_arena_startup_timeout() -> float:
    try:
        configured = float(os.getenv("BB_ARENA_STARTUP_TIMEOUT_SEC", "300"))
    except Exception:
        configured = 300.0
    return max(1.0, configured)


def _resolve_arena_startup_request_timeout() -> float:
    try:
        configured = float(os.getenv("BB_ARENA_STARTUP_REQUEST_TIMEOUT_SEC", "60"))
    except Exception:
        configured = 60.0
    return max(1.0, configured)


def _init_scoring_worker(torch_threads: int) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("BB_SCORER_DEVICE", "cpu")
    try:
        import torch

        torch.set_num_threads(max(1, torch_threads))
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass


def _get_scoring_process_pool(*, max_workers: int, torch_threads: int) -> ProcessPoolExecutor:
    global _SCORING_PROCESS_POOL, _SCORING_PROCESS_POOL_CONFIG
    desired_config = (max_workers, torch_threads)
    if _SCORING_PROCESS_POOL is not None and _SCORING_PROCESS_POOL_CONFIG == desired_config:
        return _SCORING_PROCESS_POOL

    if _SCORING_PROCESS_POOL is not None:
        _SCORING_PROCESS_POOL.shutdown(wait=False, cancel_futures=True)

    _SCORING_PROCESS_POOL = ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_scoring_worker,
        initargs=(torch_threads,),
    )
    _SCORING_PROCESS_POOL_CONFIG = desired_config
    return _SCORING_PROCESS_POOL


def _shutdown_scoring_process_pool() -> None:
    global _SCORING_PROCESS_POOL, _SCORING_PROCESS_POOL_CONFIG
    if _SCORING_PROCESS_POOL is not None:
        _SCORING_PROCESS_POOL.shutdown(wait=False, cancel_futures=True)
        _SCORING_PROCESS_POOL = None
        _SCORING_PROCESS_POOL_CONFIG = None


def _should_use_scoring_process_pool() -> bool:
    if score_jsonl is None:
        return False
    module_name = getattr(score_jsonl, "__module__", "")
    return module_name.startswith("babelbit.scoring.")


atexit.register(_shutdown_scoring_process_pool)


def _track_background_task(task: asyncio.Task, *, label: str) -> None:
    """Keep background tasks referenced and surface exceptions in logs."""
    _BACKGROUND_TASKS.add(task)

    def _done_callback(completed: asyncio.Task) -> None:
        _BACKGROUND_TASKS.discard(completed)
        try:
            exc = completed.exception()
        except asyncio.CancelledError:
            logger.info("[runner] background task cancelled: %s", label)
            return
        except Exception as e:
            logger.warning("[runner] background task inspection failed for %s: %s", label, e)
            return
        if exc is not None:
            logger.warning("[runner] background task failed (%s): %s: %s", label, type(exc).__name__, exc)

    task.add_done_callback(_done_callback)


def group_steps_into_utterances(utterance_steps: List[BBPredictedUtterance]) -> List[List[BBPredictedUtterance]]:
    """
    Group utterance steps into complete utterances.
    Each utterance ends when done=True (EOF token).
    """
    complete_utterances = []
    current_utterance_steps = []
    
    for step in utterance_steps:
        current_utterance_steps.append(step)
        
        # If this step marks the end of an utterance (done=True/EOF)
        if step.done:
            complete_utterances.append(current_utterance_steps.copy())
            current_utterance_steps = []
    
    # Handle any remaining steps that didn't form a complete utterance
    if current_utterance_steps:
        logger.warning(f"Found {len(current_utterance_steps)} incomplete utterance steps at end of dialogue")
        complete_utterances.append(current_utterance_steps)
    
    return complete_utterances


def _prepare_dialogue_score_artifacts(
    *,
    challenge_uid: Optional[str],
    challenge_type: str,
    logs_dir: Path,
    scores_dir: Path,
    miner_uid: int,
    miner_hotkey: str,
    dialogue_uid: str,
    utterance_steps: List[BBPredictedUtterance],
) -> Optional[Dict[str, object]]:
    """Build raw logs and score artifacts for one dialogue."""
    miner_id = f"uid_{miner_uid}"
    logger.info(
        "Miner %s produced %d utterance steps in dialogue %s",
        miner_id,
        len(utterance_steps),
        dialogue_uid,
    )
    complete_utterances = group_steps_into_utterances(utterance_steps)
    logger.info("Dialogue %s contains %d complete utterances", dialogue_uid, len(complete_utterances))

    scoring_events: List[dict] = []
    events_path = (
        logs_dir
        / (
            f"dialogue_run_{challenge_uid or 'unknown'}_type_{challenge_type}"
            f"_miner_{miner_uid}__hk_{miner_hotkey}__dlg_{dialogue_uid}.jsonl"
        )
    )
    with events_path.open("w", encoding="utf-8") as jf:
        for utt_index, utt_steps in enumerate(complete_utterances):
            gt = getattr(utt_steps[-1], "ground_truth", "") or ""
            if not gt.strip():
                logger.warning(
                    "Skipping utterance %d in dialogue %s for miner %s - empty ground_truth (likely timeout or early session termination)",
                    utt_index,
                    dialogue_uid,
                    miner_id,
                )
                logger.debug(
                    "[runner] skipped utt_index=%d (empty GT) in dialogue=%s miner=%s",
                    utt_index,
                    dialogue_uid,
                    miner_id,
                )
                continue

            for step_idx, step_obj in enumerate(utt_steps):
                predicted_event = {
                    "event": "predicted",
                    "challenge_type": challenge_type,
                    "utterance_index": utt_index,
                    "step": step_idx,
                    "prediction": getattr(step_obj, "prediction", "") or "",
                }
                scoring_events.append(predicted_event)
                jf.write(json.dumps(predicted_event) + "\n")

            complete_event = {
                "event": "utterance_complete",
                "challenge_type": challenge_type,
                "utterance_index": utt_index,
                "ground_truth": gt,
            }
            scoring_events.append(complete_event)
            jf.write(json.dumps(complete_event) + "\n")

    logger.info("[runner] Wrote raw dialogue log: %s", events_path)
    logger.debug("[runner] events_path size=%d bytes", events_path.stat().st_size if events_path.exists() else -1)

    if score_jsonl is None:
        logger.warning("score_jsonl unavailable; skipping scoring for dialogue %s", dialogue_uid)
        return None

    try:
        prime_score_jsonl_events(events_path, scoring_events)
        scored_doc = copy.deepcopy(score_jsonl(events_path))
        logger.debug(
            "[runner] score_jsonl produced %d utterances for dialogue %s",
            len(scored_doc.get("utterances", [])),
            dialogue_uid,
        )
        scored_doc.update({
            "challenge_uid": challenge_uid,
            "challenge_type": challenge_type,
            "miner_uid": miner_uid,
            "miner_hotkey": miner_hotkey,
            "dialogue_uid": dialogue_uid,
        })
        avg_u = float(scored_doc.get("dialogue_summary", {}).get("average_U_best_early", 0.0))
        score_path = Path(save_dialogue_score_file(scored_doc, output_dir=str(scores_dir)))
        logger.info("[runner] Scored dialogue %s U=%.4f", dialogue_uid, avg_u)
        return {
            "challenge_uid": challenge_uid,
            "challenge_type": challenge_type,
            "miner_uid": miner_uid,
            "miner_hotkey": miner_hotkey,
            "dialogue_uid": dialogue_uid,
            "dialogue_index": None,
            "avg_u": avg_u,
            "score_path": str(score_path),
            "events_path": str(events_path),
        }
    except Exception as e:
        logger.warning("Failed scoring dialogue %s: %s", dialogue_uid, e)
        return None


def _score_dialogue_cpu_worker(task: Dict[str, object]) -> Optional[Dict[str, object]]:
    utterance_steps = [
        BBPredictedUtterance.model_validate(step) if isinstance(step, dict) else step
        for step in task["utterance_steps"]
    ]
    return _prepare_dialogue_score_artifacts(
        challenge_uid=task.get("challenge_uid"),
        challenge_type=str(task["challenge_type"]),
        logs_dir=Path(str(task["logs_dir"])),
        scores_dir=Path(str(task["scores_dir"])),
        miner_uid=int(task["miner_uid"]),
        miner_hotkey=str(task["miner_hotkey"]),
        dialogue_uid=str(task["dialogue_uid"]),
        utterance_steps=utterance_steps,
    )


async def _score_miners_for_challenge(
    *,
    challenge_uid: Optional[str],
    challenge_type: str,
    miner_list: List[Miner],
    miner_dialogues: Dict[str, Dict[str, List[BBPredictedUtterance]]],
    logs_dir: Path,
    scores_dir: Path,
    submission_client: ValidationSubmissionClient,
    active_s3_manager: Optional[S3Manager],
    main_challenge_uid: Optional[str] = None,
) -> tuple[int, int, List[float]]:
    """Persist dialogue logs, score miners, and return aggregate stats."""
    main_challenge_uid = main_challenge_uid or challenge_uid
    artifact_challenge_type = challenge_type
    score_parallelism = _resolve_score_parallelism()
    io_parallelism = _resolve_score_io_parallelism(score_parallelism)
    score_worker_threads = _resolve_score_worker_threads(score_parallelism)
    dialogue_semaphore = asyncio.Semaphore(score_parallelism)
    io_semaphore = asyncio.Semaphore(io_parallelism)
    scoring_pool = _get_scoring_process_pool(
        max_workers=score_parallelism,
        torch_threads=score_worker_threads,
    ) if _should_use_scoring_process_pool() else None

    async def _process_dialogue(miner: Miner, dialogue_index: int, dialogue_uid: str, utterance_steps: List[BBPredictedUtterance]) -> Optional[Dict[str, object]]:
        nonlocal scoring_pool
        async with dialogue_semaphore:
            if scoring_pool is not None:
                loop = asyncio.get_running_loop()
                task_payload = {
                    "challenge_uid": challenge_uid,
                    "challenge_type": artifact_challenge_type,
                    "logs_dir": str(logs_dir),
                    "scores_dir": str(scores_dir),
                    "miner_uid": getattr(miner, "uid", None),
                    "miner_hotkey": getattr(miner, "hotkey", None),
                    "dialogue_uid": dialogue_uid,
                    "utterance_steps": [_serialize_prediction_step(step) for step in utterance_steps],
                }
                try:
                    artifact = await loop.run_in_executor(
                        scoring_pool,
                        _score_dialogue_cpu_worker,
                        task_payload,
                    )
                except BrokenProcessPool as e:
                    logger.warning(
                        "Scoring process pool broke while processing miner uid=%s hotkey=%s dialogue=%s; "
                        "falling back to thread scoring for the rest of this run: %s",
                        getattr(miner, "uid", "?"),
                        getattr(miner, "hotkey", "?"),
                        dialogue_uid,
                        e,
                    )
                    _shutdown_scoring_process_pool()
                    scoring_pool = None
                    artifact = await asyncio.to_thread(
                        _prepare_dialogue_score_artifacts,
                        challenge_uid=challenge_uid,
                        challenge_type=artifact_challenge_type,
                        logs_dir=logs_dir,
                        scores_dir=scores_dir,
                        miner_uid=getattr(miner, "uid", None),
                        miner_hotkey=getattr(miner, "hotkey", None),
                        dialogue_uid=dialogue_uid,
                        utterance_steps=utterance_steps,
                    )
            else:
                artifact = await asyncio.to_thread(
                    _prepare_dialogue_score_artifacts,
                    challenge_uid=challenge_uid,
                    challenge_type=artifact_challenge_type,
                    logs_dir=logs_dir,
                    scores_dir=scores_dir,
                    miner_uid=getattr(miner, "uid", None),
                    miner_hotkey=getattr(miner, "hotkey", None),
                    dialogue_uid=dialogue_uid,
                    utterance_steps=utterance_steps,
                )

        if artifact is None:
            return None

        artifact["dialogue_index"] = dialogue_index
        events_path = Path(str(artifact["events_path"]))
        score_path = Path(str(artifact["score_path"]))
        s3_log_path = None
        s3_score_path = None

        async with io_semaphore:
            if active_s3_manager:
                s3_log_path = f"{settings.S3_LOG_DIR}/logs/{events_path.name}"
                await asyncio.to_thread(active_s3_manager.upload_file, str(events_path), s3_log_path)
                logger.info("Uploaded raw dialogue log to S3: s3://%s/%s", active_s3_manager.bucket_name, s3_log_path)

            if submission_client.is_ready:
                max_attempts = 4
                for attempt in range(1, max_attempts + 1):
                    try:
                        ok = await submission_client.submit_validation_file(
                            file_path=events_path,
                            file_type="dialogue_run",
                            kind="dialogue_logs",
                            challenge_id=challenge_uid or "",
                            main_challenge_uid=main_challenge_uid,
                            miner_uid=getattr(miner, "uid", None),
                            miner_hotkey=getattr(miner, "hotkey", None),
                            dialogue_uid=dialogue_uid,
                            s3_path=s3_log_path,
                            extra_data={"challenge_type": artifact_challenge_type},
                        )
                    except Exception as e:
                        ok = False
                        logger.warning("Validation submission error for %s: %s", events_path.name, e)
                    if ok:
                        break
                    if attempt < max_attempts:
                        backoff_s = min(2**attempt, 12)
                        logger.info(
                            "Retrying validation submission for %s in %ss (attempt %d/%d)",
                            events_path.name,
                            backoff_s,
                            attempt + 1,
                            max_attempts,
                        )
                        await asyncio.sleep(backoff_s)

            if active_s3_manager:
                s3_score_path = f"submissions/{score_path.name}"
                await asyncio.to_thread(active_s3_manager.upload_file, str(score_path), s3_score_path)
                logger.info("Uploaded dialogue score to S3: s3://%s/%s", active_s3_manager.bucket_name, s3_score_path)

            if submission_client.is_ready:
                try:
                    await submission_client.submit_validation_file(
                        file_path=score_path,
                        file_type="dialogue_scores",
                        kind="dialogue_scores",
                        challenge_id=challenge_uid or "",
                        main_challenge_uid=main_challenge_uid,
                        miner_uid=getattr(miner, "uid", None),
                        miner_hotkey=getattr(miner, "hotkey", None),
                        dialogue_uid=dialogue_uid,
                        s3_path=s3_score_path,
                        extra_data={"challenge_type": artifact_challenge_type},
                    )
                except Exception as e:
                    logger.warning("Validation submission error for %s: %s", score_path, e)

        return artifact

    async def _process_miner(m: Miner) -> tuple[int, int, Optional[float]]:
        try:
            miner_key = m.hotkey
            miner_id = f"uid_{m.uid}"

            dialogues = miner_dialogues.get(miner_key, {})
            logger.debug(
                "[runner] miner uid=%s hk=%s dialogues_count=%d",
                getattr(m, "uid", "?"),
                (m.hotkey[:16] + "..."),
                len(dialogues or {}),
            )

            if not dialogues:
                logger.warning("Miner %s (uid: %s, hotkey: %s...) has no dialogues to score", miner_id, m.uid, m.hotkey[:16])
                return 0, 0, None

            has_valid_predictions = False
            for dialogue_uid, utterance_steps in dialogues.items():
                for step in utterance_steps:
                    prediction = getattr(step, "prediction", "") or ""
                    if prediction.strip():
                        has_valid_predictions = True
                        break
                if has_valid_predictions:
                    break

            if not has_valid_predictions:
                logger.warning(
                    "Miner %s (uid: %s) has no valid predictions across %d dialogues - skipping scoring",
                    miner_id,
                    m.uid,
                    len(dialogues),
                )
                logger.debug("[runner] miner %s invalid/empty predictions; skipping", miner_id)
                return 0, 0, None

            logger.info(
                "Processing %d dialogues for miner %s (uid: %s, hotkey: %s...)",
                len(dialogues),
                miner_id,
                m.uid,
                m.hotkey[:16],
            )

            dialogue_results: List[Dict[str, object]] = []
            for dialogue_index, (dialogue_uid, utterance_steps) in enumerate(dialogues.items()):
                result = await _process_dialogue(m, dialogue_index, dialogue_uid, utterance_steps)
                if result is not None:
                    dialogue_results.append(result)

            if not dialogue_results:
                logger.debug("[runner] no dialogue scores for miner %s", miner_id)
                return 0, 0, None

            dialogue_results.sort(key=lambda item: int(item.get("dialogue_index", 0)))
            dialogue_scores = [float(item["avg_u"]) for item in dialogue_results]
            dialogue_uids = [str(item["dialogue_uid"]) for item in dialogue_results]
            miner_mean_score = sum(dialogue_scores) / len(dialogue_scores)
            summary = {
                "challenge_uid": challenge_uid,
                "challenge_type": artifact_challenge_type,
                "miner_uid": getattr(m, "uid", None),
                "miner_hotkey": getattr(m, "hotkey", None),
                "dialogues": [
                    {"dialogue_uid": duid, "dialogue_average_u_best_early": ds, "dialogue_index": idx}
                    for idx, (duid, ds) in enumerate(zip(dialogue_uids, dialogue_scores))
                ],
                "challenge_mean_U": miner_mean_score,
            }
            summary_path = save_challenge_summary_file(summary, output_dir=str(scores_dir))
            logger.debug(
                "[runner] saved challenge summary for miner %s: path=%s dialogues=%d mean=%.4f",
                miner_id,
                str(summary_path),
                len(dialogue_scores),
                miner_mean_score,
            )

            s3_sub_path = None
            async with io_semaphore:
                if active_s3_manager:
                    s3_sub_path = f"submissions/{Path(summary_path).name}"
                    await asyncio.to_thread(active_s3_manager.upload_file, str(summary_path), s3_sub_path)
                    logger.info("Uploaded challenge summary to S3: s3://%s/%s", active_s3_manager.bucket_name, s3_sub_path)
                if submission_client.is_ready:
                    try:
                        await submission_client.submit_validation_file(
                            file_path=Path(summary_path),
                            file_type="challenge_scores",
                            kind="challenge_scores",
                            challenge_id=challenge_uid or "",
                            main_challenge_uid=main_challenge_uid,
                            miner_uid=getattr(m, "uid", None),
                            miner_hotkey=getattr(m, "hotkey", None),
                            dialogue_uid=None,
                            s3_path=s3_sub_path,
                            extra_data={"challenge_type": artifact_challenge_type},
                        )
                    except Exception as e:
                        logger.warning("Validation submission error for %s: %s", summary_path, e)

            logger.debug("[runner] challenge processed: uid=%s miners=1 dialogues=%d mean=%s", challenge_uid, len(dialogue_scores), f"{miner_mean_score:.4f}")
            return 1, len(dialogue_scores), miner_mean_score
        except Exception as e:
            logger.warning(
                "Failed to process miner uid=%s hotkey=%s: %s",
                getattr(m, "uid", "?"),
                getattr(m, "hotkey", "?"),
                e,
            )
            return 0, 0, None

    miner_results = await asyncio.gather(*(_process_miner(m) for m in miner_list))
    total_miners_processed = sum(result[0] for result in miner_results)
    total_dialogues_processed = sum(result[1] for result in miner_results)
    all_challenge_scores = [result[2] for result in miner_results if result[2] is not None]
    return total_miners_processed, total_dialogues_processed, all_challenge_scores


async def _run_solo_challenge_phase(
    *,
    utterance_engine_url: str,
    miner_list: List[Miner],
    miner_timeout: float,
    challenge_uid: Optional[str],
    main_challenge_uid: Optional[str],
    logs_dir: Path,
    scores_dir: Path,
    submission_client: ValidationSubmissionClient,
    active_s3_manager: Optional[S3Manager],
) -> None:
    """
    Run the solo challenge work off the critical path.

    The main challenge is already recorded by the time this starts, so the validator
    can move on to the next cycle without waiting for the solo scoring tail.
    """
    try:
        from babelbit.utils.predict_engine import call_miner_axon_endpoint

        async def prediction_callback(miner: Miner, payload: BBPredictedUtterance, context: str) -> str:
            if miner.axon_ip and miner.axon_port:
                result = await call_miner_axon_endpoint(
                    axon_ip=miner.axon_ip,
                    axon_port=miner.axon_port,
                    payload=payload,
                    context_used=context,
                    miner_hotkey=miner.hotkey,
                    timeout=miner_timeout,
                )
                if result.success and result.utterance:
                    return result.utterance.prediction or ""
                raise RuntimeError(str(result.error))
            raise RuntimeError(f"Miner {getattr(miner, 'uid', '?')} has no axon endpoint available")

        solo_dialogues, solo_uid, solo_status = await predict_with_utterance_engine_multi_miner(
            utterance_engine_url=utterance_engine_url,
            miners=miner_list,
            prediction_callback=prediction_callback,
            timeout=miner_timeout,
            max_prediction_errors=5,
            subtensor=None,
            step_block_modulo=0,
            solo=True,
            continue_after_all_miners_deactivated=True,
            miner_key_fn=lambda miner: getattr(miner, "hotkey", None),
            return_challenge_uid=True,
            return_miner_status=True,
        )
        solo_results = {
            miner_key: {"challenge_uid": solo_uid, "dialogues": dialogues}
            for miner_key, dialogues in (solo_dialogues or {}).items()
            if solo_status.get(miner_key, True)
        }
    except Exception as e:
        logger.warning("[Solo Challenge] Failed to run solo challenge: %s", e)
        return

    if not solo_results:
        logger.info("[Solo Challenge] No solo challenge results returned; skipping solo scoring")
        return

    solo_uid = next(
        (result.get("challenge_uid") for result in solo_results.values() if result.get("challenge_uid")),
        None,
    )
    if not solo_uid:
        logger.warning("[Solo Challenge] No challenge UID available in solo results; skipping scoring")
        return

    solo_miner_list: List[Miner] = []
    solo_miner_dialogues: Dict[str, Dict[str, List[BBPredictedUtterance]]] = {}
    missing_uid_hotkeys: List[str] = []

    for miner in miner_list:
        miner_key = getattr(miner, "hotkey", None)
        if not miner_key or miner_key not in solo_results:
            continue
        result = solo_results[miner_key]
        result_uid = result.get("challenge_uid")
        if not result_uid:
            missing_uid_hotkeys.append(miner_key)
            continue
        solo_miner_list.append(miner)
        solo_miner_dialogues[miner_key] = result.get("dialogues") or {}

    if missing_uid_hotkeys:
        logger.warning(
            "[Solo Challenge] Skipping %d miners with missing solo challenge UID: %s",
            len(missing_uid_hotkeys),
            ", ".join(missing_uid_hotkeys[:5]),
        )

    if not solo_miner_list:
        logger.info("[Solo Challenge] No miners processed during solo phase")
        return

    score_embedder_prewarm_task = asyncio.create_task(asyncio.to_thread(prewarm_score_dialogue_embedder))
    try:
        await score_embedder_prewarm_task
    except Exception as e:
        logger.warning("[Solo Challenge] Score embedder prewarm failed: %s", e)

    solo_total_miners, solo_total_dialogues, solo_scores = await _score_miners_for_challenge(
        challenge_uid=solo_uid,
        challenge_type="solo",
        miner_list=solo_miner_list,
        miner_dialogues=solo_miner_dialogues,
        logs_dir=logs_dir,
        scores_dir=scores_dir,
        submission_client=submission_client,
        active_s3_manager=active_s3_manager,
        main_challenge_uid=main_challenge_uid,
    )

    if solo_total_miners == 0:
        logger.info("[Solo Challenge] No miners processed during solo scoring")
        return

    solo_mean = sum(solo_scores) / len(solo_scores) if solo_scores else None
    mark_challenge_processed(
        challenge_uid=solo_uid,
        miner_count=solo_total_miners,
        total_dialogues=solo_total_dialogues,
        mean_score=solo_mean,
        challenge_type="solo",
        metadata={
            "scores_dir": str(scores_dir),
            "logs_dir": str(logs_dir),
            "solo_challenge": True,
            "paired_challenge_uid": challenge_uid,
        },
    )
    solo_mean_str = f"{solo_mean:.4f}" if solo_mean is not None else "N/A"
    logger.info(
        f"[Solo Challenge] Completed {solo_uid}: {solo_total_miners} miners, "
        f"{solo_total_dialogues} dialogues, mean_score={solo_mean_str}"
    )


def _build_round2_prediction_callback(
    *,
    routes_by_hotkey: Dict[str, ManagedRoute],
    miner_timeout: float,
    startup_timeout: float,
    startup_request_timeout: float,
):
    from babelbit.utils.predict_engine import call_managed_route_endpoint

    first_call_seen: set[str] = set()

    def _is_non_retryable_startup_error(error: str) -> bool:
        lowered = str(error or "").strip().lower()
        if not lowered:
            return False
        non_retryable_prefixes = (
            "missing_miner_uid_for_gateway",
            "empty_gateway_auth_url",
            "gateway_auth_failed",
            "400:",
            "401:",
            "403:",
            "404:",
            "422:",
        )
        return any(lowered.startswith(prefix) for prefix in non_retryable_prefixes)

    async def prediction_callback(miner: Miner, payload: BBPredictedUtterance, context: str) -> str:
        route = routes_by_hotkey.get(getattr(miner, "hotkey", ""))
        miner_uid = getattr(miner, "uid", "?")
        miner_hotkey_short = (miner.hotkey[:16] + "...") if getattr(miner, "hotkey", None) else "?"
        step_value = getattr(payload, "step", "?")
        miner_hotkey = getattr(miner, "hotkey", "") or ""
        is_startup_call = miner_hotkey not in first_call_seen
        effective_timeout = miner_timeout
        if route is None:
            _trace_miner_call(
                f"arena-skipped uid={miner_uid} hotkey={miner_hotkey_short} reason=no_managed_route",
            )
            raise RuntimeError(f"Miner {miner_uid} has no managed route")

        started_at = time.perf_counter()
        prefix_chars = len((getattr(payload, "prefix", "") or ""))
        provider = str(getattr(route, "provider", "managed_container") or "managed_container")
        route_target = str(getattr(route, "endpoint_url", "") or "")
        _trace_miner_call(
            f"arena-start uid={miner_uid} hotkey={miner_hotkey_short} provider={provider} target={route_target} step={step_value} timeout={effective_timeout:.2f}s startup_retry_timeout={startup_timeout:.2f}s startup_request_timeout={startup_request_timeout:.2f}s prefix_chars={prefix_chars}",
        )
        try:
            if is_startup_call:
                startup_deadline = time.monotonic() + max(1.0, float(startup_timeout))
                startup_attempt = 0
                last_error = "arena_startup_timeout"
                while True:
                    remaining = startup_deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError(last_error)

                    startup_attempt += 1
                    result = await call_managed_route_endpoint(
                        route=route,
                        payload=payload,
                        context_used=context,
                        miner_hotkey=miner.hotkey,
                        timeout=min(float(startup_request_timeout), max(0.1, remaining)),
                    )
                    prediction_text = ""
                    if result.utterance:
                        prediction_text = result.utterance.prediction or ""
                    if result.success and prediction_text:
                        first_call_seen.add(miner_hotkey)
                        break

                    last_error = str(result.error or "arena_startup_predict_failed")
                    if result.success and not prediction_text:
                        last_error = "arena_startup_predict_failed:empty_prediction"

                    if _is_non_retryable_startup_error(last_error):
                        raise RuntimeError(last_error)

                    logger.info(
                        "Arena startup retry pending for uid=%s hotkey=%s attempt=%d error=%s",
                        miner_uid,
                        miner_hotkey_short,
                        startup_attempt,
                        last_error,
                    )
                    await asyncio.sleep(min(1.0, max(0.0, startup_deadline - time.monotonic())))
            else:
                result = await call_managed_route_endpoint(
                    route=route,
                    payload=payload,
                    context_used=context,
                    miner_hotkey=miner.hotkey,
                    timeout=effective_timeout,
                )

            if result.success and result.utterance:
                latency = time.perf_counter() - started_at
                prediction_text = result.utterance.prediction or ""
                _trace_miner_call(
                    f"arena-success uid={miner_uid} hotkey={miner_hotkey_short} step={step_value} latency={latency:.2f}s prediction_chars={len(prediction_text)}",
                )
                return prediction_text

            latency = time.perf_counter() - started_at
            _trace_miner_call(
                f"arena-failed uid={miner_uid} hotkey={miner_hotkey_short} step={step_value} latency={latency:.2f}s error={result.error}",
            )
            raise RuntimeError(str(result.error))
        except Exception as e:
            latency = time.perf_counter() - started_at
            _trace_miner_call(
                f"arena-exception uid={miner_uid} hotkey={miner_hotkey_short} step={step_value} latency={latency:.2f}s err={type(e).__name__}:{e}",
            )
            raise

    return prediction_callback


def _build_managed_health_url(endpoint_url: str, predict_endpoint: str) -> str:
    """Derive a health URL from a managed endpoint URL."""
    url = (endpoint_url or "").strip()
    if not url:
        return ""
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"

    endpoint = str(predict_endpoint or "predict").strip().lstrip("/")
    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/")
    if not path:
        health_path = "/health"
    elif endpoint and path.split("/")[-1] == endpoint:
        base_path = path.rsplit("/", 1)[0]
        health_path = f"{base_path}/health" if base_path else "/health"
    else:
        health_path = f"{path}/health"
    return urlunparse(parsed._replace(path=health_path, query="", fragment=""))


async def _wait_for_round2_routes_health(
    *,
    routes_by_hotkey: Dict[str, ManagedRoute],
    max_wait_seconds: float = 300.0,
    ping_timeout_seconds: float = 30.0,
    ping_interval_seconds: float = 1.0,
) -> tuple[set[str], dict[str, str]]:
    """Wait for managed routes to report healthy, bounded by max_wait_seconds."""
    if not routes_by_hotkey:
        return set(), {}

    settings = get_settings()
    predict_endpoint = str(getattr(settings, "BB_MINER_PREDICT_ENDPOINT", "predict"))
    session = await get_async_client()

    pending: dict[str, ManagedRoute] = dict(routes_by_hotkey)
    healthy: set[str] = set()
    last_errors: dict[str, str] = {}
    started_at = time.monotonic()
    deadline = started_at + max(0.1, float(max_wait_seconds))
    attempt = 0
    timeout = max(0.1, float(ping_timeout_seconds))
    interval = max(0.0, float(ping_interval_seconds))

    # Provider-fronted routes do not expose a direct per-miner /health probe.
    skipped_health_probe = 0
    for hotkey, route in list(pending.items()):
        provider = str(getattr(route, "provider", "") or "").strip().lower()
        is_provider_route = provider == "gateway"
        if is_provider_route:
            healthy.add(hotkey)
            pending.pop(hotkey, None)
            skipped_health_probe += 1

    if skipped_health_probe:
        logger.info("Arena health gate bypassed for %d provider route(s)", skipped_health_probe)

    while pending and time.monotonic() < deadline:
        attempt += 1

        async def _ping_one(hotkey: str, route: ManagedRoute) -> tuple[str, bool, str]:
            health_url = _build_managed_health_url(route.endpoint_url, predict_endpoint)
            if not health_url:
                return hotkey, False, "empty_health_url"
            try:
                async with session.get(health_url, timeout=ClientTimeout(total=timeout)) as response:
                    if response.status == 200:
                        return hotkey, True, ""
                    body = (await response.text())[:120]
                    return hotkey, False, f"status={response.status} body={body}"
            except Exception as exc:
                return hotkey, False, f"{type(exc).__name__}:{exc}"

        results = await asyncio.gather(
            *(_ping_one(hotkey, route) for hotkey, route in pending.items()),
            return_exceptions=False,
        )

        for hotkey, ok, reason in results:
            if ok:
                healthy.add(hotkey)
                pending.pop(hotkey, None)
                last_errors.pop(hotkey, None)
            else:
                last_errors[hotkey] = reason

        elapsed = time.monotonic() - started_at
        if pending:
            logger.info(
                "Arena health gate waiting healthy=%d/%d pending=%d attempt=%d elapsed=%.1fs",
                len(healthy),
                len(routes_by_hotkey),
                len(pending),
                attempt,
                elapsed,
            )
            if interval > 0:
                await asyncio.sleep(interval)

    unresolved: dict[str, str] = {}
    for hotkey in pending:
        unresolved[hotkey] = last_errors.get(hotkey, "health_check_timeout")
    return healthy, unresolved


async def runner(utterance_engine_url: str | None = None, output_dir: Optional[str] = None, subtensor=None) -> None:
    _enforce_runner_logging_level()
    settings = get_settings()
    NETUID = settings.BABELBIT_NETUID
    MAX_MINERS = int(os.getenv("BB_MAX_MINERS_PER_RUN", "256"))
    utterance_engine_url = utterance_engine_url or os.getenv("BB_UTTERANCE_ENGINE_URL", "http://localhost:8000")
    enable_solo_challenge = os.getenv("BB_ENABLE_SOLO_CHALLENGE", "1").lower() in {"1", "true", "yes"}
    startup_context = _format_runner_startup_context(version=getattr(settings, "BABELBIT_VERSION", None))
    _stderr_boot(
        "runner entry "
        f"utterance_engine_url={utterance_engine_url} "
        f"netuid={NETUID} max_miners={MAX_MINERS} "
        f"solo_enabled={enable_solo_challenge} "
        f"{startup_context}",
    )
    logger.info("[RunnerBoot] %s", startup_context)
    
    # Determine directories:
    #   Raw logs:   ./logs (override with BB_OUTPUT_LOGS_DIR)
    #   Scores:     ./scores (override with BB_OUTPUT_SCORES_DIR or output_dir argument) produced after scoring
    #   output_dir argument retained for backward compatibility
    logs_dir = Path(os.getenv("BB_OUTPUT_LOGS_DIR", "logs"))
    scores_dir = Path(output_dir or os.getenv("BB_OUTPUT_SCORES_DIR", "scores"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(
        "[runner] output dirs ready: logs_dir=%s scores_dir=%s (output_dir_arg=%s)",
        str(logs_dir), str(scores_dir), str(output_dir),
    )

    s3_enabled = os.getenv("BB_ENABLE_S3_UPLOADS", "0").lower() in {"1", "true", "yes"}
    global s3_manager
    if s3_enabled and s3_manager is None:
        try:
            s3_manager = S3Manager(
                bucket_name=settings.S3_BUCKET_NAME,
                access_key=settings.S3_ACCESS_KEY_ID,
                secret_key=settings.S3_SECRET_ACCESS_KEY.get_secret_value(),
                endpoint_url=settings.S3_ENDPOINT_URL or None,
                region=settings.S3_REGION,
                addressing_style=settings.S3_ADDRESSING_STYLE or "auto",
                signature_version=settings.S3_SIGNATURE_VERSION or "s3v4",
                use_ssl=settings.S3_USE_SSL,
                prefix="",  # Empty prefix so logs go directly to bucket/logs/
            )
            logger.info("S3 Manager initialized (uploads enabled)")
        except Exception as e:
            logger.warning("S3 Manager initialization failed; disabling S3 uploads: %s", e)
            s3_manager = None
    logger.debug("[runner] S3 uploads enabled=%s active=%s", s3_enabled, bool(s3_manager))

    submission_client = ValidationSubmissionClient()
    logger.debug(
        "[runner] validation submissions ready=%s endpoint=%s",
        submission_client.is_ready,
        submission_client.submit_url if submission_client else "N/A",
    )

    try:
        challenge_uid = await get_current_challenge_uid(utterance_engine_url)
        _stderr_boot(f"runner challenge_uid={challenge_uid}")
    except Exception as e:
        _stderr_boot(f"runner challenge_uid fetch failed: {type(e).__name__}: {e}")
        logger.warning(f"Could not get current challenge ID: {e}")
        return
    logger.debug("[runner] fetched challenge_uid=%s from %s", challenge_uid, utterance_engine_url)

    # Prevents runner loop from running multiple times a challenge
    if challenge_uid:
        already_processed = get_processed_miners_for_challenge(
            str(scores_dir),
            challenge_uid,
            challenge_type="main",
        )
        if already_processed:
            _stderr_boot(
                f"runner skip already_processed challenge_uid={challenge_uid} "
                f"miners={len(already_processed)}",
            )
            logger.info(
                f"Challenge {challenge_uid} already has {len(already_processed)} scored miners. "
                f"Skipping entire run to avoid duplicate work."
            )
            return
        else:
            logger.info(f"Challenge {challenge_uid}: No existing scores found, proceeding with miner evaluation")
            logger.debug("[runner] already_processed=%s", list(already_processed) if already_processed else [])

    try:
        _stderr_boot("runner fetching miners from registry")
        miners = await get_miners_from_registry(NETUID, subtensor=subtensor)
        if not isinstance(miners, dict):
            _stderr_boot(f"runner miner registry invalid type={type(miners).__name__}")
            logger.warning(
                "Miner registry returned unexpected type %s; skipping run",
                type(miners).__name__,
            )
            return
        _stderr_boot(f"runner miners fetched count={len(miners)}")
        logger.info(f"Found {len(miners)} eligible miners from registry: {list(miners.keys())}")
        if not miners:
            _stderr_boot("runner no eligible miners")
            logger.warning("No eligible miners found on-chain.")
            return

        miner_list = list(miners.values())
        random.shuffle(miner_list)
        miner_list = miner_list[: min(MAX_MINERS, len(miner_list))]
        logger.debug(
            "[runner] miners selected=%d (max=%d)",
            len(miner_list), MAX_MINERS,
        )

        if not miner_list:
            logger.info("No miners to process after filtering")
            return

        score_embedder_prewarm_task = asyncio.create_task(asyncio.to_thread(prewarm_score_dialogue_embedder))

        # Define prediction callback for all miners
        from babelbit.utils.predict_engine import call_miner_axon_endpoint
        
        # Capture timeout value from settings before defining callback
        miner_timeout = _coerce_timeout_seconds(
            getattr(settings, "BB_MINER_TIMEOUT_SEC", None),
            default=10.0,
        )
        
        async def prediction_callback(miner: Miner, payload: BBPredictedUtterance, context: str) -> str:
            """
            Callback to get prediction from a single miner.
            Returns the prediction text or empty string on error.
            Exceptions are raised to be handled by the multi-miner function.
            """
            if miner.axon_ip and miner.axon_port:
                started_at = time.perf_counter()
                miner_uid = getattr(miner, "uid", "?")
                miner_hotkey_short = (miner.hotkey[:16] + "...") if getattr(miner, "hotkey", None) else "?"
                step_value = getattr(payload, "step", "?")
                prefix_chars = len((getattr(payload, "prefix", "") or ""))
                _trace_miner_call(
                    f"start uid={miner_uid} hotkey={miner_hotkey_short} endpoint={miner.axon_ip}:{miner.axon_port} step={step_value} prefix_chars={prefix_chars}",
                )
                logger.info(
                    "[runner] miner call start uid=%s hotkey=%s endpoint=%s:%s step=%s prefix_chars=%d",
                    miner_uid,
                    miner_hotkey_short,
                    miner.axon_ip,
                    miner.axon_port,
                    step_value,
                    prefix_chars,
                )
                try:
                    # Call via Axon endpoint with Bittensor protocol
                    result = await call_miner_axon_endpoint(
                        axon_ip=miner.axon_ip,
                        axon_port=miner.axon_port,
                        payload=payload,
                        context_used=context,
                        miner_hotkey=miner.hotkey,
                        timeout=miner_timeout
                    )
                    
                    if result.success and result.utterance:
                        latency = time.perf_counter() - started_at
                        prediction_text = result.utterance.prediction or ""
                        _trace_miner_call(
                            f"success uid={miner_uid} hotkey={miner_hotkey_short} step={step_value} latency={latency:.2f}s prediction_chars={len(prediction_text)}",
                        )
                        logger.info(
                            "[runner] miner call success uid=%s hotkey=%s step=%s latency=%.2fs prediction_chars=%d",
                            miner_uid,
                            miner_hotkey_short,
                            step_value,
                            latency,
                            len(prediction_text),
                        )
                        return result.utterance.prediction
                    else:
                        latency = time.perf_counter() - started_at
                        _trace_miner_call(
                            f"failed uid={miner_uid} hotkey={miner_hotkey_short} step={step_value} latency={latency:.2f}s error={result.error}",
                        )
                        logger.warning(
                            "[runner] miner call failed uid=%s hotkey=%s step=%s latency=%.2fs error=%s",
                            miner_uid,
                            miner_hotkey_short,
                            step_value,
                            latency,
                            result.error,
                        )
                        raise RuntimeError(f"{result.error}")
                except Exception as e:
                    latency = time.perf_counter() - started_at
                    _trace_miner_call(
                        f"exception uid={miner_uid} hotkey={miner_hotkey_short} step={step_value} latency={latency:.2f}s err={type(e).__name__}:{e}",
                    )
                    logger.error(
                        "Miner %s axon error after %.2fs: %s",
                        miner_uid,
                        latency,
                        e,
                    )
                    raise
            miner_uid = getattr(miner, "uid", "?")
            _trace_miner_call(
                f"skipped uid={miner_uid} hotkey={(miner.hotkey[:16] + '...') if getattr(miner, 'hotkey', None) else '?'} reason=no_axon_endpoint",
            )
            logger.warning(
                "[runner] miner call skipped uid=%s hotkey=%s reason=no_axon_endpoint",
                miner_uid,
                (miner.hotkey[:16] + "...") if getattr(miner, "hotkey", None) else "?",
            )
            raise RuntimeError(f"Miner {miner_uid} has no axon endpoint available")
        
        logger.info(f"Starting shared utterance session for {len(miner_list)} miners")
        _stderr_boot(
            "runner predict begin "
            f"miners={len(miner_list)} timeout={miner_timeout:.2f}",
        )
        
        # Get step block modulo from environment (default: 1 block)
        step_block_modulo = int(os.getenv("BB_STEP_BLOCK_MODULO", "0"))
        logger.debug(
            "[runner] session params: timeout=%.2fs step_block_modulo=%d", miner_timeout, step_block_modulo
        )
        
        miner_dialogues = await predict_with_utterance_engine_multi_miner(
            utterance_engine_url=utterance_engine_url,
            miners=miner_list,
            prediction_callback=prediction_callback,
            timeout=miner_timeout,
            continue_after_all_miners_deactivated=True,
            max_prediction_errors=5,
            subtensor=subtensor,
            step_block_modulo=step_block_modulo
        )
        _stderr_boot(
            "runner predict end "
            f"miners_with_dialogues={len(miner_dialogues or {})}",
        )
        try:
            miners_with_dialogues = len(miner_dialogues or {})
            total_dialogues = sum(len(v) for v in (miner_dialogues or {}).values())
            logger.debug(
                "[runner] multi-miner collected: miners_with_dialogues=%d total_dialogues=%d",
                miners_with_dialogues, total_dialogues,
            )
        except Exception:
            pass

        try:
            await score_embedder_prewarm_task
        except Exception as e:
            logger.warning("Score embedder prewarm failed: %s", e)

        total_miners_processed, total_dialogues_processed, all_challenge_scores = await _score_miners_for_challenge(
            challenge_uid=challenge_uid,
            challenge_type="main",
            miner_list=miner_list,
            miner_dialogues=miner_dialogues or {},
            logs_dir=logs_dir,
            scores_dir=scores_dir,
            submission_client=submission_client,
            active_s3_manager=s3_manager,
        )

        if challenge_uid and total_miners_processed > 0:
            overall_mean = sum(all_challenge_scores) / len(all_challenge_scores) if all_challenge_scores else None
            mark_challenge_processed(
                challenge_uid=challenge_uid,
                miner_count=total_miners_processed,
                total_dialogues=total_dialogues_processed,
                mean_score=overall_mean,
                challenge_type="main",
                metadata={
                    "scores_dir": str(scores_dir),
                    "logs_dir": str(logs_dir),
                }
            )
            logger.debug(
                "[runner] challenge processed: uid=%s miners=%d dialogues=%d mean=%s",
                challenge_uid, total_miners_processed, total_dialogues_processed,
                (f"{overall_mean:.4f}" if overall_mean is not None else "N/A"),
            )
            mean_score_str = f"{overall_mean:.4f}" if overall_mean is not None else "N/A"
            logger.info(
                f"Challenge {challenge_uid} completed: {total_miners_processed} miners, "
                f"{total_dialogues_processed} dialogues, mean_score={mean_score_str}"
            )
            _stderr_boot(
                "runner challenge complete "
                f"challenge_uid={challenge_uid} miners={total_miners_processed} "
                f"dialogues={total_dialogues_processed} mean={mean_score_str}",
            )

        if enable_solo_challenge:
            if _DEFER_HTTP_CLIENT_CLOSE:
                solo_phase_task = asyncio.create_task(
                    _run_solo_challenge_phase(
                        utterance_engine_url=utterance_engine_url,
                        miner_list=miner_list,
                        miner_timeout=miner_timeout,
                        challenge_uid=challenge_uid,
                        main_challenge_uid=challenge_uid,
                        logs_dir=logs_dir,
                        scores_dir=scores_dir,
                        submission_client=submission_client,
                        active_s3_manager=s3_manager,
                    )
                )
                _track_background_task(solo_phase_task, label=f"solo:{challenge_uid}")
                logger.info("[Solo Challenge] Scheduled background solo phase for challenge %s", challenge_uid)
            else:
                await _run_solo_challenge_phase(
                    utterance_engine_url=utterance_engine_url,
                    miner_list=miner_list,
                    miner_timeout=miner_timeout,
                    challenge_uid=challenge_uid,
                    main_challenge_uid=challenge_uid,
                    logs_dir=logs_dir,
                    scores_dir=scores_dir,
                    submission_client=submission_client,
                    active_s3_manager=s3_manager,
                )
        else:
            logger.debug("[runner] Solo challenge phase disabled via BB_ENABLE_SOLO_CHALLENGE")
            _stderr_boot("runner solo challenge disabled")
                
    except Exception as e:
        _stderr_boot(f"runner failed: {type(e).__name__}: {e}")
        try:
            traceback.print_exc()
        except Exception:
            pass
        logger.error(f"Runner failed: {type(e).__name__}: {e}", exc_info=True)
    finally:
        _close_http_clients_if_allowed(caller="runner")


async def runner_round2(utterance_engine_url: str | None = None, output_dir: Optional[str] = None, subtensor=None) -> None:
    _enforce_runner_logging_level()
    settings = get_settings()
    NETUID = settings.BABELBIT_NETUID
    MAX_MINERS = int(os.getenv("BB_MAX_MINERS_PER_RUN", "256"))
    utterance_engine_url = utterance_engine_url or os.getenv("BB_UTTERANCE_ENGINE_URL", "http://localhost:8000")
    _stderr_boot(
        "runner arena entry "
        f"utterance_engine_url={utterance_engine_url} "
        f"netuid={NETUID} max_miners={MAX_MINERS}",
    )

    logs_dir = Path(os.getenv("BB_OUTPUT_LOGS_DIR", "logs"))
    scores_dir = Path(output_dir or os.getenv("BB_OUTPUT_SCORES_DIR", "scores"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    s3_enabled = os.getenv("BB_ENABLE_S3_UPLOADS", "0").lower() in {"1", "true", "yes"}
    global s3_manager
    if s3_enabled and s3_manager is None:
        try:
            s3_manager = S3Manager(
                bucket_name=settings.S3_BUCKET_NAME,
                access_key=settings.S3_ACCESS_KEY_ID,
                secret_key=settings.S3_SECRET_ACCESS_KEY.get_secret_value(),
                endpoint_url=settings.S3_ENDPOINT_URL or None,
                region=settings.S3_REGION,
                addressing_style=settings.S3_ADDRESSING_STYLE or "auto",
                signature_version=settings.S3_SIGNATURE_VERSION or "s3v4",
                use_ssl=settings.S3_USE_SSL,
                prefix="",
            )
            logger.info("S3 Manager initialized (uploads enabled)")
        except Exception as e:
            logger.warning("S3 Manager initialization failed; disabling S3 uploads: %s", e)
            s3_manager = None

    submission_client = ValidationSubmissionClient()

    try:
        challenge_uid = await get_current_challenge_uid(utterance_engine_url)
        _stderr_boot(f"runner arena challenge_uid={challenge_uid}")
    except Exception as e:
        _stderr_boot(f"runner arena challenge_uid fetch failed: {type(e).__name__}: {e}")
        logger.warning("Could not get arena challenge ID: %s", e)
        return

    if challenge_uid:
        already_processed = get_processed_miners_for_challenge(
            str(scores_dir),
            challenge_uid,
            challenge_type="arena",
        )
        if already_processed:
            _stderr_boot(
                f"runner arena skip already_processed challenge_uid={challenge_uid} "
                f"miners={len(already_processed)}",
            )
            logger.info(
                "Arena challenge %s already has %d scored miners; skipping.",
                challenge_uid,
                len(already_processed),
            )
            return

    try:
        _stderr_boot("runner arena resolving managed routes")
        arena_miners, routes_by_hotkey = await resolve_round2_routes(
            netuid=NETUID,
            subtensor=subtensor,
        )
        if not arena_miners:
            _stderr_boot("runner arena no eligible managed miners")
            logger.warning("No eligible managed miners found for arena.")
            return

        random.shuffle(arena_miners)
        arena_miners = arena_miners[: min(MAX_MINERS, len(arena_miners))]
        routes_by_hotkey = {m.hotkey: routes_by_hotkey[m.hotkey] for m in arena_miners if m.hotkey in routes_by_hotkey}

        arena_timeout = _coerce_timeout_seconds(
            getattr(settings, "BB_ARENA_MINER_TIMEOUT_SEC", None),
            default=_coerce_timeout_seconds(getattr(settings, "BB_MINER_TIMEOUT_SEC", None), default=10.0),
        )
        arena_startup_timeout = _resolve_arena_startup_timeout()
        score_embedder_prewarm_task = asyncio.create_task(asyncio.to_thread(prewarm_score_dialogue_embedder))
        callback = _build_round2_prediction_callback(
            routes_by_hotkey=routes_by_hotkey,
            miner_timeout=arena_timeout,
            startup_timeout=arena_startup_timeout,
            startup_request_timeout=_resolve_arena_startup_request_timeout(),
        )

        step_block_modulo = int(os.getenv("BB_STEP_BLOCK_MODULO", "0"))
        _stderr_boot(
            "runner arena predict begin "
            f"miners={len(arena_miners)} timeout={arena_timeout:.2f}",
        )

        miner_dialogues = await predict_with_utterance_engine_multi_miner(
            utterance_engine_url=utterance_engine_url,
            miners=arena_miners,
            prediction_callback=callback,
            timeout=arena_timeout,
            first_step_timeout=arena_startup_timeout,
            continue_after_all_miners_deactivated=True,
            max_prediction_errors=5,
            subtensor=subtensor,
            step_block_modulo=step_block_modulo,
        )
        _stderr_boot(
            "runner arena predict end "
            f"miners_with_dialogues={len(miner_dialogues or {})}",
        )

        try:
            await score_embedder_prewarm_task
        except Exception as e:
            logger.warning("Score embedder prewarm failed: %s", e)

        total_miners_processed, total_dialogues_processed, all_challenge_scores = await _score_miners_for_challenge(
            challenge_uid=challenge_uid,
            challenge_type="arena",
            miner_list=arena_miners,
            miner_dialogues=miner_dialogues or {},
            logs_dir=logs_dir,
            scores_dir=scores_dir,
            submission_client=submission_client,
            active_s3_manager=s3_manager,
        )

        if challenge_uid and total_miners_processed > 0:
            overall_mean = sum(all_challenge_scores) / len(all_challenge_scores) if all_challenge_scores else None
            mark_challenge_processed(
                challenge_uid=challenge_uid,
                challenge_type="arena",
                miner_count=total_miners_processed,
                total_dialogues=total_dialogues_processed,
                mean_score=overall_mean,
                metadata={
                    "scores_dir": str(scores_dir),
                    "logs_dir": str(logs_dir),
                    "route_source": "list_arena_miners",
                },
            )
            mean_score_str = f"{overall_mean:.4f}" if overall_mean is not None else "N/A"
            _stderr_boot(
                "runner arena challenge complete "
                f"challenge_uid={challenge_uid} miners={total_miners_processed} "
                f"dialogues={total_dialogues_processed} mean={mean_score_str}",
            )
            logger.info(
                "Arena challenge %s completed: %d miners, %d dialogues, mean_score=%s",
                challenge_uid,
                total_miners_processed,
                total_dialogues_processed,
                mean_score_str,
            )
    except Exception as e:
        _stderr_boot(f"runner arena failed: {type(e).__name__}: {e}")
        logger.error("Arena runner failed: %s: %s", type(e).__name__, e, exc_info=True)
    finally:
        _close_http_clients_if_allowed(caller="runner arena")


async def runner_loop():
    """Runs `runner()` every N blocks (default: 2160)."""
    global _DEFER_HTTP_CLIENT_CLOSE
    _enforce_runner_logging_level()
    settings = get_settings()
    TEMPO = int(os.getenv("BABELBIT_RUNNER_TEMPO", "2160"))
    MAX_SUBTENSOR_RETRIES = int(os.getenv("BABELBIT_MAX_SUBTENSOR_RETRIES", "5"))
    run_on_startup = _env_flag(
        "BB_RUNNER_ON_STARTUP",
        default=getattr(settings, "BB_RUNNER_ON_STARTUP", False),
    )
    arena_enabled = _env_flag(
        "BB_ENABLE_ARENA_CHALLENGE",
        default=getattr(settings, "BB_ENABLE_ARENA_CHALLENGE", False),
    )
    arena_run_on_startup = _env_flag(
        "BB_ARENA_RUN_ON_STARTUP",
        default=getattr(settings, "BB_ARENA_RUN_ON_STARTUP", False),
    )
    try:
        arena_cadence_blocks = int(
            os.getenv(
                "BB_ARENA_CADENCE_BLOCKS",
                str(getattr(settings, "BB_ARENA_CADENCE_BLOCKS", 300)),
            ),
        )
    except Exception:
        arena_cadence_blocks = 300
    if arena_cadence_blocks <= 0:
        arena_cadence_blocks = 1

    st = None
    last_block = -1
    last_arena_block = -1
    last_successful_run = 0
    consecutive_failures = 0
    run_count = 0
    
    # Initialize utterance engine authentication on startup
    utterance_engine_url = os.getenv("BB_UTTERANCE_ENGINE_URL", "https://api.babelbit.ai")
    wallet_name = os.getenv("BITTENSOR_WALLET_COLD", "default")
    hotkey_name = os.getenv("BITTENSOR_WALLET_HOT", "default")
    
    init_utterance_auth(utterance_engine_url, wallet_name, hotkey_name)
    startup_context = _format_runner_startup_context(version=getattr(settings, "BABELBIT_VERSION", None))
    _stderr_boot(
        "runner_loop start "
        f"BB_RUNNER_ON_STARTUP={os.getenv('BB_RUNNER_ON_STARTUP')} "
        f"resolved={run_on_startup} "
        f"BB_ENABLE_ARENA_CHALLENGE={os.getenv('BB_ENABLE_ARENA_CHALLENGE')} "
        f"arena_enabled={arena_enabled} "
        f"BB_ARENA_CADENCE_BLOCKS={arena_cadence_blocks} "
        f"BB_ARENA_RUN_ON_STARTUP={arena_run_on_startup} "
        f"BB_UTTERANCE_ENGINE_URL={utterance_engine_url} "
        f"BABELBIT_RUNNER_TEMPO={TEMPO} "
        f"{startup_context}",
    )
    logger.info("[RunnerLoop] %s", startup_context)
    logger.info(
        "[RunnerLoop] Startup run enabled=%s (BB_RUNNER_ON_STARTUP=%s); arena enabled=%s cadence_blocks=%s startup=%s",
        run_on_startup,
        os.getenv("BB_RUNNER_ON_STARTUP"),
        arena_enabled,
        arena_cadence_blocks,
        arena_run_on_startup,
    )
    
    # Authenticate with retry logic on startup
    try:
        logger.info("[RunnerLoop] Authenticating with utterance engine on startup...")
        _stderr_boot("auth startup begin")
        await authenticate_utterance_engine()
        logger.info("[RunnerLoop] Successfully authenticated with utterance engine")
        _stderr_boot("auth startup success")
    except Exception as e:
        _stderr_boot(f"auth startup failed: {type(e).__name__}: {e}")
        try:
            traceback.print_exc()
        except Exception:
            pass
        logger.error(
            "[RunnerLoop] Failed to authenticate with utterance engine on startup: %s",
            e,
            exc_info=True,
        )
        logger.error("[RunnerLoop] Cannot proceed without authentication. Exiting.")
        return

    try:
        _DEFER_HTTP_CLIENT_CLOSE = True
        while True:
            try:
                if st is None:
                    logger.info(
                        "[RunnerLoop] Attempting to connect to subtensor gateway "
                        "(attempt %s/%s)...",
                        consecutive_failures + 1,
                        MAX_SUBTENSOR_RETRIES,
                    )
                    _stderr_boot(
                        f"subtensor connect attempt={consecutive_failures + 1}/{MAX_SUBTENSOR_RETRIES}",
                    )
                    try:
                        await reset_subtensor()  # Clear any stale cached connection
                        st = await asyncio.wait_for(get_subtensor(), timeout=60)
                        logger.info("[RunnerLoop] Successfully created subtensor connection")
                        _stderr_boot("subtensor connected")
                        
                        # Test the connection by fetching a block
                        test_block = await asyncio.wait_for(st.get_current_block(), timeout=30)
                        logger.info(f"[RunnerLoop] Connection verified at block {test_block}")
                        
                    except asyncio.TimeoutError as te:
                        st = None  # Clear invalid connection
                        await reset_subtensor()  # Also clear the global cache
                        raise TimeoutError(f"Subtensor initialization timed out: {te}")
                    except Exception as e:
                        st = None  # Clear invalid connection
                        await reset_subtensor()  # Also clear the global cache
                        logger.error(f"[RunnerLoop] Subtensor connection failed: {type(e).__name__}: {e}", exc_info=True)
                        raise

                # Try to get current block for tempo-based scheduling
                should_run_main = False
                should_run_arena = False
                block = None
                use_time_fallback = False
                
                try:
                    block = await asyncio.wait_for(st.get_current_block(), timeout=30)
                    logger.debug(f"[RunnerLoop] Current block: {block}")

                    # Refresh authentication 100 blocks before each run (or less if TEMPO < 100)
                    auth_refresh_offset = TEMPO - min(100, max(1, TEMPO - 1))
                    if block % TEMPO == auth_refresh_offset:
                        try:
                            logger.info(f"[RunnerLoop] Refreshing authentication at block {block} ({TEMPO - auth_refresh_offset} blocks before next run)")
                            await authenticate_utterance_engine()
                            logger.info("[RunnerLoop] Authentication refresh successful")
                        except Exception as auth_e:
                            logger.error(f"[RunnerLoop] Authentication refresh failed: {auth_e}")
                            # Don't stop the loop, but this will cause issues for the next runner() call
                    
                    # Main challenge trigger
                    if (run_on_startup and last_successful_run == 0) or (block > last_block and block % TEMPO == 0):
                        should_run_main = True
                        logger.info(f"[RunnerLoop] Triggering runner at block {block}")

                    # Arena challenge trigger on separate cadence
                    if arena_enabled:
                        if (arena_run_on_startup and last_arena_block < 0) or (
                            block > last_arena_block and block % arena_cadence_blocks == 0
                        ):
                            should_run_arena = True
                            logger.info(
                                "[RunnerLoop] Triggering arena runner at block %s (cadence=%s)",
                                block,
                                arena_cadence_blocks,
                            )

                    if not should_run_main and not should_run_arena:
                        # Wait for next block with timeout
                        try:
                            await asyncio.wait_for(st.wait_for_block(), timeout=60)
                        except asyncio.TimeoutError:
                            # Don't reset on timeout - just log and retry
                            logger.debug("[RunnerLoop] wait_for_block timeout (60s) — retrying")
                            await asyncio.sleep(5)
                        except Exception as e:
                            logger.warning(f"[RunnerLoop] wait_for_block error: {e}")
                            st = None
                            await reset_subtensor()
                        continue
                        
                except Exception as e:
                    # Block fetch failed - fall back to time-based scheduling
                    logger.warning(f"[RunnerLoop] Block fetch failed: {type(e).__name__}: {e}")
                    st = None  # Force reconnection on next iteration
                    await reset_subtensor()  # Clear the global cached connection
                    
                    if run_on_startup and last_successful_run == 0:
                        should_run_main = True
                        use_time_fallback = True
                        logger.warning(
                            "[RunnerLoop] Startup run is enabled and block fetch failed; "
                            "running validation via fallback path."
                        )
                        block = None
                    else:
                        time_elapsed = time.time() - last_successful_run
                        expected_interval = TEMPO * 12  # TEMPO blocks * ~12 seconds per block
                    
                        if last_successful_run > 0 and time_elapsed >= expected_interval:
                            should_run_main = True
                            use_time_fallback = True
                            logger.warning(
                                f"[RunnerLoop] Blockchain unreachable. Using time-based fallback: "
                                f"elapsed={time_elapsed:.0f}s, expected={expected_interval:.0f}s"
                            )
                        else:
                            # Not enough time has passed, or first run - skip and let retry logic handle it
                            if last_successful_run == 0:
                                logger.info("[RunnerLoop] First run - will retry connection")
                            else:
                                logger.info(f"[RunnerLoop] Only {time_elapsed:.0f}s elapsed (need {expected_interval:.0f}s), will retry")
                            raise  # Re-raise to trigger retry logic
                        
                if should_run_main:
                    if use_time_fallback:
                        logger.info("[RunnerLoop] Running validation via time-based fallback (blockchain unreachable)")
                    _stderr_boot(
                        "trigger runner call "
                        f"use_time_fallback={use_time_fallback} "
                        f"block={block}",
                    )
                    
                    await runner(subtensor=st if st is not None else None)
                    _stderr_boot("runner call completed")
                    
                    if block is not None:
                        last_block = block
                    last_successful_run = time.time()
                    consecutive_failures = 0  # Reset after successful validation cycle
                    run_count += 1
                    logger.info(f"[RunnerLoop] Completed runner cycle #{run_count}")
                    
                    if run_count >= 10:
                        logger.info("[RunnerLoop] Reached 10 successful runs, resetting subtensor connection to free resources.")
                        st = None
                        await reset_subtensor()
                        run_count = 0
                        gc.collect()

                if should_run_arena and block is not None:
                    _stderr_boot(
                        "trigger arena runner call "
                        f"block={block} cadence_blocks={arena_cadence_blocks}",
                    )
                    await runner_round2(subtensor=st if st is not None else None)
                    _stderr_boot("arena runner call completed")
                    last_arena_block = block

            except asyncio.CancelledError:
                _stderr_boot(
                    "runner_loop received CancelledError; exiting loop",
                )
                break
            except Exception as e:
                _stderr_boot(
                    f"runner_loop iteration error: {type(e).__name__}: {e}",
                )
                consecutive_failures += 1
                logger.warning(
                    f"[RunnerLoop] Error (attempt {consecutive_failures}/{MAX_SUBTENSOR_RETRIES}): {type(e).__name__}: {e}"
                )
                
                if consecutive_failures >= MAX_SUBTENSOR_RETRIES:
                    logger.error(
                        f"[RunnerLoop] Max retries ({MAX_SUBTENSOR_RETRIES}) exceeded. "
                        f"gateway={settings.SUBTENSOR_GATEWAY_URL}"
                    )
                    logger.error(
                        "[RunnerLoop] Unable to connect to subtensor gateway. "
                        "Sleeping for 5 minutes before retry cycle..."
                    )
                    consecutive_failures = 0  # Reset counter
                    st = None
                    await asyncio.sleep(300)  # Sleep 5 minutes before trying again
                else:
                    logger.info(f"[RunnerLoop] Retrying in 120 seconds...")
                    st = None
                    await asyncio.sleep(120)
    finally:
        # Ensure cleanup on exit
        _DEFER_HTTP_CLIENT_CLOSE = False
        _stderr_boot("runner_loop shutdown: closing HTTP clients")
        logger.info("[RunnerLoop] Shutting down, cleaning up resources...")
        _shutdown_scoring_process_pool()
        await reset_subtensor()
        close_http_clients()
