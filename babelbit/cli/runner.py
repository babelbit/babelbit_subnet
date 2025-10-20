from typing import List, Optional
from logging import getLogger
import os
import tempfile
from pathlib import Path
import json
import random
import asyncio
from datetime import datetime
import time

from babelbit.utils.s3_manager import S3Manager
from babelbit.utils.settings import get_settings
# from babelbit.utils.challenges import get_challenge_from_scorevision
from babelbit.utils.predict_utterances import get_current_challenge_uid, predict_with_utterance_engine
from babelbit.utils.utterance_auth import init_utterance_auth, authenticate_utterance_engine
# from babelbit.utils.evaluate import evaluate_using_vlms, post_vlm_ranking
# from babelbit.utils.cloudflare_helpers import emit_shard
from babelbit.utils.async_clients import close_http_clients
# from babelbit.vlm_pipeline.vlm_annotator import (
#     generate_annotations_for_select_frames,
# )
from babelbit.utils.miner_registry import get_miners_from_registry, Miner
from babelbit.utils.bittensor_helpers import get_subtensor
from babelbit.chute_template.schemas import BBPredictedUtterance
from babelbit.utils.file_handling import (
    get_processed_miners_for_challenge,
    save_dialogue_score_file,
    save_challenge_summary_file,
)
from datetime import timezone
from babelbit.utils.db_pool import (
    db_pool,
    insert_json_staging,
    insert_scoring_staging,
    insert_scoring_submissions_bulk,
)
# Scoring policy: Runner performs per-dialogue scoring using score_dialogue.score_jsonl
# immediately after writing the raw dialogue JSONL, then persists both dialogue score JSONs
# and a challenge summary JSON per miner. This keeps downstream evaluation simple while
# still enabling external rescoring if needed (raw logs retained in logs_dir).
try:
    from babelbit.test_scripts.score_dialogue import score_jsonl  # type: ignore
except Exception:
    score_jsonl = None  # Will log warning when attempting to score

logger = getLogger(__name__)

s3_manager: Optional[S3Manager] = None
settings = get_settings()

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


async def runner(slug: str | None = None, utterance_engine_url: str | None = None, output_dir: Optional[str] = None) -> None:
    settings = get_settings()
    NETUID = settings.BABELBIT_NETUID
    MAX_MINERS = int(os.getenv("BB_MAX_MINERS_PER_RUN", "60"))
    utterance_engine_url = utterance_engine_url or os.getenv("BB_UTTERANCE_ENGINE_URL", "http://localhost:8000")
    
    # Initialize utterance engine authentication
    wallet_name = os.getenv("BITTENSOR_WALLET_COLD", "default")
    hotkey_name = os.getenv("BITTENSOR_WALLET_HOT", "default")
    
    init_utterance_auth(utterance_engine_url, wallet_name, hotkey_name)
    
    # Authenticate with utterance engine
    try:
        await authenticate_utterance_engine()
        logger.info("Successfully authenticated with utterance engine")
    except Exception as e:
        logger.error(f"Failed to authenticate with utterance engine: {e}")
        return
    # Determine directories:
    #   Raw logs:   ./logs (override with BB_OUTPUT_LOGS_DIR)
    #   Scores:     ./scores (override with BB_OUTPUT_SCORES_DIR or output_dir argument) produced after scoring
    #   output_dir argument retained for backward compatibility
    logs_dir = Path(os.getenv("BB_OUTPUT_LOGS_DIR", "logs"))
    scores_dir = Path(output_dir or os.getenv("BB_OUTPUT_SCORES_DIR", "scores"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Database write configuration (opt-in so tests / local runs without PG don't fail)
    db_enabled = os.getenv("BB_ENABLE_DB_WRITES", "0").lower() in {"1", "true", "yes"}
    db_ready = False
    if db_enabled:
        try:
            await db_pool.init()
            db_ready = True
            logger.info("DB pool initialized (writes enabled)")
        except Exception as e:
            logger.warning("DB initialization failed; disabling DB writes: %s", e)
            db_ready = False

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

    try:
        challenge_uid = await get_current_challenge_uid(utterance_engine_url)
    except Exception as e:
        logger.warning(f"Could not get current challenge ID: {e}")
        return

    try:
        miners = await get_miners_from_registry(NETUID)
        logger.info(f"Found {len(miners)} eligible miners from registry: {list(miners.keys())}")
        if not miners:
            logger.warning("No eligible miners found on-chain.")
            return

        miner_list = list(miners.values())
        random.shuffle(miner_list)
        miner_list = miner_list[: min(MAX_MINERS, len(miner_list))]

        # TODO: use db to filter processed miners
        # (for now, just use existing score files in scores_dir)
        # Filter out miners that already have results for this challenge
        if challenge_uid:
            already_processed = get_processed_miners_for_challenge(str(scores_dir), challenge_uid)
            if already_processed:
                before_count = len(miner_list)
                miner_list = [m for m in miner_list if (getattr(m, 'uid', None), getattr(m, 'hotkey', None)) not in already_processed]
                after_count = len(miner_list)
                skipped = before_count - after_count
                if skipped:
                    logger.info(f"Skipping {skipped} miners already processed for challenge {challenge_uid}")
            else:
                logger.info(f"No previously processed miners detected for challenge {challenge_uid}")

        for m in miner_list:
            try:
                # No per-miner ChallengeLogger usage when just emitting raw logs
                challenge_logger = None
                
                # Get dialogues from utterance engine and predict turn by turn using the miner's chute
                dialogues = await predict_with_utterance_engine(
                    utterance_engine_url=utterance_engine_url,
                    chute_slug=m.slug,
                    challenge_logger=None,
                    timeout=settings.CHUTES_TIMEOUT_SEC
                )

                dialogue_scores: List[float] = []
                dialogue_uids: List[str] = []

                # Emit raw JSONL events per dialogue then score
                for dialogue_uid, utterance_steps in dialogues.items():
                    dialogue_uids.append(dialogue_uid)
                    logger.info(f"Miner {m.slug} produced {len(utterance_steps)} utterance steps in dialogue {dialogue_uid}")
                    complete_utterances = group_steps_into_utterances(utterance_steps)
                    logger.info(f"Dialogue {dialogue_uid} contains {len(complete_utterances)} complete utterances")
                    events_path = logs_dir / f"dialogue_run_{challenge_uid or 'unknown'}_miner_{m.uid}__hk_{m.hotkey}__dlg_{dialogue_uid}.jsonl"
                    with events_path.open("w", encoding="utf-8") as jf:
                        for utt_index, utt_steps in enumerate(complete_utterances):
                            for step_idx, step_obj in enumerate(utt_steps):
                                jf.write(json.dumps({
                                    "event": "predicted",
                                    "utterance_index": utt_index,
                                    "step": step_idx,
                                    "prediction": getattr(step_obj, 'prediction', '') or ''
                                }) + "\n")
                            gt = getattr(utt_steps[-1], 'ground_truth', '') or ''
                            jf.write(json.dumps({
                                "event": "utterance_complete",
                                "utterance_index": utt_index,
                                "ground_truth": gt
                            }) + "\n")
                    logger.info(f"[runner] Wrote raw dialogue log: {events_path}")
                    # Optionally, write raw events JSON to S3
                    if s3_manager:
                        # Upload to bucket/logs/ directory structure
                        s3_log_path = f"{settings.S3_LOG_DIR}/logs/{events_path.name}"
                        s3_manager.upload_file(str(events_path), s3_log_path)
                        logger.info(f"Uploaded raw dialogue log to S3: s3://{s3_manager.bucket_name}/{s3_log_path}")

                    raw_events: list[dict] = []
                    if db_ready:
                        # Re-read the file we just wrote (small size expected) to capture raw events JSON
                        try:
                            with events_path.open('r', encoding='utf-8') as rf:
                                for line in rf:
                                    line=line.strip()
                                    if line:
                                        try:
                                            raw_events.append(json.loads(line))
                                        except Exception:
                                            continue
                        except Exception as e:
                            logger.warning("Failed reading raw events for DB staging (%s): %s", events_path, e)

                        # Insert raw log into generic JSON staging table
                        if raw_events:
                            try:
                                now_utc = datetime.now(timezone.utc)
                                await insert_json_staging(
                                    file_content={
                                        "challenge_uid": challenge_uid,
                                        "miner_uid": getattr(m,'uid',None),
                                        "dialogue_uid": dialogue_uid,
                                        "events": raw_events,
                                    },
                                    file_path=str(events_path),
                                    json_created_at=now_utc,
                                )
                            except Exception as e:
                                logger.warning("Failed to insert raw log staging row: %s", e)

                    if score_jsonl is None:
                        logger.warning("score_jsonl unavailable; skipping scoring for dialogue %s", dialogue_uid)
                        continue
                    try:
                        scored_doc = score_jsonl(events_path)
                        # augment metadata
                        scored_doc.update({
                            "challenge_uid": challenge_uid,
                            "miner_uid": getattr(m, 'uid', None),
                            "miner_hotkey": getattr(m, 'hotkey', None),
                            "dialogue_uid": dialogue_uid,
                        })
                        avg_u = float(scored_doc.get("dialogue_summary", {}).get("average_U_best_early", 0.0))
                        dialogue_scores.append(avg_u)
                        score_path = save_dialogue_score_file(scored_doc, output_dir=str(scores_dir))
                        logger.info(f"[runner] Scored dialogue {dialogue_uid} U={avg_u:.4f}")
                        if s3_manager:
                            # Upload to bucket/submissions/ directory structure
                            s3_sub_path = f"submissions/{Path(score_path).name}"
                            s3_manager.upload_file(str(score_path), s3_sub_path)
                            logger.info(f"Uploaded dialogue score to S3: s3://{s3_manager.bucket_name}/{s3_sub_path}")
                        if db_ready:
                            # Insert scoring staging doc
                            try:
                                now_utc = datetime.now(timezone.utc)
                                scoring_staging_id = await insert_scoring_staging(
                                    file_content=scored_doc,
                                    file_path=str(score_path),
                                    json_created_at=now_utc,
                                )
                            except Exception as e:
                                logger.warning("Failed to insert scoring_staging row: %s", e)
                                scoring_staging_id = None
                            # Prepare per-utterance rows -> scoring_submissions_bulk
                            if scoring_staging_id is not None:
                                try:
                                    rows = []
                                    dialogue_average = scored_doc.get("dialogue_summary", {}).get("average_U_best_early")
                                    now_ts = datetime.now(timezone.utc)
                                    for utt in scored_doc.get("utterances", []):
                                        rows.append({
                                            "scoring_staging_id": scoring_staging_id,
                                            "challenge_uid": challenge_uid,
                                            "dialogue_uid": dialogue_uid,
                                            "miner_uid": getattr(m, 'uid', None),
                                            "miner_hotkey": getattr(m, 'hotkey', None),
                                            "utterance_number": utt.get("utterance_number"),
                                            "ground_truth": utt.get("ground_truth"),
                                            "best_step": utt.get("best_step"),
                                            "u_best": utt.get("U_best"),
                                            "total_steps": utt.get("total_steps"),
                                            "average_u_best_early": dialogue_average,
                                            "json_created_at": now_ts,
                                            "staging_inserted_at": now_ts,
                                        })
                                    await insert_scoring_submissions_bulk(rows)
                                except Exception as e:
                                    logger.warning("Failed bulk insert scoring_submissions: %s", e)
                    except Exception as e:
                        logger.warning("Failed scoring dialogue %s: %s", dialogue_uid, e)

                # Miner-level challenge summary
                if dialogue_scores and dialogue_uids:
                    try:
                        summary = {
                            "challenge_uid": challenge_uid,
                            "miner_uid": getattr(m, 'uid', None),
                            "miner_hotkey": getattr(m, 'hotkey', None),
                            "dialogues": [
                                {"dialogue_uid": duid, "dialogue_average_u_best_early": ds, "dialogue_index": idx}
                                for idx, (duid, ds) in enumerate(zip(dialogue_uids, dialogue_scores))
                            ],
                            "challenge_mean_U": (sum(dialogue_scores) / len(dialogue_scores)) if dialogue_scores else None,
                        }
                        summary_path = save_challenge_summary_file(summary, output_dir=str(scores_dir))
                        if s3_manager:
                            # Upload to bucket/submissions/ directory structure
                            s3_sub_path = f"submissions/{Path(summary_path).name}"
                            s3_manager.upload_file(str(summary_path), s3_sub_path)
                            logger.info(f"Uploaded challenge summary to S3: s3://{s3_manager.bucket_name}/{s3_sub_path}")
                        # Optional: store summary JSON in scoring staging as well
                        if db_ready:
                            try:
                                now_utc = datetime.now(timezone.utc)
                                await insert_scoring_staging(
                                    file_content=summary,
                                    file_path=str(summary_path),
                                    json_created_at=now_utc,
                                )
                            except Exception as e:
                                logger.warning("Failed to insert challenge summary scoring_staging: %s", e)
                    except Exception as e:
                        logger.warning("Failed to save challenge summary for miner %s: %s", getattr(m,'uid', '?'), e)

            except Exception as e:
                logger.warning(
                    "Miner uid=%s slug=%s failed: %s",
                    getattr(m, "uid", "?"),
                    getattr(m, "slug", "?"),
                    e,
                )
                # Close challenge logger if it exists
                if 'challenge_logger' in locals() and challenge_logger:
                    challenge_logger.close()
                continue
    except Exception as e:
        logger.error(e)
    finally:
        close_http_clients()
    # Outputs persisted separately: raw logs in logs_dir, scores in scores_dir


async def runner_loop():
    """Runs `runner()` every N blocks (default: 300)."""
    settings = get_settings()
    TEMPO = int(os.getenv("BABELBIT_TEMPO", "300"))

    st = None
    last_block = -1

    while True:
        try:
            if st is None:
                st = await get_subtensor()

            block = await st.get_current_block()

            if block <= last_block or block % TEMPO != 0:
                await st.wait_for_block()
                continue

            logger.info(f"[RunnerLoop] Triggering runner at block {block}")
            await runner()

            last_block = block

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"[RunnerLoop] Error: {e}; retrying…")
            st = None
            await asyncio.sleep(120)
