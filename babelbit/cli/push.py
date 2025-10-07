import time
from pathlib import Path
from logging import getLogger

from babelbit.utils.settings import get_settings
from babelbit.utils.chutes_helpers import deploy_to_chutes, share_chute, warmup_chute, get_chute_slug_and_id

from babelbit.utils.huggingface_helpers import (
    create_update_or_verify_huggingface_repo,
)
from babelbit.utils.bittensor_helpers import on_chain_commit

logger = getLogger(__name__)


async def push_ml_model(
    ml_model_path: Path | None,
    hf_revision: str | None,
    skip_chutes_deploy: bool,
    skip_bittensor_commit: bool,
    skip_warmup: bool = False,
    # warmup_video_url: str | None,
):
    start_time = time.time()
    logger.info(
        "Starting push pipeline: ml_model_path=%s, hf_revision=%s, skip_chutes_deploy=%s, skip_bittensor_commit=%s, skip_warmup=%s",
        ml_model_path, hf_revision, skip_chutes_deploy, skip_bittensor_commit, skip_warmup
    )

    # Step 1: HF repo
    logger.info("[1/5] Creating/updating HF repo with model_path=%s, hf_revision=%s", ml_model_path, hf_revision)
    hf_revision = await create_update_or_verify_huggingface_repo(
        model_path=ml_model_path, hf_revision=hf_revision
    )
    logger.info("✅ HF repo ready with revision: %s", hf_revision)

    # Step 2: Chutes deployment
    if skip_chutes_deploy:
        logger.info("[2/5] Skipping Chutes deployment (skip_chutes_deploy=True)")
        logger.info("🔍 Looking up existing chute for revision: %s", hf_revision)
        try:
            chute_slug, chute_id = await get_chute_slug_and_id(revision=hf_revision)
            if chute_id:
                logger.info("✅ Found existing chute: chute_id=%s, slug=%s", chute_id, chute_slug)
            else:
                logger.warning("⚠️ No existing chute found for revision %s", hf_revision)
                chute_id, chute_slug = None, None
        except Exception as e:
            logger.warning("⚠️ Failed to lookup existing chute: %s", e)
            chute_id, chute_slug = None, None
    else:
        logger.info("[2/5] Deploying to Chutes with revision=%s", hf_revision)
        try:
            chute_id, chute_slug = await deploy_to_chutes(
                revision=hf_revision,
                skip=skip_chutes_deploy,
            )
            if chute_id:
                logger.info("✅ Chutes deployment complete: chute_id=%s, slug=%s", chute_id, chute_slug)
            else:
                logger.warning("⚠️ Chutes deployment returned no chute_id (possibly due to existing image)")
        except Exception as e:
            logger.error("❌ Chutes deployment failed with error: %s", e)
            chute_id, chute_slug = None, None

    if chute_id:
        # Step 3: Share chute
        logger.info("[3/5] Sharing chute: chute_id=%s", chute_id)
        await share_chute(chute_id=chute_id)
        logger.info("✅ Chute shared successfully")

        # Step 4: On-chain commit
        if skip_bittensor_commit:
            logger.info("[4/5] Skipping Bittensor commit (skip_bittensor_commit=True)")
        else:
            logger.info("[4/5] Committing to Bittensor: revision=%s, chute_id=%s, slug=%s", 
                       hf_revision, chute_id, chute_slug)
            await on_chain_commit(
                skip=skip_bittensor_commit,
                revision=hf_revision,
                chute_id=chute_id,
                chute_slug=chute_slug,
            )
            logger.info("✅ On-chain commit completed")

        # Step 5: Warmup
        if skip_warmup:
            logger.info("[5/5] Skipping chute warmup (skip_warmup=True)")
            logger.info("💡 Chute will warm up automatically on first request")
        else:
            logger.info("[5/5] Warming up chute: chute_id=%s", chute_id)
            logger.info("⏳ This may take several minutes as the chute starts up...")
            try:
                await warmup_chute(chute_id=chute_id)
                logger.info("✅ Chute warmup completed successfully")
            except Exception as e:
                logger.warning("⚠️ Chute warmup failed or timed out: %s", e)
                logger.info("💡 The chute may still be starting up - it should be accessible shortly")
                logger.info("💡 You can check the chute status manually or retry warmup later")
    else:
        logger.info("Skipping share/commit/warmup steps (no chute_id available)")

    elapsed = time.time() - start_time
    warmup_status = "Skipped" if skip_warmup else "✅" if chute_id else "N/A"
    logger.info("🎉 Push pipeline completed successfully in %.1fs - HF: %s, Chute: %s/%s, On-chain: %s, Warmup: %s", 
               elapsed, hf_revision, chute_id or "None", chute_slug or "None", 
               "✅" if not skip_bittensor_commit and chute_id else ("Skipped" if skip_bittensor_commit else "N/A"),
               warmup_status)
