from logging import getLogger
from json import load, dumps
from base64 import b64decode
from traceback import print_exc
import asyncio
import os

import click
from substrateinterface import Keypair
from bittensor import wallet, async_subtensor

from babelbit.utils.settings import get_settings
from babelbit.utils.huggingface_helpers import get_huggingface_repo_name

logger = getLogger(__name__)

_SUBTENSOR = None


def load_hotkey_keypair(wallet_name: str, hotkey_name: str) -> Keypair:
    settings = get_settings()

    file_path = settings.BITTENSOR_WALLET_PATH
    try:
        with open(file_path, "r") as file:
            keypair_data = load(file)
        seed = keypair_data["secretSeed"]
        keypair = Keypair.create_from_seed(seed)
        logger.info(f"Loaded keypair from {file_path}")
        return keypair
    except Exception as e:
        raise ValueError(f"Failed to load keypair: {str(e)}")


async def get_subtensor():
    """Connexion (et fallback) au subtensor — cache global."""
    global _SUBTENSOR
    settings = get_settings()

    if _SUBTENSOR is None:
        logger.info("🔗 Connecting to Bittensor subtensor: %s", settings.BITTENSOR_SUBTENSOR_ENDPOINT)
        _SUBTENSOR = async_subtensor(settings.BITTENSOR_SUBTENSOR_ENDPOINT)
        try:
            await _SUBTENSOR.initialize()
            logger.info("✅ Bittensor connection established: %s", settings.BITTENSOR_SUBTENSOR_ENDPOINT)
        except Exception as e:
            logger.error("❌ Primary Bittensor connection failed (%s): %s", settings.BITTENSOR_SUBTENSOR_ENDPOINT, str(e))
            logger.info("� Troubleshooting: If using local subtensor, ensure it's running and accessible at %s", settings.BITTENSOR_SUBTENSOR_ENDPOINT)
            logger.info("💡 SSL errors often indicate protocol mismatch - check if your local node expects wss:// or ws://")
            logger.info("�🔄 Attempting fallback connection: %s", settings.BITTENSOR_SUBTENSOR_FALLBACK)
            _SUBTENSOR = async_subtensor(settings.BITTENSOR_SUBTENSOR_FALLBACK)
            await _SUBTENSOR.initialize()
            logger.info("✅ Fallback Bittensor connection established: %s", settings.BITTENSOR_SUBTENSOR_FALLBACK)
    return _SUBTENSOR


async def on_chain_commit(
    skip: bool, revision: str, chute_id: str, chute_slug: str | None
) -> None:
    # Try real on-chain; fallback to logging if bittensor not available
    settings = get_settings()
    repo_name = get_huggingface_repo_name()
    w = wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )
    payload = {
        "model": repo_name,
        "revision": revision,
        "chute_id": chute_id,
        "slug": chute_slug,
        "hotkey": w.hotkey.ss58_address,
    }
    logger.info("📋 Commit payload: %s", payload)
    try:
        if skip:
            logger.info("⏭️ On-chain commit skipped (skip=True)")
            logger.info("📄 Payload that would have been committed: %s", payload)
            return

        logger.info("🔗 Preparing on-chain commitment...")
        sub = await get_subtensor()

        await sub.set_reveal_commitment(
            wallet=w,
            netuid=settings.BABELBIT_NETUID,
            data=dumps(payload),
            blocks_until_reveal=1,
        )
        logger.info("✅ On-chain commitment submitted successfully")
    except Exception as e:
        error_name = type(e).__name__
        error_msg = str(e)
        
        if "AccountNotAllowedCommit" in error_msg:
            logger.error("❌ On-chain commit failed: Account not authorized to commit to subnet %s", settings.BABELBIT_NETUID)
            logger.info("💡 This usually means your wallet needs to be registered as a validator/miner first")
            logger.info("💡 Or you may need to wait for registration confirmation on the network")
        elif "InsufficientBalance" in error_msg:
            logger.error("❌ On-chain commit failed: Insufficient balance to pay transaction fees")
            logger.info("💡 Add more TAO to your coldkey wallet to cover transaction costs")
        else:
            logger.error("❌ On-chain commit failed: %s: %s", error_name, error_msg)
        
        logger.info("📄 Payload attempted: %s", payload)


async def _set_weights_with_confirmation(
    wallet,
    netuid: int,
    uids: list[int],
    weights: list[float],
    wait_for_inclusion: bool = False,
    retries: int = 10,
    delay_s: float = 2.0,
    log_prefix: str = "[bb-local]",
) -> bool:
    import bittensor as bt

    for attempt in range(retries):
        try:
            st = await get_subtensor()
            ref = await st.get_current_block()
            # (sync) submission via non-async client 
            bt.subtensor(
                os.getenv("BITTENSOR_SUBTENSOR_ENDPOINT", "finney")
            ).set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=uids,
                weights=weights,
                wait_for_inclusion=wait_for_inclusion,
            )
            await st.wait_for_block()
            meta = await st.metagraph(netuid)
            try:
                idx = meta.hotkeys.index(wallet.hotkey.ss58_address)
                lu = meta.last_update[idx]
                if lu >= ref:
                    logger.info(
                        f"{log_prefix} confirmation OK (last_update {lu} >= ref {ref})"
                    )
                    return True
                logger.warning(
                    f"{log_prefix} not included yet (last_update {lu} < ref {ref}), retry…"
                )
            except ValueError:
                logger.warning(
                    f"{log_prefix} wallet hotkey not found in metagraph; retry…"
                )
        except Exception as e:
            logger.warning(
                f"{log_prefix} attempt {attempt+1}/{retries} error: {type(e).__name__}: {e}"
            )
        await asyncio.sleep(delay_s)
    return False


async def test_metagraph() -> bool:
    """
    Test metagraph connectivity and display basic subnet information.
    """
    settings = get_settings()

    test_netuid = settings.BABELBIT_NETUID

    logger.info(f"\n=== Testing Metagraph Connection ===")
    logger.info(f"NETUID: {test_netuid}")
    logger.info(
        f"BITTENSOR_SUBTENSOR_ENDPOINT: {settings.BITTENSOR_SUBTENSOR_ENDPOINT}"
    )
    logger.info(
        f"BITTENSOR_SUBTENSOR_FALLBACK: {settings.BITTENSOR_SUBTENSOR_FALLBACK}"
    )

    try:
        # Test subtensor connection
        logger.info("\n1. Testing subtensor connection...")
        st = await get_subtensor()
        logger.info("✓ Subtensor connection successful")

        # Get current block
        logger.info("\n2. Getting current block...")
        current_block = await st.get_current_block()
        logger.info(f"✓ Current block: {current_block}")

        # Test metagraph fetch
        logger.info(f"\n3. Fetching metagraph for netuid {test_netuid}...")
        meta = await st.metagraph(test_netuid)
        logger.info(f"✓ Metagraph fetched successfully")

        # Display basic metagraph info
        logger.info(f"\n=== Metagraph Information ===")
        logger.info(f"Total neurons: {len(meta.hotkeys)}")
        logger.info(f"Block: {meta.block}")

        # Test wallet connection if configured
        wallet_cold = settings.BITTENSOR_WALLET_COLD
        wallet_hot = settings.BITTENSOR_WALLET_HOT

        logger.info(f"\n4. Testing wallet connection...")
        logger.info(f"Cold wallet: {wallet_cold}")
        logger.info(f"Hot wallet: {wallet_hot}")

        try:
            w = wallet(
                name=wallet_cold.get_secret_value(),
                hotkey=wallet_hot.get_secret_value(),
            )
            logger.info(f"✓ Wallet loaded successfully")
            logger.info(f"Hotkey SS58: {w.hotkey.ss58_address}")

            # Check if wallet is registered in subnet
            try:
                idx = meta.hotkeys.index(w.hotkey.ss58_address)
                click.echo(f"✓ Wallet found in metagraph at index: {idx}")
                click.echo(f"Last update: {meta.last_update[idx]}")
                if hasattr(meta, "stake") and len(meta.stake) > idx:
                    click.echo(f"Stake: {meta.stake[idx]}")
                if hasattr(meta, "trust") and len(meta.trust) > idx:
                    click.echo(f"Trust: {meta.trust[idx]}")
            except ValueError as e:
                logger.error(
                    "⚠ Wallet hotkey not found in metagraph (not registered in subnet)"
                )

        except Exception as e:
            logger.error(f"✗ Wallet loading failed: {e}")

        # Display some hotkeys for verification
        logger.info(f"\n=== Sample Hotkeys (first 5) ===")
        for i, hk in enumerate(meta.hotkeys[:5]):
            logger.info(f"UID {i}: {hk}")

        # Test the get_weights function logic
        logger.info(f"\n5. Testing weights calculation logic...")
        try:
            # Create a simple mapping test
            hk_to_uid = {hk: i for i, hk in enumerate(meta.hotkeys)}
            logger.info(f"✓ Hotkey to UID mapping created: {len(hk_to_uid)} entries")
        except Exception as e:
            logger.error(f"✗ Hotkey mapping failed: {e}")

        logger.info(f"\n=== Test Complete ===")
        logger.info("✓ All metagraph functions appear to be working correctly")

    except Exception as e:
        logger.error(f"\n✗ Metagraph test failed: {type(e).__name__}: {e}")

        print_exc()
        return False

    return True
