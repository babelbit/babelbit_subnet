"""
Register miner's axon on the Bittensor network.
This script only handles on-chain registration so validators can discover the miner.
The actual serving of predictions is handled by serve_miner.py
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.settings import get_settings

logger = logging.getLogger(__name__)


def _load_bittensor_types() -> tuple[Any, Any, Any, Any]:
    try:
        from bittensor import Metagraph, Subtensor, Wallet
        from bittensor.core.axon import Axon
    except ImportError as exc:
        raise RuntimeError("bittensor is required for axon registration") from exc

    return Wallet, Subtensor, Metagraph, Axon


def register_axon(external_ip_override=None, port_override=None):
    """Register the miner's axon on-chain."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    wallet_cls, subtensor_cls, metagraph_cls, axon_cls = _load_bittensor_types()

    settings = get_settings()

    # Get wallet configuration from env to avoid coupling to miner runtime settings
    wallet_name = os.getenv("BITTENSOR_WALLET_COLD") or os.getenv("BT_WALLET") or "default"
    hotkey_name = os.getenv("BITTENSOR_WALLET_HOT") or os.getenv("BT_HOTKEY") or "default"
    wallet = wallet_cls(name=wallet_name, hotkey=hotkey_name)

    # Get axon configuration
    axon_port = port_override or settings.MINER_AXON_PORT
    external_ip = external_ip_override or settings.MINER_EXTERNAL_IP
    if external_ip == "auto":
        external_ip = None

    logger.info(f"Wallet: {wallet_name}/{hotkey_name}")
    logger.info(f"Hotkey: {wallet.hotkey.ss58_address}")
    logger.info(f"Registering axon at {external_ip or 'auto'}:{axon_port}")

    # Create axon
    axon = axon_cls(
        wallet=wallet,
        port=axon_port,
        external_ip=external_ip,
    )

    # Get network configuration
    network = settings.BITTENSOR_SUBTENSOR_ENDPOINT
    netuid = settings.BABELBIT_NETUID

    # Register the axon on-chain
    logger.info(f"Registering on {network} netuid {netuid}")
    subtensor = subtensor_cls(network=network)

    # Check if hotkey is registered on the subnet
    try:
        metagraph = metagraph_cls(netuid=netuid, network=network, subtensor=subtensor)
        hotkey = wallet.hotkey.ss58_address

        if hotkey not in metagraph.hotkeys:
            logger.error(f"❌ Hotkey {hotkey} is not registered on subnet {netuid}")
            logger.error(f"")
            logger.error(f"First register your hotkey to the subnet:")
            logger.error(f"   btcli subnet register --netuid {netuid} --wallet.name {wallet_name} --wallet.hotkey {hotkey_name}")
            logger.error(f"")
            return

        uid = metagraph.hotkeys.index(hotkey)
        logger.info(f"Hotkey is registered with UID: {uid}")

    except Exception as e:
        logger.warning(f"Could not check metagraph: {e}")

    try:
        axon.serve(
            netuid=netuid,
            subtensor=subtensor,
        )
        logger.info(f"✅ Axon registered successfully!")
        logger.info(f"   IP: {axon.external_ip}")
        logger.info(f"   Port: {axon.external_port}")
        logger.info(f"   Hotkey: {wallet.hotkey.ss58_address}")
        logger.info("")
        logger.info("Now start your miner server with: uv run bb server")

    except Exception as e:
        logger.error(f"❌ Failed to register axon: {e}")
        if "Custom error: 10" in str(e):
            logger.error("")
            logger.error("This error usually means your hotkey is not registered on the subnet.")
            logger.error(f"Register your hotkey first:")
            logger.error(f"   btcli subnet register --netuid {netuid} --wallet.name {wallet_name} --wallet.hotkey {hotkey_name}")
        raise


def main() -> int:
    # Parse custom arguments before bt.wallet() interferes
    parser = argparse.ArgumentParser(
        description="Register miner's axon on the Bittensor network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register with IP from settings/env
  python cli/register_axon.py
  
  # Register with specific external IP
  python cli/register_axon.py --external-ip x.x.x.x

  # Register with specific IP and port
  python cli/register_axon.py --external-ip x.x.x.x --port 8010
  
  # Register with auto-detected IP
  python cli/register_axon.py --external-ip auto
        """,
        add_help=True,
        allow_abbrev=False
    )
    parser.add_argument(
        "--external-ip",
        type=str,
        help="External IP address for the axon (overrides MINER_EXTERNAL_IP setting)",
        default=None
    )
    parser.add_argument(
        "--port",
        type=int,
        help="External/listening port for the axon (overrides MINER_AXON_PORT)",
        default=None
    )

    # Parse known args to avoid conflicts with bittensor's parser
    args, unknown = parser.parse_known_args()

    # If user asked for help with our custom args, show it
    if "-h" in unknown or "--help" in unknown:
        parser.print_help()
        return 0

    register_axon(external_ip_override=args.external_ip, port_override=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
