"""
Register miner's axon on the Bittensor network.
This script only handles on-chain registration so validators can discover the miner.
The actual serving of predictions is handled by serve_miner.py
"""
import argparse
import logging
import bittensor as bt

from babelbit.utils.settings import get_settings

logger = logging.getLogger(__name__)


def register_axon(external_ip_override=None):
    """Register the miner's axon on-chain."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    settings = get_settings()
    
    # Get wallet configuration
    wallet_name = settings.BITTENSOR_WALLET_COLD
    hotkey_name = settings.BITTENSOR_WALLET_HOT
    wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
    
    # Get axon configuration
    axon_port = settings.MINER_AXON_PORT
    external_ip = external_ip_override or settings.MINER_EXTERNAL_IP
    
    logger.info(f"Wallet: {wallet_name}/{hotkey_name}")
    logger.info(f"Hotkey: {wallet.hotkey.ss58_address}")
    logger.info(f"Registering axon at {external_ip or 'auto'}:{axon_port}")
    
    # Create axon
    axon = bt.axon(
        wallet=wallet,
        port=axon_port,
        external_ip=external_ip,
    )
    
    # Get network configuration
    network = settings.BITTENSOR_SUBTENSOR_ENDPOINT
    netuid = settings.BABELBIT_NETUID
    
    # Register the axon on-chain
    logger.info(f"Registering on {network} netuid {netuid}")
    subtensor = bt.subtensor(network=network)
    
    # Check if hotkey is registered on the subnet
    try:
        metagraph = bt.metagraph(netuid=netuid, network=network)
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
        logger.info("Now start your miner server with: uv run python babelbit/miner/serve_miner.py")
        
    except Exception as e:
        logger.error(f"❌ Failed to register axon: {e}")
        if "Custom error: 10" in str(e):
            logger.error("")
            logger.error("This error usually means your hotkey is not registered on the subnet.")
            logger.error(f"Register your hotkey first:")
            logger.error(f"   btcli subnet register --netuid {netuid} --wallet.name {wallet_name} --wallet.hotkey {hotkey_name}")
        raise



if __name__ == "__main__":
    # Parse custom arguments before bt.wallet() interferes
    parser = argparse.ArgumentParser(
        description="Register miner's axon on the Bittensor network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register with IP from settings/env
  python babelbit/miner/register_axon.py
  
  # Register with specific external IP
  python babelbit/miner/register_axon.py --external-ip x.x.x.x
  
  # Register with auto-detected IP
  python babelbit/miner/register_axon.py --external-ip auto
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
    
    # Parse known args to avoid conflicts with bittensor's parser
    args, unknown = parser.parse_known_args()
    
    # If user asked for help with our custom args, show it
    if '-h' in unknown or '--help' in unknown:
        parser.print_help()
        exit(0)
    
    register_axon(external_ip_override=args.external_ip)