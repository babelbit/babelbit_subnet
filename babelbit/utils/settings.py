from functools import lru_cache
from os import getenv
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

__version__ = "0.1.0"


class Settings(BaseModel):
    # Bittensor
    BITTENSOR_WALLET_COLD: str
    BITTENSOR_WALLET_HOT: str
    BITTENSOR_WALLET_PATH: Path
    BITTENSOR_NETWORK: str
    BITTENSOR_SUBTENSOR_ENDPOINT: str
    BITTENSOR_SUBTENSOR_FALLBACK: str

    # Babelbit Core
    BABELBIT_NETUID: int
    BABELBIT_TEMPO: int
    BABELBIT_CACHE_DIR: Path
    BABELBIT_VERSION: str
    BABELBIT_API_TIMEOUT_S: int
    BABELBIT_MAX_CONCURRENT_API_CALLS: int
    BB_MINER_PREDICT_ENDPOINT: str
    BB_MINER_TIMEOUT_SEC: int
    BB_UTTERANCE_ENGINE_URL: str
    BB_RUNNER_ON_STARTUP: bool
    BB_SUBMIT_API_URL: str
    BB_ENABLE_ARENA_CHALLENGE: bool
    BB_ARENA_CADENCE_BLOCKS: int
    BB_ARENA_RUN_ON_STARTUP: bool
    BB_ARENA_GATEWAY_URL: str
    BB_ARENA_CONTAINERS_API_PATH: str
    BB_ARENA_CONTAINERS_STATUS: str
    BB_ARENA_CONTAINERS_WINDOW_SECONDS: int
    BB_ARENA_CONTAINERS_TIMEOUT_SEC: int
    BB_ARENA_RUNSYNC_API_PATH: str
    BB_ARENA_GATEWAY_AUTH_API_PATH: str
    BB_ARENA_MINER_TIMEOUT_SEC: int

    # HuggingFace
    HUGGINGFACE_USERNAME: str
    HUGGINGFACE_API_KEY: SecretStr
    HUGGINGFACE_CONCURRENCY: int

    # Signer
    SIGNER_URL: str
    SIGNER_SEED: SecretStr
    SIGNER_HOST: str
    SIGNER_PORT: int

    # Subtensor Gateway
    SUBTENSOR_GATEWAY_URL: str
    SUBTENSOR_GATEWAY_HOST: str
    SUBTENSOR_GATEWAY_PORT: int
    SUBTENSOR_GATEWAY_TIMEOUT_S: int

    # Database (PostgreSQL)
    PG_HOST: str
    PG_PORT: int
    PG_DB: str
    PG_USER: str
    PG_PASSWORD: SecretStr

    # S3 / Object Storage
    BB_ENABLE_S3_UPLOADS: bool = False
    S3_ENDPOINT_URL: str
    S3_REGION: str
    S3_ACCESS_KEY_ID: str
    S3_SECRET_ACCESS_KEY: SecretStr
    S3_BUCKET_NAME: str
    S3_SUBMISSIONS_DIR: str
    S3_LOG_DIR: str
    S3_ADDRESSING_STYLE: str
    S3_SIGNATURE_VERSION: str
    S3_USE_SSL: bool

    # Miner configuration
    MINER_MODEL_ID: str
    MINER_MODEL_REVISION: Optional[str]
    MINER_AXON_PORT: int
    MINER_DEVICE: str
    MINER_LOAD_IN_8BIT: bool
    MINER_LOAD_IN_4BIT: bool
    MINER_EXTERNAL_IP: Optional[str]

    # Development mode settings
    BB_DEV_MODE: bool = False
    BB_LOCAL_MINER_IP: Optional[str] = None


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        # Bittensor settings
        BITTENSOR_WALLET_COLD=getenv("BITTENSOR_WALLET_COLD", "default"),
        BITTENSOR_WALLET_HOT=getenv("BITTENSOR_WALLET_HOT", "default"),
        BITTENSOR_WALLET_PATH=Path(getenv("BITTENSOR_WALLET_PATH", "~/.bittensor/wallets")).expanduser(),
        BITTENSOR_NETWORK=getenv("BITTENSOR_NETWORK", "finney"),
        BITTENSOR_SUBTENSOR_ENDPOINT=getenv("BITTENSOR_SUBTENSOR_ENDPOINT", "finney"),
        BITTENSOR_SUBTENSOR_FALLBACK=getenv(
            "BITTENSOR_SUBTENSOR_FALLBACK", "wss://lite.sub.latent.to:443"
        ),

        # Babelbit core
        BABELBIT_NETUID=int(getenv("BABELBIT_NETUID", "59")),
        # enforcing tempo to 100 blocks to avoid vtrust issues
        BABELBIT_TEMPO=100,
        BABELBIT_CACHE_DIR=Path(getenv("BABELBIT_CACHE_DIR", "~/.babelbit")).expanduser().resolve(),
        BABELBIT_VERSION=getenv("BABELBIT_VERSION", __version__),
        BABELBIT_API_TIMEOUT_S=int(getenv("BABELBIT_API_TIMEOUT_S", "10")),
        BABELBIT_MAX_CONCURRENT_API_CALLS=int(getenv("BABELBIT_MAX_CONCURRENT_API_CALLS", "1")),
        BB_MINER_PREDICT_ENDPOINT=getenv("BB_MINER_PREDICT_ENDPOINT", "predict"),
        BB_MINER_TIMEOUT_SEC=int(getenv("BB_MINER_TIMEOUT_SEC", "10")),
        BB_UTTERANCE_ENGINE_URL=getenv("BB_UTTERANCE_ENGINE_URL", "https://api.babelbit.ai"),
        BB_RUNNER_ON_STARTUP=getenv("BB_RUNNER_ON_STARTUP", "false").strip().lower() in ("1", "true", "yes"),
        BB_SUBMIT_API_URL=getenv("BB_SUBMIT_API_URL", "https://scoring.babelbit.ai"),
        BB_ENABLE_ARENA_CHALLENGE=getenv("BB_ENABLE_ARENA_CHALLENGE", "true").strip().lower() in ("1", "true", "yes"),
        BB_ARENA_CADENCE_BLOCKS=int(getenv("BB_ARENA_CADENCE_BLOCKS", "300")),
        BB_ARENA_RUN_ON_STARTUP=getenv("BB_ARENA_RUN_ON_STARTUP", "false").strip().lower() in ("1", "true", "yes"),
        BB_ARENA_GATEWAY_URL=getenv("BB_ARENA_GATEWAY_URL", "https:/gw.babelbit.ai/"),
        BB_ARENA_CONTAINERS_API_PATH=getenv("BB_ARENA_CONTAINERS_API_PATH", "/list_arena_miners"),
        BB_ARENA_CONTAINERS_STATUS=getenv("BB_ARENA_CONTAINERS_STATUS", "running"),
        BB_ARENA_CONTAINERS_WINDOW_SECONDS=int(getenv("BB_ARENA_CONTAINERS_WINDOW_SECONDS", "300")),
        BB_ARENA_CONTAINERS_TIMEOUT_SEC=int(getenv("BB_ARENA_CONTAINERS_TIMEOUT_SEC", "10")),
        BB_ARENA_RUNSYNC_API_PATH=getenv("BB_ARENA_RUNSYNC_API_PATH", "/runsync"),
        BB_ARENA_GATEWAY_AUTH_API_PATH=getenv("BB_ARENA_GATEWAY_AUTH_API_PATH", "/auth/token"),
        BB_ARENA_MINER_TIMEOUT_SEC=int(getenv("BB_ARENA_MINER_TIMEOUT_SEC", "10")),

        # Development / local testing flags
        BB_DEV_MODE=getenv("BB_DEV_MODE", "0").lower() in ("1", "true", "yes"),
        BB_LOCAL_MINER_IP=getenv("BB_LOCAL_MINER_IP", ""),

        # HuggingFace settings
        HUGGINGFACE_USERNAME=getenv("HUGGINGFACE_USERNAME", ""),
        HUGGINGFACE_API_KEY=SecretStr(getenv("HUGGINGFACE_API_KEY", "")),
        HUGGINGFACE_CONCURRENCY=int(getenv("HUGGINGFACE_CONCURRENCY", "2")),

        # Signer settings
        SIGNER_URL=getenv("SIGNER_URL", "http://signer:8080"),
        SIGNER_SEED=SecretStr(getenv("SIGNER_SEED", "")),
        SIGNER_HOST=getenv("SIGNER_HOST", "127.0.0.1"),
        SIGNER_PORT=int(getenv("SIGNER_PORT", "8080")),
        SUBTENSOR_GATEWAY_URL=getenv(
            "SUBTENSOR_GATEWAY_URL", "http://subtensor-gateway:8090"
        ),
        SUBTENSOR_GATEWAY_HOST=getenv("SUBTENSOR_GATEWAY_HOST", "0.0.0.0"),
        SUBTENSOR_GATEWAY_PORT=int(getenv("SUBTENSOR_GATEWAY_PORT", "8090")),
        SUBTENSOR_GATEWAY_TIMEOUT_S=int(getenv("SUBTENSOR_GATEWAY_TIMEOUT_S", "30")),

        # Database settings
        PG_HOST=getenv("PG_HOST", "db"),
        PG_PORT=int(getenv("PG_PORT", "5432")),
        PG_DB=getenv("PG_DB", "babelbit"),
        PG_USER=getenv("PG_USER", "babelbit"),
        PG_PASSWORD=SecretStr(getenv("PG_PASSWORD", "babelbit")),

        # S3 / Object Storage settings
        S3_ENDPOINT_URL=getenv("S3_ENDPOINT_URL", ""),
        S3_REGION=getenv("S3_REGION", "us-east-1"),
        S3_ACCESS_KEY_ID=getenv("S3_ACCESS_KEY_ID", ""),
        S3_SECRET_ACCESS_KEY=SecretStr(getenv("S3_SECRET_ACCESS_KEY", "")),
        S3_BUCKET_NAME=getenv("S3_BUCKET_NAME", ""),
        S3_SUBMISSIONS_DIR=getenv("S3_SUBMISSIONS_DIR", "challenges"),
        S3_LOG_DIR=getenv("S3_LOG_DIR", "logs"),
        S3_ADDRESSING_STYLE=getenv("S3_ADDRESSING_STYLE", "path"),
        S3_SIGNATURE_VERSION=getenv("S3_SIGNATURE_VERSION", "s3v4"),
        S3_USE_SSL=getenv("S3_USE_SSL", "true").lower() in ("true", "1", "yes"),

        # Miner configuration
        MINER_MODEL_ID=getenv("MINER_MODEL_ID", "babelbit-ai/base-miner"),
        MINER_MODEL_REVISION=getenv("MINER_MODEL_REVISION"),
        MINER_AXON_PORT=int(getenv("MINER_AXON_PORT", "8091")),
        MINER_DEVICE=getenv("MINER_DEVICE", "cpu"),
        MINER_LOAD_IN_8BIT=getenv("MINER_LOAD_IN_8BIT", "0").lower() in {"1", "true", "yes"},
        MINER_LOAD_IN_4BIT=getenv("MINER_LOAD_IN_4BIT", "0").lower() in {"1", "true", "yes"},
        MINER_EXTERNAL_IP=getenv("MINER_EXTERNAL_IP"),
    )
