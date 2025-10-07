from os import getenv
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

__version__ = "0.1.0"


class Settings(BaseModel):
    # Bittensor
    BITTENSOR_WALLET_COLD: str
    BITTENSOR_WALLET_HOT: str
    BITTENSOR_WALLET_PATH: Path
    BITTENSOR_SUBTENSOR_ENDPOINT: str
    BITTENSOR_SUBTENSOR_FALLBACK: str

    # Chutes
    CHUTES_API_KEY: SecretStr
    CHUTES_USERNAME: str
    CHUTES_MINER_PREDICT_ENDPOINT: str
    CHUTES_MINER_BASE_URL_TEMPLATE: str
    CHUTES_API_N_RETRIES: int
    CHUTES_TIMEOUT_SEC: int
    PATH_CHUTE_TEMPLATES: Path

    # Babelbit Core
    BABELBIT_NETUID: int
    BABELBIT_TEMPO: int
    BABELBIT_CACHE_DIR: Path
    BABELBIT_VERSION: str
    BABELBIT_API_TIMEOUT_S: int
    BABELBIT_MAX_CONCURRENT_API_CALLS: int
    BB_MINER_PREDICT_ENDPOINT: str
    BB_ENABLE_DB_WRITES: bool
    BB_UTTERANCE_ENGINE_URL: str


    # Template paths
    FILENAME_BB_MAIN: str
    FILENAME_BB_SCHEMAS: str
    FILENAME_BB_SETUP_UTILS: str
    FILENAME_BB_LOAD_UTILS: str
    FILENAME_BB_PREDICT_UTILS: str
    
    # Chute template filenames (aliases for compatibility)
    FILENAME_CHUTE_MAIN: str
    FILENAME_CHUTE_SCHEMAS: str
    FILENAME_CHUTE_SETUP_UTILS: str
    FILENAME_CHUTE_LOAD_UTILS: str
    FILENAME_CHUTE_PREDICT_UTILS: str

    # HuggingFace
    HUGGINGFACE_USERNAME: str
    HUGGINGFACE_API_KEY: SecretStr
    HUGGINGFACE_CONCURRENCY: int

    # Signer
    SIGNER_URL: str
    SIGNER_SEED: SecretStr
    SIGNER_HOST: str
    SIGNER_PORT: int

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


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        # Bittensor settings
        BITTENSOR_WALLET_COLD=getenv("BITTENSOR_WALLET_COLD", "default"),
        BITTENSOR_WALLET_HOT=getenv("BITTENSOR_WALLET_HOT", "default"),
        BITTENSOR_WALLET_PATH=Path(getenv("BITTENSOR_WALLET_PATH", "~/.bittensor/wallets")).expanduser(),
        BITTENSOR_SUBTENSOR_ENDPOINT=getenv("BITTENSOR_SUBTENSOR_ENDPOINT", "finney"),
        BITTENSOR_SUBTENSOR_FALLBACK=getenv(
            "BITTENSOR_SUBTENSOR_FALLBACK", "wss://lite.sub.latent.to:443"
        ),
        
        # Chutes settings (required)
        CHUTES_API_KEY=SecretStr(getenv("CHUTES_API_KEY", "")),
        CHUTES_USERNAME=getenv("CHUTES_USERNAME", ""),
        CHUTES_MINER_PREDICT_ENDPOINT=getenv("CHUTES_MINER_PREDICT_ENDPOINT", "predict"),
        CHUTES_MINER_BASE_URL_TEMPLATE=getenv("CHUTES_MINER_BASE_URL_TEMPLATE", "http://{slug}.chutes.ai"),
        CHUTES_API_N_RETRIES=int(getenv("CHUTES_API_N_RETRIES", "0")),
        CHUTES_TIMEOUT_SEC=int(getenv("CHUTES_TIMEOUT_SEC", "10")),
        # Template paths - resolve to absolute to avoid duplication
        PATH_CHUTE_TEMPLATES=Path(
            getenv(
                "PATH_CHUTE_TEMPLATES",
                str(Path(__file__).resolve().parent.parent / "chute_template"),
            )
        ).expanduser().resolve(),

        # Babelbit core
        BABELBIT_NETUID=int(getenv("BABELBIT_NETUID", "44")),
        BABELBIT_TEMPO=int(getenv("BABELBIT_TEMPO", "360")),
        BABELBIT_CACHE_DIR=Path(getenv("BABELBIT_CACHE_DIR", "~/.babelbit")).expanduser().resolve(),
        BABELBIT_VERSION=getenv("BABELBIT_VERSION", __version__),
        BABELBIT_API_TIMEOUT_S=int(getenv("BABELBIT_API_TIMEOUT_S", "10")),
        BABELBIT_MAX_CONCURRENT_API_CALLS=int(getenv("BABELBIT_MAX_CONCURRENT_API_CALLS", "1")),
        BB_MINER_PREDICT_ENDPOINT=getenv("BB_MINER_PREDICT_ENDPOINT", "predict"),
        BB_ENABLE_DB_WRITES=getenv("BB_ENABLE_DB_WRITES", "0").lower() in ("1", "true", "yes"),
        BB_UTTERANCE_ENGINE_URL=getenv("BB_UTTERANCE_ENGINE_URL", "http://localhost:8999"),
        
        

        FILENAME_BB_MAIN=getenv("FILENAME_BB_MAIN", "chute.py.j2"),
        FILENAME_BB_SCHEMAS=getenv("FILENAME_BB_SCHEMAS", "schemas.py"),
        FILENAME_BB_SETUP_UTILS=getenv("FILENAME_BB_SETUP_UTILS", "setup.py"),
        FILENAME_BB_LOAD_UTILS=getenv("FILENAME_BB_LOAD_UTILS", "load.py"),
        FILENAME_BB_PREDICT_UTILS=getenv("FILENAME_BB_PREDICT_UTILS", "predict.py"),
        
        # Chute template filenames (aliases for compatibility)
        FILENAME_CHUTE_MAIN=getenv("FILENAME_CHUTE_MAIN", getenv("FILENAME_BB_MAIN", "chute.py.j2")),
        FILENAME_CHUTE_SCHEMAS=getenv("FILENAME_CHUTE_SCHEMAS", getenv("FILENAME_BB_SCHEMAS", "schemas.py")),
        FILENAME_CHUTE_SETUP_UTILS=getenv("FILENAME_CHUTE_SETUP_UTILS", getenv("FILENAME_BB_SETUP_UTILS", "setup.py")),
        FILENAME_CHUTE_LOAD_UTILS=getenv("FILENAME_CHUTE_LOAD_UTILS", getenv("FILENAME_BB_LOAD_UTILS", "load.py")),
        FILENAME_CHUTE_PREDICT_UTILS=getenv("FILENAME_CHUTE_PREDICT_UTILS", getenv("FILENAME_BB_PREDICT_UTILS", "predict.py")),
        
        # HuggingFace settings
        HUGGINGFACE_USERNAME=getenv("HUGGINGFACE_USERNAME", ""),
        HUGGINGFACE_API_KEY=SecretStr(getenv("HUGGINGFACE_API_KEY", "")),
        HUGGINGFACE_CONCURRENCY=int(getenv("HUGGINGFACE_CONCURRENCY", "2")),
        
        # Signer settings
        SIGNER_URL=getenv("SIGNER_URL", "http://signer:8080"),
        SIGNER_SEED=SecretStr(getenv("SIGNER_SEED", "")),
        SIGNER_HOST=getenv("SIGNER_HOST", "127.0.0.1"),
        SIGNER_PORT=int(getenv("SIGNER_PORT", "8080")),
        
        # Database settings
        PG_HOST=getenv("PG_HOST", "localhost"),
        PG_PORT=int(getenv("PG_PORT", "5432")),
        PG_DB=getenv("PG_DB", "postgres"),
        PG_USER=getenv("PG_USER", "postgres"),
        PG_PASSWORD=SecretStr(getenv("PG_PASSWORD", "")),
        
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
    )
