# ---- Prometheus ----
import os

try:
    from prometheus_client import Counter, Gauge, CollectorRegistry, start_http_server
    _PROMETHEUS_AVAILABLE = True
except ModuleNotFoundError:
    _PROMETHEUS_AVAILABLE = False

    class _NoopMetric:
        def labels(self, **_kwargs):
            return self

        def inc(self, *_args, **_kwargs):
            return None

        def set(self, *_args, **_kwargs):
            return None

    def Counter(*_args, **_kwargs):
        return _NoopMetric()

    def Gauge(*_args, **_kwargs):
        return _NoopMetric()

    class CollectorRegistry:  # pragma: no cover - trivial fallback
        def __init__(self, *args, **kwargs):
            pass

    def start_http_server(*_args, **_kwargs):
        return None

from babelbit.utils.settings import get_settings

settings = get_settings()
CACHE_DIR = settings.BABELBIT_CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PROM_REG = CollectorRegistry(auto_describe=True)

SHARDS_READ_TOTAL = Counter(
    "shards_read_total", "Total shard lines read (raw)", registry=PROM_REG
)
SHARDS_VALID_TOTAL = Counter(
    "shards_valid_total", "Total shard lines passed validation", registry=PROM_REG
)
SCORES_BY_UID = Gauge("scores_by_uid", "Score by uid", ["uid"], registry=PROM_REG)
WEIGHT_BY_UID = Gauge("weights", "Current weight by uid", ["uid"], registry=PROM_REG)
RANK_BY_UID = Gauge("rank", "Current rank by uid (1=best)", ["uid"], registry=PROM_REG)
CURRENT_WINNER = Gauge("current_winner_uid", "UID of current winner", registry=PROM_REG)
LASTSET_GAUGE = Gauge(
    "lastset", "Unix time of last successful set_weights", registry=PROM_REG
)
PREDICT_COUNT = Counter(
    "predict_count", "Predict calls counted from shards", ["model"], registry=PROM_REG
)
INDEX_KEYS_COUNT = Gauge(
    "index_keys_count", "Number of keys in index", registry=PROM_REG
)
CACHE_FILES = Gauge("cache_files", "Cached shard jsonl files", registry=PROM_REG)


def _start_metrics():
    try:
        if not _PROMETHEUS_AVAILABLE:
            return
        port = int(os.getenv("BABELBIT_METRICS_PORT", "8010"))
        addr = os.getenv("BABELBIT_METRICS_ADDR", "0.0.0.0")
        start_http_server(port, addr, registry=PROM_REG)
    except Exception:
        pass
