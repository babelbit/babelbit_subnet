# score_dialogue.py
import argparse
import datetime
import json
import logging
import os
import random
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger("score_dialogue")

# ---------------- embedder + cosine ----------------

EMBEDDER_NAME = os.getenv("EMBEDDER_NAME", "mixedbread-ai/mxbai-embed-large-v1")
EMBED_DIM = int(os.getenv("EMBED_DIM", "64"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
EMBED_DEVICE = os.getenv("BB_SCORER_DEVICE", "cpu")

BASELINE_PAIRS = int(os.getenv("BASELINE_PAIRS", "100"))
BASELINE_SEED = int(os.getenv("BASELINE_SEED", "0"))

GT_EMBED_CACHE_MAX = int(os.getenv("GT_EMBED_CACHE_MAX", "4096"))
BASELINE_CACHE_MAX = int(os.getenv("BASELINE_CACHE_MAX", "1024"))
PRELOADED_EVENTS_CACHE_MAX = int(os.getenv("PRELOADED_EVENTS_CACHE_MAX", "1024"))

_embedder = None
_gt_embed_cache: "OrderedDict[Tuple[str, int, str], torch.Tensor]" = OrderedDict()
_baseline_cache: "OrderedDict[Tuple[str, int, int, int, Tuple[str, ...]], float]" = OrderedDict()
_preloaded_events_cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
_CACHE_LOCK = threading.RLock()


def _get_embedder():
    """Get or initialize the sentence transformer model."""
    global _embedder
    if _embedder is None:
        with _CACHE_LOCK:
            if _embedder is None:
                _embedder = SentenceTransformer(
                    EMBEDDER_NAME,
                    truncate_dim=EMBED_DIM,
                    device=EMBED_DEVICE,
                )
    return _embedder


def prewarm_score_dialogue_embedder() -> None:
    """Load the scorer embedder early so the first dialogue avoids model init latency."""
    _get_embedder()


def _cache_set_with_limit(cache: OrderedDict, key: Any, value: Any, max_items: int) -> None:
    with _CACHE_LOCK:
        if key in cache:
            cache.pop(key, None)
        cache[key] = value
        while len(cache) > max_items:
            cache.popitem(last=False)


def _encode_gt_texts_cached(texts: List[str]) -> torch.Tensor:
    """
    Return normalized embeddings for texts, batching only cache misses.
    Cache key includes embedder name + truncate dim to avoid cross-model mixups.
    """
    embedder = _get_embedder()
    keys = [(EMBEDDER_NAME, EMBED_DIM, t) for t in texts]
    with _CACHE_LOCK:
        missing = [t for key, t in zip(keys, texts) if key not in _gt_embed_cache]
    if missing:
        encoded = embedder.encode(
            missing,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=EMBED_BATCH_SIZE,
        )
        if isinstance(encoded, torch.Tensor) and encoded.dim() == 1:
            encoded = encoded.unsqueeze(0)
        missing_map: Dict[str, torch.Tensor] = {}
        for text, emb in zip(missing, encoded):
            missing_map[text] = emb
            _cache_set_with_limit(
                _gt_embed_cache,
                (EMBEDDER_NAME, EMBED_DIM, text),
                emb,
                GT_EMBED_CACHE_MAX,
            )
    else:
        missing_map = {}

    with _CACHE_LOCK:
        resolved: List[torch.Tensor] = []
        for key, text in zip(keys, texts):
            emb = _gt_embed_cache.get(key)
            if emb is None and missing:
                emb = missing_map.get(text)
            if emb is not None:
                resolved.append(emb)
        return torch.stack(resolved, dim=0)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _calibrated_semantic(cos_raw: float, baseline_b: float) -> float:
    """
    Calibrated semantic similarity in [0,1]:
      s = clamp01((c - b) / (1 - b))
    """
    denom = max(1.0 - baseline_b, 1e-6)
    return _clamp01((cos_raw - baseline_b) / denom)


def _estimate_baseline_b_from_embeddings(embs: torch.Tensor, n_pairs: int, seed: int) -> float:
    """Estimate baseline b from already-normalized sentence embeddings."""
    if embs is None or embs.size(0) < 2:
        return 0.0

    rng = random.Random(seed)
    total = 0.0
    count = 0
    n = int(embs.size(0))

    for _ in range(max(1, n_pairs)):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            j = (j + 1) % n
        # embeddings are normalized, so dot == cosine
        c = util.dot_score(embs[i], embs[j]).item()
        total += float(c)
        count += 1

    return float(total / max(1, count))


def _estimate_baseline_b(sentences: List[str], n_pairs: int, seed: int) -> float:
    """Estimate baseline b by averaging cosine over random pairs from cached GT embeddings."""
    sentences = [s for s in (sentences or []) if (s or "").strip()]
    if len(sentences) < 2:
        return 0.0

    cache_key = (EMBEDDER_NAME, EMBED_DIM, n_pairs, seed, tuple(sentences))
    with _CACHE_LOCK:
        cached = _baseline_cache.get(cache_key)
    if cached is not None:
        return cached

    embs = _encode_gt_texts_cached(sentences)
    baseline_b = _estimate_baseline_b_from_embeddings(embs, n_pairs=n_pairs, seed=seed)
    _cache_set_with_limit(_baseline_cache, cache_key, baseline_b, BASELINE_CACHE_MAX)
    return baseline_b




# ---------------- utilities ----------------

def _strip_eof(s: str) -> str:
    s = (s or "").strip()
    return s[:-3].strip() if s.endswith("EOF") else s


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def prime_score_jsonl_events(path: Union[Path, str], events: List[Dict[str, Any]]) -> None:
    """
    Prime in-memory events for a given path so score_jsonl can avoid re-reading/parsing JSONL.
    The cache entry is consumed exactly once by score_jsonl(path).
    """
    path_key = str(Path(path))
    _cache_set_with_limit(
        _preloaded_events_cache,
        path_key,
        list(events),
        PRELOADED_EVENTS_CACHE_MAX,
    )


def _consume_preloaded_events(path: Path) -> Optional[List[Dict[str, Any]]]:
    with _CACHE_LOCK:
        return _preloaded_events_cache.pop(str(path), None)


def _edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ca = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ca == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m]


def _char_similarity(a: str, b: str) -> float:
    a = a or ""
    b = b or ""
    mx = max(len(a), len(b))
    if mx == 0:
        return 0.0
    ed = _edit_distance(a, b)
    return max(0.0, 1.0 - (ed / mx))


# ---------------- logging / output paths ----------------

def _open_score_log(jsonl_path: Path) -> Tuple[Path, str]:
    """
    score file location: scores/<input-stem>_run_<YYYYMMDD_HHMMSS>-score.txt
    Returns (txt_path, ts_str)
    """
    stem = jsonl_path.stem
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("scores")
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{stem}_run_{ts}-score.txt"
    return txt_path, ts


def _configure_logging(score_txt_path: Path) -> None:
    """
    Configure logger to write to BOTH stdout and the score txt file.
    All messages are emitted via logger.info()/logger.debug(); no print().
    """
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if main() is called multiple times.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(score_txt_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)


# --------------- scoring ---------------

def score_jsonl(path: Union[Path, str], lex_weight: float = 0.0, show_steps: bool = True) -> Dict[str, Any]:
    path = Path(path)
    json_doc: Dict[str, Any] = {
        "log_file": str(path),
        "dialogue_uid": None,
        "utterances": [],
        "dialogue_summary": {"average_U_best_early": 0.0},
    }

    by_utt: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    preloaded_events = _consume_preloaded_events(path)
    source_iter: Iterable[Dict[str, Any]] = preloaded_events if preloaded_events is not None else _iter_jsonl(path)
    for obj in source_iter:
        idx = obj.get("utterance_index")
        if idx is None:
            continue
        b = by_utt.setdefault(int(idx), {"predicted": [], "revealed": [], "complete": []})
        ev = obj.get("event")
        if ev == "predicted":
            b["predicted"].append(obj)
        elif ev == "revealed":
            b["revealed"].append(obj)
        elif ev == "utterance_complete":
            b["complete"].append(obj)

    use_lexical = lex_weight > 0.0
    use_semantic = lex_weight < 1.0

    # GT pool for baseline b and cached GT embeddings
    gt_pool: List[str] = []
    for idx in sorted(by_utt.keys()):
        comp = by_utt[idx]["complete"]
        if not comp:
            continue
        gt_full = _strip_eof((comp[-1].get("ground_truth") or "").strip())
        if gt_full:
            gt_pool.append(gt_full)

    baseline_b = _estimate_baseline_b(gt_pool, n_pairs=BASELINE_PAIRS, seed=BASELINE_SEED) if use_semantic else 0.0

    logger.debug("")
    logger.debug("=== %s (JSONL per-step) ===", str(path))
    logger.debug("")
    logger.debug("Embedder: %s  EMBED_DIM=%s  EMBED_BATCH_SIZE=%s", EMBEDDER_NAME, EMBED_DIM, EMBED_BATCH_SIZE)
    logger.debug(
        "Semantic baseline b from %d GT utts over %d random pairs (seed=%d): b=%.6f",
        len(gt_pool),
        BASELINE_PAIRS,
        BASELINE_SEED,
        baseline_b,
    )

    logger.debug("")

    embedder = _get_embedder() if use_semantic else None
    pppl_cache: Dict[str, float] = {}
    dialogue_scores: List[float] = []

    for idx in sorted(by_utt.keys()):
        bucket = by_utt[idx]
        preds = bucket["predicted"]
        comp = bucket["complete"]
        if not comp:
            continue

        gt_full = _strip_eof((comp[-1].get("ground_truth") or "").strip())

        rev_steps = [r.get("step") for r in bucket["revealed"] if isinstance(r.get("step"), int)]
        total_steps = (max(rev_steps) + 1) if rev_steps else len(preds)

        step_to_pred: Dict[int, str] = {}
        for p in preds:
            s = p.get("step")
            if isinstance(s, int):
                step_to_pred[int(s)] = p.get("prediction") or ""

        pred_texts = [_strip_eof(step_to_pred.get(s, "")) for s in range(total_steps)]

        if use_semantic:
            # ----- semantic: compute cos_raw_mat ONCE and use it -----
            gt_emb = _encode_gt_texts_cached([gt_full]).squeeze(0)
            pred_embs = embedder.encode(
                pred_texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=EMBED_BATCH_SIZE,
            )

            # embeddings are normalized => dot_score == cosine similarity
            cos_raw_mat = util.dot_score(pred_embs, gt_emb).squeeze(-1)  # (steps,)

            # precompute calibrated semantics as a Python list for simple access
            semantic_scores: List[float] = [
                _calibrated_semantic(float(cos_raw_mat[s].item()), baseline_b) for s in range(total_steps)
            ]
        else:
            semantic_scores = [0.0] * total_steps
            cos_raw_mat = torch.zeros(total_steps, dtype=torch.float32)


        best_step = 0
        best_U = -1.0
        steps_array: List[Dict[str, Any]] = []

        if show_steps:
            logger.debug("[utt %d] ground_truth: %s", idx, gt_full)
            logger.debug("step\tlex\tcos_raw\tsem_cal\tearli\tU_step\tprediction")

        for s in range(total_steps):
            pred = pred_texts[s]

            lex_s = _char_similarity(pred, gt_full) if use_lexical else 0.0
            cos_raw = float(cos_raw_mat[s].item())
            sem_s = semantic_scores[s]


            earliness_s = 1.0 / (s + 1)


            U_step = ((lex_s * lex_weight) + (sem_s * (1.0 - lex_weight))) * earliness_s

            if show_steps:

                logger.debug(
                    "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%s",
                    s,
                    lex_s,
                    cos_raw,
                    sem_s,
                    earliness_s,
                    U_step,
                    pred,
                )

            if U_step > best_U:
                best_U = U_step
                best_step = s

            steps_array.append(
                {
                    "step": int(s),
                    "lexical_similarity": round(lex_s, 4),
                    "semantic_cosine_raw": round(cos_raw, 6),
                    "semantic_similarity": round(sem_s, 4),
                    "earliness": round(earliness_s, 4),
                    "U_step": round(float(U_step), 4),
                    "prediction": pred,
                }
            )

        logger.info("[utt %d] BEST step=%d  U_best=%.4f  total_steps=%d", idx, best_step, best_U, total_steps)
        logger.info("")

        dialogue_scores.append(best_U)
        json_doc["utterances"].append(
            {
                "utterance_number": int(idx),
                "ground_truth": gt_full,
                "steps": steps_array,
                "best_step": int(best_step),
                "U_best": round(float(best_U), 4),
                "total_steps": int(total_steps),
            }
        )

    dialogue_avg = (sum(dialogue_scores) / len(dialogue_scores)) if dialogue_scores else 0.0
    logger.info("Dialogue average U (best-early): %.4f", dialogue_avg)
    logger.info("")

    json_doc["dialogue_summary"]["average_U_best_early"] = round(dialogue_avg, 4)
    json_doc["dialogue_summary"]["semantic_baseline_b"] = round(float(baseline_b), 6)

    return json_doc


# --------------- CLI ---------------

def main(argv: List[str] = None) -> None:
    argv = argv or sys.argv
    parser = argparse.ArgumentParser(description="Score phrase_completion JSONL logs per-step with earliness.")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to dialogue_run_*.jsonl log")
    parser.add_argument(
        "--lex-weight",
        type=float,
        default=0.5,
        help="Weight for lexical similarity (0..1); 1-lex used for semantic",
    )
    args = parser.parse_args(argv[1:])

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise SystemExit(f"ERROR: --jsonl file not found: {jsonl_path}")

    score_txt_path, ts_part = _open_score_log(jsonl_path)
    _configure_logging(score_txt_path)

    json_doc = score_jsonl(jsonl_path, lex_weight=args.lex_weight, show_steps=True)

    # Write JSON sibling file next to the txt score, using same timestamp piece
    try:
        out_json = score_txt_path.with_name(f"{jsonl_path.stem}_run_{ts_part}-score.json")
        with out_json.open("w", encoding="utf-8") as jf:
            json.dump(json_doc, jf, ensure_ascii=False, indent=2)
        logger.info("Wrote JSON summary: %s", str(out_json))
    except Exception as e:
        logger.info("Failed to write JSON summary: %s", str(e))


if __name__ == "__main__":
    main()
