import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

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

def _edit_distance(a: str, b: str) -> int:
    # Levenshtein distance (O(nm), sufficient for short utterances)
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
            dp[j] = min(dp[j] + 1,      # deletion
                        dp[j - 1] + 1,  # insertion
                        prev + cost)    # substitution
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

def _token_jaccard(a: str, b: str) -> float:
    ta = set((a or "").split())
    tb = set((b or "").split())
    inter = len(ta & tb)
    union = len(ta | tb)
    return (inter / union) if union else 0.0

# --- begin: score-log (add-only, mirrors STDOUT to a file; no logic changes) ---

def _open_score_log(argv: List[str]) -> Tuple[Path, Path]:
    """
    Decide score file location: scores/<input-stem>_run_<YYYYMMDD_HHMMSS>-score.txt
    Returns (txt_path, run_ts_str) where run_ts_str is the encoded timestamp part.
    """
    first_input = None
    # find the first non-flag arg that looks like an input file path
    for i, a in enumerate(argv[1:], start=1):
        if not a.startswith("-"):
            first_input = a
            break
    stem = Path(first_input).stem if first_input else "dialogue"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("scores")
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{stem}_run_{ts}-score.txt"
    return txt_path, ts

class Tee:
    """Write to both stdout and a file handle."""
    def __init__(self, file_handle):
        self.file_handle = file_handle
        self.stdout = sys.stdout
    def write(self, s: str):
        self.stdout.write(s)
        self.file_handle.write(s)
    def flush(self):
        self.stdout.flush()
        self.file_handle.flush()

# --- end: score-log ---

# --------------- JSONL scoring ---------------

def score_jsonl(path: Path, lex_weight: float = 0.5, show_steps: bool = True) -> Dict[str, Any]:
    """
    Compute per-step metrics and print human-readable tables.
    Returns the JSON document we also persist to scores/*.json.
    """
    # JSON output doc (strict; validates against dialogue-score.schema.json)
    json_doc = {
        "log_file": str(path),
        "dialogue_uid": None,  # intentionally null; not yet implemented
        "utterances": [],
        "dialogue_summary": {"average_U_best_early": 0.0},
    }

    # Group events by utterance_index
    by_utt: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    for obj in _iter_jsonl(path):
        idx = obj.get("utterance_index")
        if idx is None:
            continue
        b = by_utt.setdefault(idx, {"predicted": [], "revealed": [], "complete": []})
        ev = obj.get("event")
        if ev == "predicted":
            b["predicted"].append(obj)
        elif ev == "revealed":
            b["revealed"].append(obj)
        elif ev == "utterance_complete":
            b["complete"].append(obj)

    print(f"\n=== {path} (JSONL per-step) ===\n")

    dialogue_scores: List[float] = []
    for idx in sorted(by_utt.keys()):
        bucket = by_utt[idx]
        preds = bucket["predicted"]
        comp  = bucket["complete"]

        if not comp:
            # Can't score without final ground truth; skip this utt
            continue

        gt_full = _strip_eof((comp[-1].get("ground_truth") or "").strip())

        # Determine total steps from revealed events; fallback to #preds
        rev_steps = [r.get("step") for r in bucket["revealed"] if isinstance(r.get("step"), int)]
        total_steps = (max(rev_steps) + 1) if rev_steps else len(preds)

        # map: step -> last prediction at that step
        step_to_pred: Dict[int, str] = {}
        for p in preds:
            s = p.get("step")
            if isinstance(s, int):
                step_to_pred[s] = p.get("prediction") or ""

        best_step = None
        best_U = -1.0

        steps_array: List[Dict[str, Any]] = []

        if show_steps:
            print(f"[utt {idx}] ground_truth: {gt_full}")
            print("step\tlex\tsem\tearli\tU_step\tprediction")

        for s in range(total_steps):
            pred = _strip_eof(step_to_pred.get(s, ""))
            lex_s = _char_similarity(pred, gt_full)
            sem_s = _token_jaccard(pred, gt_full)
            earliness_s = 1.0 / (s + 1)  # first prediction gets weight 1.0
            U_step = ((lex_s * lex_weight) + (sem_s * (1.0 - lex_weight))) * earliness_s

            if show_steps:
                print(f"{s}\t{round(lex_s,4)}\t{round(sem_s,4)}\t{round(earliness_s,4)}\t{round(U_step,4)}\t{pred}")

            if U_step > best_U:
                best_U = U_step
                best_step = s

            # collect per-step JSON
            steps_array.append({
                "step": int(s),
                "lexical_similarity": round(lex_s, 4),
                "semantic_similarity": round(sem_s, 4),
                "earliness": round(earliness_s, 4),
                "U_step": round(U_step, 4),
                "prediction": pred,
            })

        print(f"[utt {idx}] BEST step={best_step}  U_best={round(best_U,4)}  total_steps={total_steps}\n")
        dialogue_scores.append(best_U)

        # append utterance JSON
        json_doc["utterances"].append({
            "utterance_number": int(idx),
            "ground_truth": gt_full,
            "steps": steps_array,
            "best_step": int(best_step if best_step is not None else 0),
            "U_best": round(best_U if best_U is not None else 0.0, 4),
            "total_steps": int(total_steps),
        })

    dialogue_avg = (sum(dialogue_scores) / len(dialogue_scores)) if dialogue_scores else 0.0
    print(f"\nDialogue average U (best-early): {round(dialogue_avg, 4)}\n")

    # finalize JSON
    json_doc["dialogue_summary"]["average_U_best_early"] = round(dialogue_avg, 4)
    return json_doc

# --------------- CLI ---------------

def main(argv: List[str] = None) -> None:
    argv = argv or sys.argv
    parser = argparse.ArgumentParser(description="Score phrase_completion JSONL logs per-step with earliness.")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to dialogue_run_*.jsonl log")
    parser.add_argument("--lex-weight", type=float, default=0.5, help="Weight for lexical similarity (0..1); 1-sem used for semantic")
    args = parser.parse_args(argv[1:])

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"ERROR: --jsonl file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(2)

    # Open score txt and tee stdout
    score_txt_path, ts_part = _open_score_log(argv)
    with score_txt_path.open("w", encoding="utf-8") as fh:
        tee = Tee(fh)
        old_stdout = sys.stdout
        sys.stdout = tee
        try:
            # Run scorer
            json_doc = score_jsonl(jsonl_path, lex_weight=args.lex_weight, show_steps=True)
        finally:
            sys.stdout = old_stdout

    # Write JSON sibling file next to the txt score, using same timestamp piece
    try:
        out_json = score_txt_path.with_name(f"{jsonl_path.stem}_run_{ts_part}-score.json")
        with out_json.open("w", encoding="utf-8") as jf:
            json.dump(json_doc, jf, ensure_ascii=False, indent=2)
    except Exception as e:
        # Never interfere with existing behavior
        pass

if __name__ == "__main__":
    main()