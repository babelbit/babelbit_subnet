#!/usr/bin/env python3
"""
score_dialogue.py — per-step, earliness-first scoring for phrase_completion JSONL logs.
Also supports per-dialogue JSON summaries (exact match), but JSONL mode is primary.

JSONL mode:
- For each utterance, every `predicted` step is compared to the final ground truth.
- Per-step scores:
    lex_s       = character similarity (1 - edit_distance / max_len)
    sem_s       = token-set Jaccard similarity
    earliness_s = 1 / (step + 1)          # step is 0-based (first prediction gets weight 1.0)
    U_step      = ((lex_s + sem_s) / 2) * earliness_s
- Reports the best early step (argmax U_step) as the utterance score.
- Dialogue average U = mean of best-early scores across utterances.

JSON mode (summary):
- Prints exact match counts only (for completeness).

Usage:
  # Per-step (recommended)
  python3 score_dialogue.py --jsonl logs/run_001.jsonl

  # Multiple logs or mix of formats
  python3 score_dialogue.py --jsonl logs/run_001.jsonl --jsonl logs/run_002.jsonl
  python3 score_dialogue.py out/dialogue_001.prediction.json

Options:
  --lex-weight FLOAT    # weight for lexical vs semantic inside the per-step blend (default 0.5)
  --no-steps            # hide per-step tables; show only per-utterance and dialogue summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# --- begin: score-log (add-only, mirrors STDOUT to a file; no logic changes) ---
import builtins, datetime, os, sys

def _open_score_log(argv):
    """Open scores/<input-stem>_run_<YYYYMMDD_HHMMSS>-score.txt for writing."""
    first_input = next((a for a in argv[1:] if not a.startswith("-")), None)
    stem = Path(first_input).stem if first_input else "dialogue"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("scores", exist_ok=True)
    return open(Path("scores") / f"{stem}_run_{ts}-score.txt", "w", encoding="utf-8")
# --- end: score-log (add-only) ---

# ---------------- utilities ----------------

def _strip_eof(s: str) -> str:
    s = (s or "").strip()
    return s[:-3].strip() if s.endswith("EOF") else s

def _iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    if la > lb:
        a, b, la, lb = b, a, lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        cur = [j] + [0]*la
        bj = b[j-1]
        for i in range(1, la + 1):
            cost = 0 if a[i-1] == bj else 1
            cur[i] = min(prev[i] + 1, cur[i-1] + 1, prev[i-1] + cost)
        prev = cur
    return prev[la]

def _char_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    m = max(len(a or ""), len(b or ""))
    if m == 0:
        return 1.0
    dist = _levenshtein(a or "", b or "")
    return 1.0 - (dist / m)

def _token_jaccard(a: str, b: str) -> float:
    A = set((a or "").split())
    B = set((b or "").split())
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

# --------------- JSON summary mode ---------------

def score_summary_json(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    turns = data.get("turns", [])
    correct = 0
    total = len(turns)
    for t in turns:
        gt = (t.get("ground_truth") or "").strip()
        pred = _strip_eof((t.get("final_prediction") or "").strip())
        if gt == pred:
            correct += 1
    print(f"{path}: {correct}/{total} utterances matched exactly")

# --------------- JSONL per-step mode ---------------

def score_jsonl(path: Path, lex_weight: float = 0.5, show_steps: bool = True) -> None:
    # Group events by utterance_index
    by_utt: Dict[int, Dict[str, List[Dict]]] = {}
    for obj in _iter_jsonl(path):
        idx = obj.get("utterance_index")
        if idx is None:
            continue
        bucket = by_utt.setdefault(idx, {"predicted": [], "revealed": [], "complete": []})
        ev = obj.get("event")
        if ev == "predicted":
            bucket["predicted"].append(obj)
        elif ev == "revealed":
            bucket["revealed"].append(obj)
        elif ev == "utterance_complete":
            bucket["complete"].append(obj)

    # Score each utterance
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
        step_to_pred = {}
        for p in preds:
            s = p.get("step")
            if isinstance(s, int):
                step_to_pred[s] = p.get("prediction") or ""

        best_step = None
        best_U = -1.0

        if show_steps:
            print(f"\n[utt {idx}] ground_truth: {gt_full}")
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

        print(f"[utt {idx}] BEST step={best_step}  U_best={round(best_U,4)}  total_steps={total_steps}")
        dialogue_scores.append(best_U)

    # Dialogue average
    dialogue_avg = (sum(dialogue_scores) / len(dialogue_scores)) if dialogue_scores else 0.0
    print(f"\nDialogue average U (best-early): {round(dialogue_avg, 4)}\n")

# --------------- CLI ---------------

def main():
    p = argparse.ArgumentParser(description="Per-step scorer for phrase_completion (JSONL primary; JSON summary supported)")
    p.add_argument("inputs", nargs="*", help="Paths to .json (summary) and/or .jsonl (log) files")
    p.add_argument("--jsonl", dest="jsonl", action="append", help="One or more JSONL logs to score")
    p.add_argument("--lex-weight", type=float, default=0.5, help="Weight for lexical vs semantic similarity (default 0.5)")
    p.add_argument("--no-steps", action="store_true", help="Hide per-step tables (show only summaries)")
    args = p.parse_args()

    # Collect inputs
    paths: List[Path] = [Path(s) for s in args.inputs]
    if args.jsonl:
        paths += [Path(s) for s in args.jsonl]

    if not paths:
        p.print_help()
        return

    for path in paths:
        if path.suffix.lower() == ".jsonl":
            print(f"\n=== {path} (JSONL per-step) ===")
            score_jsonl(path, lex_weight=args.lex_weight, show_steps=(not args.no_steps))
        elif path.suffix.lower() == ".json":
            print(f"\n=== {path} (JSON summary) ===")
            score_summary_json(path)
        else:
            # Try to sniff file type
            text = path.read_text(encoding="utf-8")
            first = next((c for c in text if not c.isspace()), "")
            if first in "{[":
                print(f"\n=== {path} (JSON summary) ===")
                score_summary_json(path)
            else:
                print(f"\n=== {path} (JSONL per-step) ===")
                score_jsonl(path, lex_weight=args.lex_weight, show_steps=(not args.no_steps))

if __name__ == "__main__":
    # Mirror all prints to a timestamped score file without touching logic
    _log_fh = _open_score_log(sys.argv)
    _orig_print = builtins.print

    def _dual_print(*args, **kwargs):
        end = kwargs.get("end", "\n")
        sep = kwargs.get("sep", " ")
        text = sep.join(str(a) for a in args) + end
        _log_fh.write(text)
        if kwargs.get("flush", False):
            _log_fh.flush()
        _orig_print(*args, **kwargs)

    builtins.print = _dual_print
    try:
        main()
    finally:
        builtins.print = _orig_print
        try:
            _log_fh.close()
        except Exception:
            pass
