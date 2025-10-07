#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# ---------- helpers to read inputs ----------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # skip malformed lines but keep going
                continue
    return rows

def read_challenge(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def dialogue_key(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    # Use UID when present, else index
    return (row.get("dialogue_uid"), row.get("dialogue_index"))

# ---------- conversion: run JSONL -> scorer JSONL per dialogue ----------

def convert_run_to_scorer_jsonl(
    rows: List[Dict[str, Any]],
    challenge: Dict[str, Any],
) -> Dict[Tuple[Optional[str], Optional[int]], List[Dict[str, Any]]]:
    """
    Build, for each dialogue, a JSONL list in the schema that score_dialogue.py expects:

      {"event":"predicted", "utterance_index": i, "step": s, "prediction": "..."}
      {"event":"utterance_complete", "utterance_index": i, "ground_truth": "..."}

    We infer steps from the order of 'prediction' strings in your run JSONL, and we pull ground truth
    from the challenge file by (dialogue_uid || dialogue_index, utterance_index).
    """
    # Map (dlg_uid, dlg_idx) -> list of ground truth utterances
    gt_by_dialogue: Dict[Tuple[Optional[str], Optional[int]], List[str]] = {}

    if "dialogues" in challenge:
        # challenge file with multiple dialogues
        for d_idx, d in enumerate(challenge["dialogues"]):
            uid = d.get("dialogue_uid")
            utts = [str(u) for u in d.get("utterances", [])]
            gt_by_dialogue[(uid, d_idx)] = utts
    else:
        # single-dialogue file (fallback)
        uid = challenge.get("dialogue_uid")
        utts = [str(u) for u in challenge.get("utterances", [])]
        gt_by_dialogue[(uid, 0)] = utts

    # We will accumulate converted events per dialogue
    out: Dict[Tuple[Optional[str], Optional[int]], List[Dict[str, Any]]] = {}

    # Track a running step counter per (dialogue_key, utterance_index)
    step_counter: Dict[Tuple[Tuple[Optional[str], Optional[int]], int], int] = {}

    for row in rows:
        key = dialogue_key(row)
        # Ignore the very last terminal "dialogue" that may only contain a single "EOF EOF / done:true"
        word = row.get("input_word")
        done = bool(row.get("done", False))
        if done and word == "EOF EOF":
            # do not create a new dialogue chunk solely for the terminal marker
            # (if this dialogue already has content, we'll still append; otherwise skip)
            # We'll only append if we have an existing out[key]
            if key not in out:
                continue

        # Initialize output bucket
        out.setdefault(key, [])
        utt_index = row.get("utterance_index")
        if not isinstance(utt_index, int):
            # without utterance index we cannot align; skip
            continue

        # If this row has a prediction (i.e., it's not a boundary-only step), emit a 'predicted' event
        pred = row.get("prediction") or ""
        # We only count steps that accompany actual tokens (non-boundary) leading to a guess.
        if word not in ("EOF", "EOF EOF"):
            # bump step counter for this utterance
            sc_key = (key, utt_index)
            step = step_counter.get(sc_key, 0)
            # emit event
            out[key].append({
                "event": "predicted",
                "utterance_index": utt_index,
                "step": step,
                "prediction": pred,
            })
            step_counter[sc_key] = step + 1

        # On utterance boundary, emit utterance_complete with ground truth
        if word == "EOF":
            # IMPORTANT: The run-log row that contains "EOF" typically carries the *next* utterance_index.
            # We want to close the *previous* utterance here.
            complete_idx = (utt_index - 1) if isinstance(utt_index, int) and utt_index > 0 else 0

            # fetch ground truth for this dialogue
            gt_utts = gt_by_dialogue.get(key)
            # If UID missing in run but present in challenge, we can still map by index
            if gt_utts is None:
                # try match by dialogue_index only (scan keys)
                for (uid2, idx2), lst in gt_by_dialogue.items():
                    if idx2 == key[1]:
                        gt_utts = lst
                        break

            if not gt_utts:
                gt = ""
            else:
                gt = gt_utts[complete_idx] if 0 <= complete_idx < len(gt_utts) else ""

            out[key].append({
                "event": "utterance_complete",
                "utterance_index": complete_idx,
                "ground_truth": gt,
            })

    # Remove any dialogues that ended up with no events (e.g., terminal-only)
    out = {k: v for k, v in out.items() if v}
    return out

# ---------- run the existing single-dialogue scorer per dialogue ----------

def run_per_dialogue_scoring(
    scorer_path: Path,
    dialogue_jsonl: Path,
    pass_through_args: List[str],
) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(scorer_path), "--jsonl", str(dialogue_jsonl)]
    if pass_through_args:
        cmd.extend(pass_through_args)
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def extract_summary_from_stdout(stdout: str) -> Dict[str, Any]:
    """
    Best-effort: look for 'dialogue average U' headline in scorer stdout.
    If your scorer also writes a JSON next to the TXT, prefer reading that file in your own setup.
    """
    summary: Dict[str, Any] = {}
    for line in stdout.splitlines():
        ls = line.strip().lower()
        if "dialogue average u" in ls and ":" in line:
            try:
                val = float(line.split(":")[-1].strip())
                summary["dialogue_average_u_best_early"] = val
            except Exception:
                pass
    return summary

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Score a multi-dialogue run JSONL by converting it for score_dialogue.py and aggregating results.")
    ap.add_argument("--jsonl", required=True, help="Path to the combined run JSONL produced by phrase_completion.py")
    ap.add_argument("--challenge", required=True, help="Path to the challenge JSON that contains ground-truth utterances")
    ap.add_argument("--scorer", default="score_dialogue.py", help="Path to the existing single-dialogue scorer script")
    ap.add_argument("--outdir", default="scores", help="Directory to write the challenge summary")
    ap.add_argument("--scorer-arg", action="append", default=[], help="Extra args to pass to score_dialogue.py (repeatable)")
    args = ap.parse_args()

    run_path = Path(args.jsonl).resolve()
    challenge_path = Path(args.challenge).resolve()
    scorer_path = Path(args.scorer).resolve()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(run_path)
    if not rows:
        print(f"[error] no rows in {run_path}", file=sys.stderr)
        sys.exit(2)

    challenge = read_challenge(challenge_path)
    challenge_uid = challenge.get("challenge_uid")

    # Convert into scorer-friendly JSONL per dialogue
    converted = convert_run_to_scorer_jsonl(rows, challenge)

    if not converted:
        print("[error] nothing to score after conversion (check inputs)", file=sys.stderr)
        sys.exit(3)

    tmpdir = Path(tempfile.mkdtemp(prefix="dlg_splits_"))

    per_dialogue_summaries: List[Dict[str, Any]] = []
    for (dlg_uid, dlg_idx), events in converted.items():
        # Write a temp JSONL for the existing scorer
        stem_uid = dlg_uid if dlg_uid is not None else f"idx{dlg_idx}"
        chunk_path = tmpdir / f"{run_path.stem}__dlg_{stem_uid}.jsonl"
        with chunk_path.open("w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

        # Score it via the existing script
        print(f"[info] scoring dialogue {dlg_uid or dlg_idx} -> {chunk_path.name}")
        cp = run_per_dialogue_scoring(scorer_path, chunk_path, args.scorer_arg)
        if cp.returncode != 0:
            print(f"[warn] scorer returned {cp.returncode} for dialogue {dlg_uid or dlg_idx}", file=sys.stderr)
            if cp.stderr:
                print(cp.stderr, file=sys.stderr)

        # Try to extract a headline number from stdout
        summary = extract_summary_from_stdout(cp.stdout)
        summary.update({
            "dialogue_uid": dlg_uid,
            "dialogue_index": dlg_idx,
        })
        per_dialogue_summaries.append(summary)

    # Aggregate mean
    vals = [d.get("dialogue_average_u_best_early") for d in per_dialogue_summaries if isinstance(d.get("dialogue_average_u_best_early"), (int, float))]
    challenge_mean = float(sum(vals) / len(vals)) if vals else None

    # Write challenge summary JSON
    summary_path = outdir / f"{run_path.stem}-challenge.json"
    result = {
        "run_file": str(run_path),
        "challenge_file": str(challenge_path),
        "challenge_uid": challenge_uid,
        "dialogues": per_dialogue_summaries,
        "challenge_mean_U": challenge_mean,
    }
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote challenge summary -> {summary_path}")
    if challenge_mean is not None:
        print(f"[ok] challenge mean U: {challenge_mean:.6f}")

if __name__ == "__main__":
    main()