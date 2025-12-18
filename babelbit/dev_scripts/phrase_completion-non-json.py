#!/usr/bin/env python3
"""
phrase_completion.py — client that predicts utterances against the engine, with PER-STEP logging.

Primary output: JSONL step log for scoring
  logs/<dialogue-stem>_run_YYYYMMDD_HHMMSS.jsonl

Requires:
  - OPENAI_API_KEY (env)
  - OPENAI_MODEL   (env, e.g., gpt-4o-mini)
Optional:
  - OPENAI_API_BASE (default https://api.openai.com/v1)
  - OPENAI_TEMPERATURE (default 0.4)
  - OPENAI_MAX_TOKENS  (default 48)
"""
# Not needed for Py 3.12++
# from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests

# -------------------------------
# Engine calls
# -------------------------------

def engine_start(engine_url: str) -> Tuple[Optional[str], Optional[str], bool, Optional[int]]:
    r = requests.post(f"{engine_url}/start", timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("end"):
        return None, None, True, None
    return data["session_id"], data["word"], False, data.get("utterance_index")

def engine_next(engine_url: str, session_id: str, prediction: str) -> dict:
    r = requests.post(
        f"{engine_url}/next",
        json={"session_id": session_id, "prediction": prediction},
        timeout=10,
    )
    if r.status_code == 404:
        return {"__error__": "unknown_session"}
    r.raise_for_status()
    return r.json()

# -------------------------------
# OpenAI-backed predictor
# -------------------------------

def _one_line(s: str) -> str:
    return " ".join((s or "").split())

def _ensure_single_eof(s: str) -> str:
    s = _one_line(s)
    s = re.sub(r"(?:\s*EOF\s*)+$", "", s).strip()
    return f"{s} EOF"

def _strip_prefix_if_model_repeated(prefix: str, text: str) -> str:
    text = text.strip()
    if prefix and text.startswith(prefix):
        return text[len(prefix):].lstrip()
    return text

def guess_full_utterance(prefix_text: str, context_utterances: List[str]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _ensure_single_eof(prefix_text)

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.4"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "48"))

    ctx = "\n".join(context_utterances[-4:]) if context_utterances else "(no prior context)"

    system_prompt = (
        "You complete colloquial utterances naturally.\n"
        "Return ONLY the continuation (suffix) for the current utterance.\n"
        "Do NOT repeat the given prefix. Do NOT include quotes or commentary.\n"
        "End with the token 'EOF'."
    )

    user_prompt = (
        f"Recent context:\n{ctx}\n\n"
        f"Prefix tokens:\n{prefix_text}\n\n"
        "Task: Provide ONLY the continuation (suffix), ending with 'EOF'."
    )

    try:
        r = requests.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=20,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        suffix = _one_line(raw)
        suffix = _strip_prefix_if_model_repeated(prefix_text, suffix)
        suffix = _ensure_single_eof(suffix)
        suffix_wo_eof = suffix[:-3].strip() if suffix.endswith("EOF") else suffix
        full = f"{prefix_text} {suffix_wo_eof}".strip() if prefix_text else suffix_wo_eof
        return _ensure_single_eof(full)
    except Exception:
        return _ensure_single_eof(prefix_text)

# -------------------------------
# Logging helpers
# -------------------------------

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def log_jsonl(path: Optional[Path], obj: dict) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Predict utterances against the utterance engine (per-step logging only)")
    ap.add_argument("--engine-url", default="http://localhost:8000", help="Engine base URL")
    ap.add_argument("--dialogue", help="Dialogue file path (used for naming logs only)")
    ap.add_argument("--jsonl", help="JSONL log path (overrides default)")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-step console prints")
    ap.add_argument("--context-utts", type=int, default=4, help="How many prior utterances to pass as context")
    args = ap.parse_args()

    engine_url = args.engine_url.rstrip("/")

    # Determine JSONL log path
    run_id = time.strftime("%Y%m%d_%H%M%S")
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
    else:
        stem = Path(args.dialogue).stem if args.dialogue else "dialogue"
        jsonl_path = Path(f"logs/{stem}_run_{run_id}.jsonl")

    utt_index = 0
    context_memory: List[str] = []

    while True:
        session_id, first_word, end_flag, engine_utt_idx = engine_start(engine_url)
        if end_flag:
            break

        revealed_tokens = [first_word]
        step = 0

        while True:
            prefix_text = " ".join(revealed_tokens)
            prediction = guess_full_utterance(prefix_text, context_memory)

            log_jsonl(jsonl_path, {
                "ts": _now_iso(),
                "event": "predicted",
                "utterance_index": utt_index,
                "step": step,
                "prefix": prefix_text,
                "prediction": prediction,
                "context_used": context_memory[-args.context_utts:],
            })

            if not args.quiet:
                print(f"input -> {prefix_text}")
                print(f"output -> {prediction}")

            resp = engine_next(engine_url, session_id, prediction)
            if resp.get("__error__"):
                break

            next_tok = resp.get("next")
            done = bool(resp.get("done"))

            log_jsonl(jsonl_path, {
                "ts": _now_iso(),
                "event": "revealed",
                "utterance_index": utt_index,
                "step": step,
                "revealed_next": next_tok,
                "done": done,
            })

            if next_tok:
                revealed_tokens.append(next_tok)

            step += 1
            if done:
                gt = " ".join(revealed_tokens)
                log_jsonl(jsonl_path, {
                    "ts": _now_iso(),
                    "event": "utterance_complete",
                    "utterance_index": utt_index,
                    "ground_truth": gt,
                    "final_prediction": prediction,
                })
                summary_gt = gt[:-3].strip() if gt.endswith("EOF") else gt
                context_memory.append(summary_gt)
                utt_index += 1
                break

    print(f"[saved JSONL] {jsonl_path}")

if __name__ == "__main__":
    main()
