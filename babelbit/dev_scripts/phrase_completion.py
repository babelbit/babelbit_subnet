#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

# ---------------
# Engine helpers
# ---------------

def engine_start(engine_url: str) -> Tuple[str, str, bool, Optional[int], Dict[str, Any]]:
    """
    POST /start with no body. The engine chooses the file (via --challenge or path from server).
    Returns: (session_id, first_word, done_flag, utterance_index, full_response_json)
    """
    r = requests.post(f"{engine_url}/start", timeout=10)
    r.raise_for_status()
    data = r.json()
    # Back-compat: engine returns both 'token' and 'word'
    first_word = data.get("word", data.get("token"))
    return data["session_id"], first_word, bool(data.get("done", False)), data.get("utterance_index"), data


def engine_next(engine_url: str, session_id: str, prediction: str) -> Dict[str, Any]:
    """
    POST /next with the last prediction. Returns full JSON from engine.
    """
    r = requests.post(
        f"{engine_url}/next",
        json={"session_id": session_id, "prediction": prediction},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# ---------------
# LLM helper
# ---------------

def guess_full_utterance(
    api_base: str,
    api_key: Optional[str],
    model: str,
    temperature: float,
    prefix_text: str,
    context_memory: str = "",
) -> str:
    """
    Very simple utterance completion call to a /v1/chat/completions-compatible endpoint.
    Returns a single-string completion (no trailing EOFs).
    """
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Keep prompt simple; this mirrors the original intent (predict the rest of the utterance).
    system_msg = (
        "You are a helpful assistant that completes the current utterance naturally and succinctly. "
        "Return only the completed utterance text without quotes or extra commentary."
    )
    user_msg = f"Context:\n{context_memory}\n\nContinue the utterance that begins with:\n{prefix_text}"

    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    }

    resp = requests.post(url, headers=headers, json=body, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    # OpenAI-style shape
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        # fall back to some string if the response shape is unexpected
        return ""


# ---------------
# Logging helpers
# ---------------

def now_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def open_run_log(log_dir: Path) -> Tuple[Path, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"challenge_run_{now_ts()}.jsonl"
    f = path.open("a", encoding="utf-8")
    return path, f


def write_jsonl(fh, obj: Dict[str, Any]) -> None:
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fh.flush()


# ---------------
# Main
# ---------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Phrase completion client for utterance_engine.")
    ap.add_argument("--engine-url", required=True, help="Base URL for the running utterance engine, e.g. http://127.0.0.1:8999")
    # Kept for back-compat; ignored now (engine owns the data selection)
    ap.add_argument("--dialogue", default=None, help="(Ignored) Historic path argument; engine decides the input now.")
    ap.add_argument("--logdir", default="logs", help="Directory to write JSONL logs")
    ap.add_argument("--openai-api-base", default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"))
    ap.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--temperature", type=float, default=float(os.environ.get("OPENAI_TEMPERATURE", "0.2")))
    args = ap.parse_args()

    engine_url = args.engine_url.rstrip("/")

    # Open log
    log_path, log_fh = open_run_log(Path(args.logdir))
    print(f"[logging to] {log_path}")

    # One start; then /next until done: true
    session_id, word, done, engine_utt_idx, resp = engine_start(engine_url)

    # optional UIDs in the very first response
    challenge_uid = resp.get("challenge_uid")
    dialogue_uid = resp.get("dialogue_uid")

    # State used to build each utterance prefix and optional context memory.
    prefix_text = ""   # the visible growing prefix for the current utterance
    context_memory = ""  # light context string we pass to the LLM (can be improved later)

    # Process the very first token as input → guess output
    # (Loop only stops if the engine says done: true)
    while True:
        if done:
            # terminal state from engine
            break

        # Engine always returns a 'word' (alias of 'token') or None at terminal
        if word is None:
            # nothing more to do; engine will mark done on next call
            resp = engine_next(engine_url, session_id, prediction="")
            done = bool(resp.get("done", False))
            if done:
                break
            word = resp.get("word", resp.get("token"))
            continue

        # Print input token
        print(f"input -> {word}")

        # Accumulate prefix unless this is a boundary token
        if word not in ("EOF", "EOF EOF"):
            prefix_text = f"{prefix_text + ' ' if prefix_text else ''}{word}"

        # Decide whether to call LLM now. The classic behavior is: after any token, attempt to
        # produce the rest of the utterance.
        if word in ("EOF", "EOF EOF"):
            # When we hit a boundary, finalize and do NOT generate a prediction.
            prediction = ""
            if word == "EOF":
                # end of utterance → reset prefix, keep context up to you
                prefix_text = ""
            elif word == "EOF EOF":
                # end of dialogue → reset both prefix and (optionally) context
                prefix_text = ""
                context_memory = ""  # keep simple; depends on your design
        else:
            # Generate a completion for the current utterance-in-progress
            try:
                prediction = guess_full_utterance(
                    api_base=args.openai_api_base,
                    api_key=args.openai_api_key,
                    model=args.model,
                    temperature=args.temperature,
                    prefix_text=prefix_text,
                    context_memory=context_memory,
                )
            except Exception as e:
                # Fail soft: still step the engine to avoid wedging the session
                prediction = ""
                print(f"[warn] LLM call failed: {e}", file=sys.stderr)

            # Human-readable output line (as before)
            if prediction:
                print(f"output -> {prediction} EOF")
            else:
                print("output ->  EOF")

        # Send prediction to engine, get the next token step
        resp = engine_next(engine_url, session_id, prediction)

        # Add UIDs (if provided by engine) for downstream scoring; keep other fields intact
        step_log = {
            "ts": time.time(),
            "event": "step",
            "engine_url": engine_url,
            "session_id": session_id,
            "input_word": word,
            "prediction": prediction,
            "done": bool(resp.get("done", False)),
            "challenge_uid": resp.get("challenge_uid", challenge_uid),
            "dialogue_uid": resp.get("dialogue_uid", dialogue_uid),
            "dialogue_index": resp.get("dialogue_index"),
            "utterance_index": resp.get("utterance_index"),
            "token_index": resp.get("token_index"),
        }
        write_jsonl(log_fh, step_log)

        # Update loop controls from response
        done = bool(resp.get("done", False))
        word = resp.get("word", resp.get("token"))
        # Update cached UIDs in case engine changes dialogues
        challenge_uid = resp.get("challenge_uid", challenge_uid)
        dialogue_uid = resp.get("dialogue_uid", dialogue_uid)

        # If we just crossed an utterance boundary (EOF), keep simple: accumulate a tiny rolling context
        # You can change this to suit your strategy without affecting engine protocol.
        if word == "EOF":
            # After we *receive* EOF as the next token, nothing else to do here
            pass
        elif word == "EOF EOF":
            # After we *receive* dialogue boundary, keep going — DO NOT restart
            pass

    log_fh.close()
    print("[done] engine returned done: true; run completed.")


if __name__ == "__main__":
    main()
