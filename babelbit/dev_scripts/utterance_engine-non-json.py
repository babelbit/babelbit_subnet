#!/usr/bin/env python3
"""
utterance_engine.py — minimal HTTP engine that serves one dialogue file.

Endpoints:
  POST /start -> {"end": true} when finished, else {"session_id", "word", "utterance_index"}
  POST /next  -> body: {"session_id": "...", "prediction": "<string>"}
                 returns {"next": "<token or EOF>", "done": bool}
Notes:
- Dialogue file format:
    <utterance 0> EOF
    <utterance 1> EOF
    ...
    EOF EOF           # single line indicating end-of-dialogue
- Tokens are whitespace-split; 'EOF' must be the last token for each utterance.
- This engine does NOT judge the prediction; it only reveals the next token.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from pathlib import Path
from typing import Dict, List, Optional
import uuid

# -------------------------------
# Dialogue loading
# -------------------------------

def load_dialogue(path: Path) -> List[List[str]]:
    """Return a list of utterances; each utterance is a list of tokens (ending with 'EOF')."""
    utterances: List[List[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s == "EOF EOF":
                break
            if not s:
                continue
            toks = s.split()
            if not toks or toks[-1] != "EOF":
                raise ValueError("Every utterance line must end with 'EOF'")
            utterances.append(toks)
    return utterances

# -------------------------------
# Server state
# -------------------------------

class State:
    def __init__(self, utterances: List[List[str]]):
        self.utterances = utterances
        self.next_utt_index = 0
        self.sessions: Dict[str, Dict] = {}

    def start(self) -> dict:
        if self.next_utt_index >= len(self.utterances):
            return {"end": True}
        utt = self.utterances[self.next_utt_index]
        sess_id = str(uuid.uuid4())
        # pointer points to index of the *first* token (already revealed by /start)
        self.sessions[sess_id] = {"utt_index": self.next_utt_index, "ptr": 0}
        first_word = utt[0]
        # Move pointer to after first token, since it's revealed
        self.sessions[sess_id]["ptr"] = 1
        # Increment engine's global "next utterance" only on /next(done=True)
        return {"session_id": sess_id, "word": first_word, "utterance_index": self.next_utt_index}

    def next(self, session_id: str, prediction: str) -> dict:
        sess = self.sessions.get(session_id)
        if not sess:
            return {"__error__": "unknown_session"}
        utt_index = sess["utt_index"]
        utt = self.utterances[utt_index]
        ptr = sess["ptr"]
        if ptr >= len(utt):
            # Safety: already finished
            return {"next": None, "done": True}
        nxt = utt[ptr]
        ptr += 1
        done = ptr >= len(utt)
        sess["ptr"] = ptr
        if done:
            # Advance engine's global index so the next /start moves to the next utterance
            self.next_utt_index = utt_index + 1
            # Free the session
            del self.sessions[session_id]
        return {"next": nxt, "done": done}

# -------------------------------
# HTTP handler
# -------------------------------

class Handler(BaseHTTPRequestHandler):
    state: Optional[State] = None

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            data = json.loads(body or "{}")
        except json.JSONDecodeError:
            data = {}

        if parsed.path == "/start":
            payload = self.state.start()
            self._send_json(200, payload)
            return

        if parsed.path == "/next":
            payload = self.state.next(data.get("session_id", ""), data.get("prediction", ""))
            code = 200 if "__error__" not in payload else 404
            self._send_json(code, payload)
            return

        self._send_json(404, {"error": "not_found"})

    def log_message(self, fmt, *args):
        # Quiet server logs
        return

    def _send_json(self, code: int, obj: dict):
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

def main():
    ap = argparse.ArgumentParser(description="Utterance engine for dialogue files")
    ap.add_argument("--dialogue", required=True, help="Path to dialogue file")
    ap.add_argument("--port", type=int, default=8000, help="Port to serve on (default 8000)")
    args = ap.parse_args()

    utterances = load_dialogue(Path(args.dialogue))
    Handler.state = State(utterances)
    httpd = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[engine] Serving {len(utterances)} utterances on http://localhost:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[engine] Shutting down.")

if __name__ == "__main__":
    main()
