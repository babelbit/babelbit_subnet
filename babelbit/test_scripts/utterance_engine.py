#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import uuid

# Path to a challenge JSON file provided via CLI (used if /start has no 'path')
CHALLENGE_PATH: Optional[Path] = None

# ----------------------
# Data loading
# ----------------------

class LoadedChallenge:
    """Unified in-memory view:
       - challenge_uid: str | None
       - dialogues: List[Tuple[dialogue_uid:str|None, utterances:List[str]]]
    """
    def __init__(self, challenge_uid: Optional[str], dialogues: List[Tuple[Optional[str], List[str]]]):
        self.challenge_uid = challenge_uid
        self.dialogues = dialogues

def _tokenize(u: str) -> List[str]:
    # Keep it simple and stable: whitespace tokenization
    u = u.strip()
    return u.split() if u else []

def load_input(path: Path) -> LoadedChallenge:
    """
    Autodetect input type:
      - Challenge JSON: { challenge_uid, dialogues: [ {dialogue_uid, utterances:[...]} ...] }
      - Dialogue JSON:  { dialogue_uid, utterances:[...] }
      - Plaintext: one utterance per non-empty line (legacy)
    """
    text = path.read_text(encoding="utf-8")
    # Try JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            if "challenge_uid" in obj and "dialogues" in obj:
                ch_uid = str(obj["challenge_uid"])
                dialogues: List[Tuple[Optional[str], List[str]]] = []
                for d in obj["dialogues"]:
                    dlg_uid = str(d.get("dialogue_uid", "")) if "dialogue_uid" in d else None
                    utts = [str(u) for u in d.get("utterances", [])]
                    dialogues.append((dlg_uid, utts))
                return LoadedChallenge(ch_uid, dialogues)
            if "dialogue_uid" in obj and "utterances" in obj:
                dlg_uid = str(obj["dialogue_uid"])
                utts = [str(u) for u in obj.get("utterances", [])]
                return LoadedChallenge(
                    challenge_uid=None,
                    dialogues=[(dlg_uid, utts)]
                )
    except Exception:
        pass

    # Fallback: plaintext lines (legacy)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return LoadedChallenge(challenge_uid=None, dialogues=[(None, lines)])

# ----------------------
# Session state
# ----------------------

class Session:
    def __init__(self, lc: LoadedChallenge):
        self.challenge_uid = lc.challenge_uid
        self.dialogues = lc.dialogues  # List[(dialogue_uid, [utterances])]
        # Position
        self.di = 0  # dialogue index
        self.ui = 0  # utterance index
        self.ti = 0  # token index within utterance
        # Pre-tokenize all utterances
        self.tokens: List[List[List[str]]] = []
        for _dlg_uid, utts in self.dialogues:
            self.tokens.append([_tokenize(u) for u in utts])

        # Synthesized boundary emission
        self.pending_emit: Optional[str] = None           # 'EOF' or 'EOF_EOF'
        self.pending_advance_utterance: bool = False
        self.pending_advance_dialogue: bool = False

    def _current(self) -> Tuple[Optional[str], List[str]]:
        dlg_uid, utts = self.dialogues[self.di]
        return dlg_uid, utts

    def _advance_token(self) -> None:
        self.ti += 1

    def _advance_utterance(self) -> None:
        self.ui += 1
        self.ti = 0

    def _advance_dialogue(self) -> None:
        self.di += 1
        self.ui = 0
        self.ti = 0

    def _in_bounds(self) -> bool:
        return self.di < len(self.dialogues)

    def _utterance_in_bounds(self) -> bool:
        return self._in_bounds() and self.ui < len(self.dialogues[self.di][1])

    def _tokens_in_bounds(self) -> bool:
        return self._utterance_in_bounds() and self.ti < len(self.tokens[self.di][self.ui])

    def snapshot(self) -> Dict[str, Any]:
        # SAFE at end: don't index if we're past the last dialogue
        if self._in_bounds():
            dlg_uid, _utts = self._current()
        else:
            dlg_uid = None
        return {
            "challenge_uid": self.challenge_uid,
            "dialogue_index": self.di,
            "dialogue_uid": dlg_uid,
            "utterance_index": self.ui,
            "token_index": self.ti
        }

    def step(self) -> Dict[str, Any]:
        """
        Advance by one token. If utterance ends, emit 'EOF' then advance.
        If dialogue ends, emit 'EOF EOF' then advance.
        Return the next token (or done).
        """
        # Apply any pending structural advances (scheduled after a boundary emission)
        if self.pending_advance_dialogue:
            self._advance_dialogue()
            self.pending_advance_dialogue = False

        if self.pending_advance_utterance:
            self._advance_utterance()
            self.pending_advance_utterance = False
            # After advancing utterance, if we've exhausted this dialogue, schedule EOF_EOF
            if self._in_bounds() and not self._utterance_in_bounds():
                self.pending_emit = "EOF_EOF"
                self.pending_advance_dialogue = True

        # If we're completely done, return terminal snapshot
        if not self._in_bounds():
            return {"done": True, **self.snapshot(), "token": None}

        # If a synthetic boundary token is due, emit it now
        if self.pending_emit == "EOF_EOF":
            self.pending_emit = None
            return {"done": False, **self.snapshot(), "token": "EOF EOF"}

        if self.pending_emit == "EOF":
            self.pending_emit = None
            # After emitting EOF, we advance the utterance on the next turn
            self.pending_advance_utterance = True
            return {"done": False, **self.snapshot(), "token": "EOF"}

        # If we are at an utterance with no tokens (empty string), emit EOF immediately
        if not self._tokens_in_bounds():
            # Emit EOF for legacy compatibility, then move to next utterance on following call
            self.pending_advance_utterance = True
            return {"done": False, **self.snapshot(), "token": "EOF"}

        # Normal token path
        tok = self.tokens[self.di][self.ui][self.ti]
        self._advance_token()

        # If we just consumed the last token of the utterance, schedule EOF for next call
        if not self._tokens_in_bounds():
            self.pending_emit = "EOF"

        return {"done": False, **self.snapshot(), "token": tok}

# ----------------------
# HTTP API
# ----------------------

_SESSIONS: Dict[str, Session] = {}

class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: Dict[str, Any]) -> None:
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
    #    self.send_header("Access-Control-Allow-Origin", "*")  # enable if needed later
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            return self._json(400, {"error": "invalid JSON"})

        if self.path == "/start":
            return self._start(body)
        if self.path == "/next":
            return self._next(body)

        return self._json(404, {"error": "unknown endpoint"})

    def _start(self, body: Dict[str, Any]) -> None:
        """
        POST /start
        body: { "path": "path/to/input.(json|txt)" }  # optional if --challenge is provided
        returns: { session_id, token, word, done, challenge_uid, dialogue_index, dialogue_uid, utterance_index, token_index }
        """
        p = body.get("path")
        if p:
            path = Path(p)
        else:
            # Fallback to CLI-provided --challenge path if no body/path was sent
            global CHALLENGE_PATH
            if CHALLENGE_PATH is None:
                return self._json(400, {"error": "missing 'path' and no --challenge set on server"})
            path = CHALLENGE_PATH

        if not Path(path).exists():
            return self._json(400, {"error": f"file not found: {path}"})

        lc = load_input(Path(path))
        sess = Session(lc)
        sid = str(uuid.uuid4())
        _SESSIONS[sid] = sess

        # Emit first item using the same path as /next
        nx = sess.step()
        if "token" in nx and nx["token"] is not None:
            nx = {**nx, "word": nx["token"]}
        return self._json(200, {"session_id": sid, **nx})

    def _next(self, body: Dict[str, Any]) -> None:
        """
        POST /next
        body: { "session_id": "...", "prediction": "..." }
        returns: { token, word, done, challenge_uid, dialogue_index, dialogue_uid, utterance_index, token_index }
        """
        sid = body.get("session_id")
        if not sid or sid not in _SESSIONS:
            return self._json(400, {"error": "invalid or missing session_id"})
        # prediction is currently ignored, but we accept it to keep contract stable
        sess = _SESSIONS[sid]

        nx = sess.step()
        if "token" in nx and nx["token"] is not None:
            nx = {**nx, "word": nx["token"]}
        return self._json(200, nx)

# ----------------------
# CLI bootstrap
# ----------------------

def serve(host: str, port: int):
    # Allow immediate rebinding of the same port after quick restarts
    HTTPServer.allow_reuse_address = True
    server = HTTPServer((host, port), Handler)
    print(f"utterance_engine listening on http://{host}:{port}")
    server.serve_forever()

def main():
    global CHALLENGE_PATH
    ap = argparse.ArgumentParser(description="Serve dialogue/challenge tokens one at a time over HTTP.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8999)
    # Parse a challenge path (used when /start doesn't include 'path')
    ap.add_argument("--challenge", type=str, default=None,
                    help="Path to a challenge JSON file (used if /start has no 'path').")
    args = ap.parse_args()

    CHALLENGE_PATH = Path(args.challenge).expanduser().resolve() if args.challenge else None
    serve(args.host, args.port)

if __name__ == "__main__":
    main()
