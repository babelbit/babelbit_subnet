# Utterance Completion Mining Tools
## Background
The tools here include a basic script **phrase_completion.py** to predict the remainder of a phrase as each word is received providing greater context, e.g.
**ground truth:** Welcome to Babelbit's mining repo
**word 1:** Welcome **prefix** Welcome **prediction** campers!
**word 2:** to ** **prefix** Welcome to **prediction** Disney World
**word 3:** Babelbit's **prefix** Welcome to Babelbit's **prediction** new website
**word 4:** mining **prefix** Welcome to Babelbit's **prediction** mining extravaganza
**word 5:** Welcome to Babelbit's mining repo EOF

It is not very good at it. This is what you have to improve. There is an interesting nuance. Since this predictive power will ultimately be used for doing faster translations, it doesn't matter if you get the words wrong, if the meaning of the entire phrase is correct. So "hi there people" and "hello guests" might score higher than you think. This depends on the context of the dialogue. 

## What each script does

* **utterance\_engine.py** — serves up each utterance one token at a time, for a *challenge* (which is one or many dialogues or conversations) over HTTP.
* **phrase\_completion.py** — the miner/client. Reads tokens from the engine, predicts utterance completions, and writes a JSONL run log.
* **score\_dialogue.py** — scores a **single dialogue** JSONL (human-readable TXT + machine-readable JSON per dialogue). This includes both semantic and lexical scores. 
* **score\_challenge.py** — converts a **full run JSONL** into per-dialogue scorer inputs, calls `score_dialogue.py` on each, and writes a challenge summary JSON.

---

# Input formats

## Challenge (engine input)

JSON with a challenge UID and a list of dialogues. Each dialogue has a UID and an array of ground-truth utterances (strings). Example:&#x20;

```json
{
  "challenge_uid": "ch-001",
  "dialogues": [
    {
      "dialogue_uid": "dlg-002",
      "utterances": [
        "it's friday lets finish early",
        "too much to do i'm afraid",
        "you're no fun"
      ]
    },
    {
      "dialogue_uid": "dlg-003",
      "utterances": [
        "welcome to our conference",
        "thank you",
        "please go ahead with your talk",
        "today i will talk about talking"
      ]
    }
  ]
}
```

## Run log (client output; input to challenge scorer)

Written by `phrase_completion.py` as **JSON Lines** (one object per engine step). Key fields:

* `input_word`: token from engine (including `"EOF"` and `"EOF EOF"` delimiters)
* `prediction`: your completion for the current utterance (empty on boundaries)
* `done`: true on the final step of the **whole** challenge
* `challenge_uid`, `dialogue_uid`, `dialogue_index`, `utterance_index`, `token_index`

File naming: `logs/challenge_run_<YYYYMMDD>_<HHMMSS>.jsonl`

---

# Output formats

## Per-dialogue (from score\_dialogue.py)

* **Human-readable TXT** (per dialogue): shows ground truth, each step’s lexical/semantic/earliness, `U_step`, best step, and a final “Dialogue average U (best-early)” line. Example:  and&#x20;
* **Machine-readable JSON** (per dialogue): mirrors the same content, per utterance/per step + `dialogue_summary.average_U_best_early`. Examples:  turn5file3

## Per-challenge (from score\_challenge.py)

* **Challenge summary JSON** with:

  * `challenge_uid`
  * `dialogues`: array of `{dialogue_uid, dialogue_index, dialogue_average_u_best_early}`
  * `challenge_mean_U` (mean across dialogues)
    Example:&#x20;

---

# Environment variables

## phrase\_completion.py (LLM)

* `OPENAI_API_BASE` (default: `https://api.openai.com/v1`)
* `OPENAI_API_KEY` (if your endpoint requires a Bearer token)
* `OPENAI_MODEL` (default: `gpt-4o-mini`)
* `OPENAI_TEMPERATURE` (default: `0.2`)

All of these can also be overridden via CLI flags (see below).

## utterance\_engine.py / score\_dialogue.py / score\_challenge.py

* No required env vars.

---

# Required / useful arguments

## utterance\_engine.py

```
--host           (default 127.0.0.1)
--port           (default 8999)
--challenge PATH (path to challenge JSON; used if client’s /start doesn’t provide a path)
```

## phrase\_completion.py

```
--engine-url           (required) e.g. http://127.0.0.1:8999
--logdir               (default logs)
--openai-api-base      (or $OPENAI_API_BASE)
--openai-api-key       (or $OPENAI_API_KEY)
--model                (or $OPENAI_MODEL)
--temperature          (or $OPENAI_TEMPERATURE)
```

Notes:

* The client **does not** read the challenge file (that’s intentional). It streams tokens from the engine and writes a single run JSONL spanning all dialogues in the challenge.
* The client prints human-readable `input -> ...` and `output -> ... EOF` to the console; the JSONL is what scoring uses.

## score\_dialogue.py (single dialogue)

```
--jsonl PATH   (per-dialogue JSONL in the scorer’s expected schema)
# (Other flags same as your current script; keep using them as before.)
```

## score\_challenge.py (multi-dialogue aggregation)

```
--jsonl      PATH  (the run file produced by phrase_completion.py)
--challenge  PATH  (the same challenge JSON the engine used)
--scorer     PATH  (path to score_dialogue.py; default: score_dialogue.py)
--outdir     DIR   (where to write the challenge summary; default: scores)
--scorer-arg ARG   (optional; repeat to pass extra flags through to score_dialogue.py)
```

---

# End-to-end: multi-dialogue challenge

1. **Start the engine** (serves the whole challenge):

```bash
python3 utterance_engine.py \
  --host 127.0.0.1 \
  --port 8999 \
  --challenge /abs/path/to/challenge-001.json
```

2. **Run the client** (writes one run JSONL across all dialogues):

```bash
# Optional: export LLM config
export OPENAI_API_BASE="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-..."               # if needed
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_TEMPERATURE="0.2"

python3 phrase_completion.py \
  --engine-url http://127.0.0.1:8999
# -> logs/challenge_run_<timestamp>.jsonl
```

3. **Score the full challenge** (convert + per-dialogue scoring + summary):

```bash
python3 score_challenge.py \
  --jsonl logs/challenge_run_20250924_140626.jsonl \
  --challenge data/challenge-001.json \
  --scorer score_dialogue.py \
  --outdir scores
```

You should see:

* Per-dialogue TXT & JSON files from `score_dialogue.py` (examples show step tables and best-early metrics).  and
* A **challenge summary** JSON with mean U across dialogues.&#x20;

---

## Notes & tips

* **Dialogue boundaries**: the engine emits `"EOF"` at the end of each utterance and `"EOF EOF"` at the end of each dialogue; `phrase_completion.py` no longer restarts on `"EOF EOF"`, so a single run naturally spans all dialogues.
* **Per-dialogue scoring schema**: the scorer’s per-dialogue JSON reflects file path, dialogue UID (if known), each utterance’s best step, and `average_U_best_early` at the bottom (TXT) / in `dialogue_summary` (JSON). Examples show exactly the structure you’ll get.
* **Challenge summary**: compact roll-up—just UIDs, each dialogue’s average, and the overall mean.&#x20;
