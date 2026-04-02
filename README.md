<p align="center">
  <img width="265" height="281" alt="Babelbit logo Black" src="https://github.com/user-attachments/assets/055577f8-0ff4-4d67-9153-e66c00688bb2" />
</p>

## 1: Our Ambition: An Interpreter's Job Cannot be Performed by Translation Software

In one sense, Babelbit is a Bittensor subnet developing low-latency predictive translation. However, the goal of our product is to behave like a **human interpreter**, which is a significantly different job from translation.

Speech-to-speech translation metrics are focused on making translations faster and more accurate, and usually transform speech into text to achieve this. To get better at this bit by bit, a community like Bittensor probably wouldn't be needed.

So before you get started, we'd like to explain just how different our approach is, and also give you a few strategies for how you might get going. That isn't to say you won't come up with better ones yourselves, but we'd like to make sure that ML experts who haven't worked in this field before don't have to work it all out from scratch.

**See Section 11 at the end of this document**

## 2: What is The Orthodox Approach?

Speech-to-speech translation or simultaneous translation, is usually designed as an extension of the techniques used in text-based translation. The expertise - great though it is - comes from the gold standard in translating books, so it works something like this:

1. Speech-to-Text
2. Text-to-Text Translation
3. Contextual Changes, e.g. replacing an English metaphor with a similar meaning one, like replacing "That's just not cricket" with the German phrase, "Das ist nicht die feine Art". Translating a book and keeping the style, means that a metaphor should be replaced by a metaphor.
4. Text-to-Speech

## 3: What is Babelbit's Approach?
There are **three completely separate techniques** which make Babelbit different from the usual multi-step process. All of these make the user experience better, and all improve latency.

### 3.1: PHRASE PREDICTION
Anyone familiar with our previous challenges will already know that a big part of our latency strategy, is start translating as soon as the meaning is clear. This means that we haveto train models in a different way. We have to train the models to predict when they can, and not to predict when they can't:
"May the force be..." can be translated before the English phrase is finished.
"My favourite poem is..." cannot be translated until hearing the end.

**We hope you'll agree that this is exciting stuff for any machine learning expert. We are not just optimising an off-the-shelf LLM, we are creating new kinds of LLMS.**

### 3.2: ONE-SHOT SPEECH-MODE TRANSLATION
We are already building our base script on a speech-mode language model. That is we are taking a model which tokenises speech, rather than converting it into text, and then training it to generate translations from chunks of meaning comprising tokenised speech in other languages, rather than text.

**When you add this approach to 3.1 above, it is even more exciting, building an innately predictive model, but with predictions which are chunks of speech, not chunks of text**

### 3.3: PARAPHRASING
**The gold standard of interpretion is not being as literal and accurate as possible**. In section 2 above we included an example where a metaphor was replaced by a metaphor. This is close to an art form for good translators, but can be very confusing for listeners to an interpreter, as sometimes even the translation is not in the listeners native language.

Good interpretation is a matter of getting the meaning across in a succinct, polite and culturally sensitive way.

Here are some examples, which will make it clear how different our approach is.
**NOTE: The best interpreters diverge from precise, literal translations of what is said, as follows:**
1. Eliminate accidental repetitions
    1. *I.. I think.. I think that it would be best to finish now.*
    2. *It would be best to end here*
2. Eliminate completely gratuitous expletives
    1. *What a load of f--king nonsense*
    2. *What a load of nonsense*
3. Replace meaningful expletives with polite alternatives
    1. *What a load of shit*
    2. *What a load of rubbish*
4. Paraphrase rambling expressions to make them succinct
    1. *I mean, when you really stop and think about it, it kind of speaks for itself*
    2. *If you think about it, it is obvious*
5. Replace figurative or metaphorical expressions with clear, literal ones
    1. *That's not cricket*
    2. *That's not fair*
6. Be as culturally sensitive as possible
    1. *Muhammad ibn Abdullah was an Abrahamic religious cult leader*
    2. *The Prophet Muhammad, is regarded by Muslims as the final messenger of God*

## 4: A Fairer Approach to Mining Challenges

The text prediction challenge we designed in October 2025, was a task designed to reward miners that make useful predictions early, including predictions that are semantically right before the full utterance is revealed. This allowed us to prove that it was possible to reduce translation latency in a new way.

However, we noticed that some creative approaches to prediction didn't score well, but inspired some good ideas. So it occurred to us that while we still want to reward the biggest performance gains, we don't want any hard-working machine learning engineer to be working for nothing.

So we have come up with a two phase contest - a qualifying round where every contestant gets a proportion of the allotted emissions, and The Arena where the qualifying contestants compete to win the rest.

This is a new evolution of our development, and we will need our mining community to remain ever adaptable with us as we progress - after all we are trying to maximise the performance of the world's first machine interpreter. So this is how we're planning things at launch:

**The Qualifying Round** will share 20% of the emissions between all the contestants (unless they're caught cheating), in proportion to their scores. It probably won't make anyone rich, but our hope is that the hard work will be rewarded in another way - getting better and better - until you qualify for the second phase.

The qualifiers then compete in **The Arena** for a chance at winning the remaining 80%.

## 5: Babelbit Mining Setup

This repository is the operator guide and reference implementation for the Babelbit validator stack. It is primarily for:

- validators running the validator, runner, signer, and supporting services
- miners who need the validator-facing compatibility requirements for qualifying and arena participation

Some submission, scoring, and managed deployment workflows are handled by Babelbit-operated services and are intentionally described here only at a high level.

`qualifying` is round 1. `arena` is round 2.

**This repo isn't intended for mining:** Please refer to the [Babelbit Miner repo](https://github.com/babelbit/babelbit_miner) for further instructions on how to run your miner and submit it to the arena.

### 5.1: What This Repo Covers

Validators use this repo to:

- run the validator process
- run the runner that evaluates miners
- run the signer service
- run the subtensor gateway
- initialise the Postgres schema used for persisted scores

Miners use this repo to:

- understand the validator-facing miner API contract
- understand request-signing and endpoint compatibility requirements
- reference shared environment defaults used by the validator stack

Miner participation is not just "run an axon". Miners also need to actively provide:

- a Docker image
- a Hugging Face repository handle

Those submission artifacts are part of the broader Babelbit participation flow used by Babelbit-operated infrastructure. The private mechanics behind that flow are intentionally not documented here.

### 5.2: Challenge Tiers

The subnet is split into two challenge tiers.

- `Qualifying` receives 20% of the incentive. `Qualifying` miners are rewarded in proportion to their scores.
- `Arena` receives 80% of the incentive. `Arena` is winner-takes-all.

Arena participation is restricted to the best-performing miners from qualifying.

- A miner must have won at least one qualifying challenge in the last 7 days to be arena-eligible.
- The 7-day check uses a rolling window (i.e. 7x24 hours).
- There are currently 7 arena eligibility slots (this number may increase in the future as we continue to evolve).
- Slots are ordered by number of wins and then by score.
- The arena winner spot is awarded on a per challenge basis (this may also evolve in the future to a longer winner's place standing). 

### 5.3: How Participation Works

At a high level, the subnet works like this:

1. Validators retrieve the active challenge and evaluate miners.
2. Qualifying miners are discovered through Bittensor axon metadata.
3. Miners also provide a Docker image and a Hugging Face repository handle for Babelbit-managed participation flows.
4. Validator outputs are submitted to Babelbit-operated services for score aggregation.
5. Arena selection is based on recent qualifying performance and win history.
6. Arena rewards are determined from the current winning position.


## 6: Babelbit Validator Setup

### 6.1: Prerequisites

- A Bittensor wallet and hotkey
- Python 3.10-3.13 if running locally
- Docker if using the recommended deployment path
- (Optional) S3-compatible object storage for logs/artifacts and Postgres database

### 6.2: Install `btcli`

```bash
pip install bittensor-cli
```

### 6.3: Create a wallet

Create a coldkey:

```bash
btcli wallet new_coldkey --n_words 24 --wallet.name my-wallet
```

Create a hotkey:

```bash
btcli wallet new_hotkey --wallet.name my-wallet --n_words 24 --wallet.hotkey my-hotkey
```

### 6.4: Required validator environment

Set the required wallet, network, service, and database settings in `.env`:

```bash
BITTENSOR_WALLET_PATH=~/.bittensor/wallets/my-wallet/hotkeys/my-hotkey
BITTENSOR_WALLET_COLD=my-wallet
BITTENSOR_WALLET_HOT=my-hotkey

BABELBIT_NETUID=59
BITTENSOR_SUBTENSOR_ENDPOINT=finney

SIGNER_URL=http://127.0.0.1:8080
SUBTENSOR_GATEWAY_URL=http://127.0.0.1:8090

BB_UTTERANCE_ENGINE_URL=https://api.babelbit.ai/
BB_SUBMIT_API_URL=https://scoring.babelbit.ai/
BB_ARENA_GATEWAY_URL=https://gw.babelbit.ai/
BB_SUBMIT_TIMEOUT_S=30
BB_MINER_TIMEOUT_SEC=10
```

### 6.5: Optional validator settings

Optional Postgres and S3-compatible storage:

```bash
PG_HOST=your-pg-host
PG_PORT=your-pg-port
PG_DB=your-pg-db-name
PG_USER=your-pg-user
PG_PASSWORD=your-pg-password

BB_ENABLE_S3_UPLOADS=1
S3_ENDPOINT_URL=your-s3-endpoint-url
S3_REGION=s3-region
S3_ACCESS_KEY_ID=your-s3-access-key
S3_SECRET_ACCESS_KEY=your-s3-secret
S3_BUCKET_NAME=your-s3-bucket
S3_SUBMISSIONS_DIR=challenges
S3_LOG_DIR=logs
S3_ADDRESSING_STYLE=path
S3_SIGNATURE_VERSION=s3v4
S3_USE_SSL=true
```

Additional runner, arena, metrics, and scoring settings are available in [`env.example`](./env.example).

### 6.6: Initialize Postgres (Optional)

Initialize the database schema:

```bash
psql -h YOUR_PG_HOST -p YOUR_PG_PORT -U YOUR_PG_USER -d YOUR_PG_DB -f sql/init.sql
```

### 6.7: Local setup

If you want to run outside Docker:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv sync
```

The CLI entrypoint is:

```bash
bb
```

### 6.8: Validator services

The main validator-side commands are:

- `bb runner`: evaluates miners on the active challenge cadence
- `bb validate`: calculates and submits weights
- `bb signer`: runs the signing service used by validator components
- `bb subtensor-gateway`: runs the gateway used by the validator stack

### 6.9: Recommended validator deployment

Run the validator stack with Docker:

```bash
docker compose down
docker compose pull
docker compose up --build -d
docker compose logs -f --tail 100
```

### 6.10: Local validator run

If running locally, make sure the signer URL points to a reachable local signer:

```bash
SIGNER_URL=http://127.0.0.1:8080
```

Then run the services you need:

```bash
bb -vv signer
bb -vv subtensor-gateway
bb -vv runner
bb -vv validate
```

## 7: Miner Setup

### 7.1: Scope

The actual miner implementation lives in an external repository or private codebase. This repository does not currently ship a runnable miner package.

Use your miner repo for:

- process entrypoints
- axon registration scripts
- miner-specific tests
- model-specific decoding and serving configuration

Use this repo for the validator-facing requirements your miner must satisfy.

### 7.2: What miners need to do

Miners participating in Babelbit need to maintain two things:

1. A reachable qualifying miner that validators can discover through Bittensor axon metadata.
2. Submission artifacts for the broader Babelbit flow: a Docker image and a Hugging Face repository handle.

This repo documents the public compatibility requirements. Private submission and managed deployment mechanics are intentionally omitted.

### 7.3: Prerequisites

- Python 3.10-3.13
- Enough RAM or VRAM for the chosen model
- A Bittensor wallet and registered hotkey
- A reachable public IP and port for your axon-compatible miner
- Hugging Face access if your model is gated or private

### 7.4: Shared miner-related environment

The validator stack and related tooling currently reference these shared miner-related settings:

```bash
BITTENSOR_WALLET_PATH=~/.bittensor/wallets/my-wallet/hotkeys/my-hotkey
BITTENSOR_WALLET_COLD=my-wallet
BITTENSOR_WALLET_HOT=my-hotkey
BITTENSOR_SUBTENSOR_ENDPOINT=finney
BABELBIT_NETUID=59

HUGGINGFACE_USERNAME=your-username
HUGGINGFACE_API_KEY=your-api-key

MINER_MODEL_ID=babelbit-ai/base-miner
MINER_AXON_PORT=8091
MINER_DEVICE=cpu
MINER_LOAD_IN_8BIT=0
MINER_LOAD_IN_4BIT=0

# Optional
# MINER_MODEL_REVISION=main
# MINER_EXTERNAL_IP=your-public-ip
```

Model-specific generation knobs belong in the external miner repo, not in this validator stack.

### 7.5: Register on the subnet

Register your hotkey on the subnet:

```bash
btcli subnet register --wallet.name my-wallet --wallet.hotkey my-hotkey --netuid 59
```

### 7.6: Publish axon metadata

Register your axon metadata with whichever tool your miner repo provides. Validators discover qualifying miners from Bittensor axon metadata, so the published IP and port must match the miner endpoint you actually expose.

If your miner is reachable through a different public IP than the host can infer automatically, publish that explicit external IP and port in your axon metadata.

### 7.7: Compatibility requirements

Your external miner implementation should satisfy these validator-facing expectations:

- Register on netuid `59` with a hotkey that is discoverable through Bittensor axon metadata.
- Expose a prediction endpoint at `POST /predict` by default. Validators can be pointed at a different path with `BB_MINER_PREDICT_ENDPOINT`, but `/predict` is the current default.
- Optionally expose `GET /healthz` for operational monitoring.
- Accept Bittensor-style signed request headers. The validator currently sends `bt_header_*` headers such as hotkey, nonce, UUID, signature, axon IP, and axon port.
- Return JSON compatible with the validator schema: `success`, `model`, `utterance`, `context_used`, optional `error`, and `complete`.

### 7.8: Local compatibility testing

If the validator runs in Docker and your miner runs on the host machine, enable validator dev routing so `127.0.0.1` style axon addresses can be translated correctly:

```bash
BB_DEV_MODE=1
BB_LOCAL_MINER_IP=127.0.0.1
```

Run the concrete miner tests from the external miner repository you are using.

## 8: Miner Submission Artifacts

Running a qualifying miner is only one part of participation.

Miners should also maintain:

- a Docker image for their submitted runtime
- a Hugging Face repository handle for the submitted model

These artifacts are used by Babelbit-managed participation flows. This README does not document the private operational details behind those systems, but miners should treat the container image and Hugging Face repo as part of their production submission surface.

## 9: Performance Notes

### 9.1: CPU and Mac usage

- Small models are suitable for CPU testing.
- Large models on CPU will usually be too slow for competitive inference.
- Apple Silicon MPS can work for some models, but compatibility varies.

### 9.2: GPU usage

- Set `MINER_DEVICE=cuda` on NVIDIA systems.
- Use quantization if you need to reduce VRAM pressure.

### 9.3: Quantization

- `MINER_LOAD_IN_8BIT=1` can reduce memory use.
- `MINER_LOAD_IN_4BIT=1` can reduce memory use further.

## 10: Troubleshooting

### 10.1: "Torch not compiled with CUDA enabled"

Use:

```bash
MINER_DEVICE=cpu
```

### 10.2: MPS-related failures on Apple Silicon

If MPS is unstable for your model, switch to CPU:

```bash
MINER_DEVICE=cpu
```

### 10.3: Prediction timeouts

- Your model may be too large for the hardware.
- Reduce model size or enable quantization.
- Test with a smaller model first to confirm the serving path works.

### 10.4: Hugging Face download failures

- Check that the model ID is correct.
- Make sure your Hugging Face token has access if the model is gated.
- Confirm the target revision exists if using `MINER_MODEL_REVISION`.

## 11: What would we Try if we were miners?

### 11.1: Try out different model architectures.

**1: MAYBE THE ORTHODOXY IS RIGHT AFTER ALL**
We made a lot of our one-shot approach above, but who knows? It might be possible to optimise the old-school STT-translate-TTS, and outperform the one-shot version, because that way you'd be building on 40 years of research and optimisation. Each stage can be independently swapped, profiled, and optimised. It's entirely possible that a tightly tuned cascade outperforms a one-shot model, especially early on.

**2: THE MOSHI WAY**
One of our recent starting points:
Audio -> Audio+Semantic tokens +predicted text tokens -> transformer -> audio tokens -> Audio.
This has the advantage of combining very well-established text-prediction with generating speech from tokenised audio.

**3: DISTILLATION OF TRANSFORMERS INTO RECURRENT ARCHITECTURES**
Transformer attention is quadratic in sequence length, which directly impacts latency. A recurrent-style architecture — such as a state-space model (Mamba) or a linear-attention variant (RWKV) — gives you linear-time inference while retaining much of the original model's quality.

Check out the following:
- https://arxiv.org/abs/2603.15569
- https://arxiv.org/abs/2312.00752
- https://github.com/state-spaces/mamba
- https://github.com/BlinkDL/RWKV-LM
- https://arxiv.org/abs/2503.14456

### 11.2: CAUSALITY AND RECEPTIVE FIELD TRICKS

**1: SHIFTING THE RECEPTIVE FIELD TO THE PAST**, no future lookahead (“causality”) - most models, particularly Convnets, assume the receptive field is centred around “now”.  This add architectural latency of 1/2 the receptive field.  Shiftng the receptive field to be fully causal removes this latency, but requires retraining the model.

**2: KV-CACHING** to avoid redundant recomputation.

**3: SPECULATIVE DECODING** combines well with KV-Caching -- Use a smaller, faster draft model to generate multiple candidate tokens in parallel, then verify them against the main model.

**4: SMALLER MODELS** A 12-layer transformer has roughly half the per-step latency of a 24-layer one. Requires balancing reduced accuracy with reduced latency.

### 11.3: DSP tricks

**1: LOWERING THE SAMPLE RATE** speeds up the entire pipeline. How low can you go?

**2: SMALLER HOPS** for the time -> frequency FFT. More computation, however.

**3: USE A CAUSAL NEURAL VOCODER** like a fully-causal HifiGAN which effectively has zero latency (as opposed to an inverse FFT which has a 3X hop-size latency)

### 11.4: Train/Prompt a language model to perform like the best human interpreter
This is one of our favourite approaches, and outlined in section 3. When you're not sure about the limits of AI's capabilities, think about what humans do.

There are basically three ways of pushing a model in this direction:

- **Fine-Tuning on Interpreter-Quality Pairs**:
Curate a dataset of source speech paired with high-quality interpretations (not literal translations), and fine-tune. This can be done with LoRA or with full fine-tuning of some or all layers — though full fine-tuning risks catastrophic forgetting and requires a lot more VRAM (all tensors must track gradients). Or if you're really constrained, use prompt tuning (learn embeddings for a few soft tokens)

- **2: Prompt Engineering**:
This is the cheapest experiment to run, and a good starting point before committing to fine-tuning.

- **3: Full Training from Scratch**:
Starting from an uninitialised model and training end-to-end on the speech-to-speech task (as in Google's Translatotron lineage), or on a hybrid objective that conditions on LM-generated "inner monologue" text (closer to the Hibiki approach). This give you the most architectural freedom.

