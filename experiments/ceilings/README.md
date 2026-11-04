# ceilings: testing the Coffee Law's three ceilings

An experiment harness for the reviewer's request on *The Coffee Law* (Bhatt, 2026): separate the
**information**, **reasoning** and **communication** ceilings of agent teams under matched
resource budgets.

- Design: [specs/001-ceiling-discrimination-experiments/](specs/001-ceiling-discrimination-experiments/) (spec, plan, research, data model, contracts, tasks)
- Background: [litreview-coffee-law.md](litreview-coffee-law.md) (an explainer of the paper)
- Paper and formal proofs: [../../paper/](../../paper/), [../../formal_proofs/](../../formal_proofs/)
- Notebook: [notebooks/ceilings_small_run.ipynb](notebooks/ceilings_small_run.ipynb)
- Local test-run findings: [TESTING.md](TESTING.md)
- Free-tier pilot on NVIDIA build (Oct 2026): [PILOT_RESULTS.md](PILOT_RESULTS.md)
- Paid run on OpenRouter, ten models (Oct 2026): [RESULTS.md](RESULTS.md)

The fork is public: <https://github.com/Ishan-K-S/coffeelaw>; this project lives in
`experiments/ceilings/`. The rest of the repository is the upstream paper code from
`mbhatt1/coffeelaw`.

## Install (Python 3.10+)

```bash
pip install -e ".[dev,notebook]"
pytest -q        # no model calls: theory, contracts, generators, statistics, verdicts, end to end
```

## Models

| `--model` | What it is | Setup |
|---|---|---|
| `sim-bounded` | simulated agent with known ceilings; free, validates the pipeline only | none |
| `claude-<model>` (e.g. `claude-opus-5-5`) | Anthropic API; native `effort` sets depth | `ANTHROPIC_API_KEY` |
| `ollama/<model>` (e.g. `ollama/qwen2.5:7b`) | local open-weights model via Ollama | `OLLAMA_HOST` (default localhost) |
| `openai/<model>` | any OpenAI-style `/chat/completions` server: hosted APIs, vLLM, llama.cpp | `CEILINGS_OPENAI_BASE_URL`, `CEILINGS_OPENAI_API_KEY`, optional `CEILINGS_OPENAI_PRICE_IN/OUT` |

### Hosted OpenAI-style providers (NVIDIA build)

`scripts/nvidia_env.sh <name> <command...>` runs a command in a capped `ceilings:dev` container
against NVIDIA build's free endpoints. It reads the key from `~/.nvidia_api_key` (override with
`KEY_FILE`) and passes every variable to Docker by name only, so the key never appears in a
command line, log or file in the repository. Per-model settings live in
[configs/nvidia_build.json](configs/nvidia_build.json) (`CEILINGS_OPENAI_PROFILES`): how to switch
off a model's built-in reasoning, answer-token caps, and any served-name alias. The profile is
stored in every run record.

- `CEILINGS_OPENAI_RPM` spaces requests within one process; free tiers count requests, not
  dollars, so `--max-cost` does not stop a run there. Use `--max-requests N` (exit code 4) or
  `--limit`.
- 429 and 5xx replies are retried with backoff (`CEILINGS_OPENAI_RETRIES`, default 8), honouring
  `Retry-After`.
- Hidden reasoning (`<think>` blocks or a `reasoning_content` field) is removed before parsing,
  kept in the transcript, and charged through `completion_tokens`.
- Servers that ignore `response_format` still answer: every answer turn states the JSON format,
  and a reply that discusses several options without choosing one is scored as a parse failure
  rather than guessed from.
- `python scripts/doctor.py --model <m>` runs one instance per family (baseline and full fix)
  and prints parse methods, validity, hidden reasoning, truncation, tokens and requests.

### Ablations (`--harness`)

Design choices the ablations vary are harness options, held fixed within one `--root`:
`probe=agent` (the model picks probes instead of rotation), `sender=encode` (agent B gets no
ready-made message), `schema=answer_first`, `shuffle=off`, `temp=0`, and
`reason=0:128:512:1024:2048` (reasoning caps low..max). Delta levels need no option (`--delta`).
Non-default options are part of every run ID and every run record stores the resolved set
(`harness`), so ablation runs can never be mistaken for main-design runs. See
[src/ceilings/options.py](src/ceilings/options.py).

### Sample size

`ceilings power` reads pilot runs and gives the per-cell n each pre-registered comparison needs
(TOST at the margin under a true difference of 0, exact McNemar at the pilot gain), planned at
Bonferroni alpha within each Holm family. Put the result in `prereg/config.json:
sample_sizes.per_cell` before freezing.

### Design changes after the October 2026 pilot

See [PILOT_RESULTS.md](PILOT_RESULTS.md). True answers are balanced across instances; the
information family lists its host table in a random order per instance and can state `known`
attributes up front (`--difficulty k=3,p=0,known=1`); the communication families reason at
`medium` depth in every arm; K's strength can put reasoning on some samples only (`medium:1`); the
single-agent A2 reference is matched by its own reasoning depth; `prepare` confirms the gate with
an equivalence test (`--gate confirm`, the default; about 250+ pilot instances per family); ties
are broken with the run's seed; run records carry `p_true`; H6 uses the full Fano bound; and each
ablation must be cost-matched under its own `--harness` options.

### Design changes after the October 2026 paid run (arm design 2)

The paid run ([RESULTS.md](RESULTS.md)) was made with arm design 1. Checking every test against
its run records found arms that did not test what they were meant to; design 2 fixes them.

- **K** (H2): `medium:1` meant one of the four samples reasoned, and the three that did not
  out-voted it (reasoning sample right 90-100%, team vote 40-57%). The team now answers with its
  reasoning samples only: `high:2` is two samples, both reasoning.
- **M on reason** (H2): both agents hold every fact, so a free-text message let B work the chain
  and send the result, which is compute, not communication. M's channel there now delivers only
  lines copied word for word from the task, in the task's order (`message_mode="quote"`).
- **A2** (H5): cost matching chose depth `low` for A2 on 8 of 9 reason models (four agents with no
  reasoning, relaying the first agent's guess), and on comm its 0-bit channel meant B, C and D
  never made a call. A2's agents now always reason, relay through a 200-token channel on every
  family (on comm without B, whose notes would carry M's information), and the single-agent
  reference is matched to A2's realized cost rather than the level's target.
- **Instances**: a setting with too few distinct prompts now fails instead of repeating them
  (Gemma info at k=2 had 96 distinct prompts behind 850 instances).
- **Answer cap**: the answer turn allows 60 tokens per option (16 hosts: 960), up from a flat 400
  that truncated up to 46% of a cell's H6 answers.

Two independent reviews of design 2 left three things to state in its pre-registration, which
[prereg/DESIGN2_PLAN.md](prereg/DESIGN2_PLAN.md) drafts along with the rerun's scope:
- K now also cuts the sample count, which is conservative for H2.
- M on reason is a null by construction (a consistency check).
- Four reasoning agents cost about 2-2.7× the medium target, so A2 is not matched to a level
  and H5 is tested on reason only, at A2's realized cost.

`ceilings.arms.ARM_DESIGN` is part of every new run ID and run record, and `ceilings run`
refuses to add design-2 runs to a root that holds design-1 runs. Use a new `--root` and a new
pre-registration for a design-2 run.

Not changed, because no harness change can fix them: H6 and H7 test information-theoretic bounds
that no honest team can beat, and under the fixed message format (B sends the first L bits) H7's
flat-then-rising shape is set by the format. They can only fail through a leak or a bug, so they
are consistency checks rather than tests that could have gone either way.

For models without a native effort setting, depth is a reasoning stage capped at
`low 0 / medium 512 / high 1024 / xhigh 2048 / max 4096` tokens (override with
`CEILINGS_REASONING_TOKENS`). Keep at least about 20 tokens per reasoning step the task needs.
Temperature is sent explicitly (`CEILINGS_TEMPERATURE`, default 0.7). Claude keeps refusal fallbacks
**off**, so no model substitution can occur, and every adapter marks a run invalid if the provider
serves a different model.

### Agent teams (arm design 3, not yet run)

Design 2's A2 was a fixed relay of one-call agents, and on weaker models the final agent copied
the wrong guesses it was sent. Design 3 tests H5 on teams that work the way deployed agent systems
do (`ceilings.teams`; plan and pre-registration draft in
[prereg/DESIGN3_PLAN.md](prereg/DESIGN3_PLAN.md)):

- Every agent runs a loop of turns: it thinks (a reasoning stage at medium depth), then takes one
  action written as JSON, and the harness shows the result in its next turn.
- **S** is one agent alone. **L** is a lead that must hand parts of the work to 2-3 workers, which
  see the task and their brief, and the lead sees only their reports. **W** is 4 peers sharing a
  notes board and voting. **W0** (optional) is W without the board. **R** is an author and a
  reviewer.
- All arms share one team budget of total tokens, checked before each turn. An agent whose turns
  or budget run out is asked for its answer at once, and the run is flagged as a forced answer.
- The task is the reason family with long chains cut into 3 sections in chain order
  (`layout=sections`, split `design3`), so the work can be divided by section.
- `ceilings check` also verifies that every agent's first prompt holds every fact, that no agent
  sees another's reasoning unless it was sent, and that every turn, budget and text cap holds.

```bash
ceilings --root data/design3 team-run --model sim-bounded --d 12 --n 20 --budget 20000
```

`--budget 0` runs without a token cap (the chain-length pilot). Analysis of the team arms
(H5a-H5c) is not written yet.

## A real run, in order

```bash
M=ollama/qwen2.5:7b      # or claude-opus-5-5, openai/<model>
ceilings calibrate --model $M --concurrency 4                 # per-model difficulty search
ceilings prepare   --model $M --n-pilot 200 --n-main 850 --concurrency 4   # gate + cost matching
ceilings prereg freeze                                        # fill prereg/config.json first
ceilings run       --model $M --concurrency 4 --max-cost 50 --require-frozen
ceilings sweep     --model $M --concurrency 4
ceilings check     --model $M     # did the runs go as designed? (integrity, quality, expectations)
ceilings analyze                  # verdicts; needs >= 2 evaluable models to agree (pre-registered)
ceilings audit
ceilings numbers --family info --arm E --delta medium --out reports/numbers_entries.json
```

- `calibrate` searches each family's difficulty grid until baseline sits at chance and the
  family's own intervention reaches 90%. A family no setting passes is a finding: that model
  cannot show that ceiling in isolation.
- `analyze` reports **not evaluable** for any model and family whose gate failed, that has fewer
  instances than `prereg/config.json: sample_sizes.per_cell`, or that lacks a required arm.
- `numbers` writes entries for the paper's `paper/numbers.json` registry, so the camera-ready
  checker can rerun `ceilings show` and fail if a printed number goes stale.

## Running on a laptop

Model inference heats a laptop. Keep runs throttled and under the watchdog, which caps the
container and kills the run if macOS reports a thermal warning, Ollama's CPU stays high, or a time
limit is reached:

```bash
docker build -t ceilings:dev .
python3 scripts/watchdog.py --name smoke --model gemma3:1b --cpus 1.5 --memory 2g \
  --env OLLAMA_HOST=http://host.docker.internal:11434 -- bash scripts/local_confirm.sh
```

`scripts/local_confirm.sh` and `scripts/local_smoke.sh` set a 50% duty cycle
(`CEILINGS_OLLAMA_DUTY`), 4 Ollama threads, short reasoning budgets, and cool-downs between phases.

## On GCP

```bash
git clone https://github.com/Ishan-K-S/coffeelaw.git
cd coffeelaw/experiments/ceilings
```

Open `notebooks/ceilings_small_run.ipynb` (its first cell installs the package), or use Docker:
`docker build -t ceilings .` then `docker run --rm -v "$PWD":/work ceilings` for the tests.
