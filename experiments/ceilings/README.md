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

For models without a native effort setting, depth is a reasoning stage capped at
`low 0 / medium 256 / high 1024 / xhigh 2048 / max 4096` tokens (override with
`CEILINGS_REASONING_TOKENS`). Keep at least about 20 tokens per reasoning step the task needs.
Temperature is sent explicitly (`CEILINGS_TEMPERATURE`, default 0.7). Claude keeps refusal fallbacks
**off**, so no model substitution can occur, and every adapter marks a run invalid if the provider
serves a different model.

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
