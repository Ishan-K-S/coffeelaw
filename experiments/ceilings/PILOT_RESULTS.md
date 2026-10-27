# Free-tier pilot on NVIDIA build (October 2026)

A small pilot of the ceiling-discrimination experiment on hosted models, run to exercise every
phase with capable models and to size a paid full run. **These are pilot results, not
confirmatory evidence:** n = 30 per cell against a pre-registered 850, the pre-registration was not
frozen, and several design problems (below) were found and fixed during the run. Machine-readable
outputs are in [results/pilot-2026-10/](results/pilot-2026-10/); raw runs and transcripts are in
`data/` (not committed).

## Setup

- **Provider:** NVIDIA build free endpoints (`https://integrate.api.nvidia.com/v1`), about 40
  requests per minute per key plus per-model rate limits that came and went. Per-model settings
  (how reasoning is switched off, answer caps, one served-name alias) are in
  [configs/nvidia_build.json](configs/nvidia_build.json) and are stored in every run record.
- **Harness:** the default design (rotation probes, ready-made comm message, probabilities first,
  shuffled options, temperature 0.7), all runs in capped Docker containers.
- **Models.** Ten were chosen and smoke-tested; Gemma 4 31B and Llama 3.2 90B never answered
  within 300 s and were replaced by DiffusionGemma 26B and Step 5. To fit the free tier the pilot
  was then cut to six fast models for calibration and two for the main grid:

| Model | Developer | Calibrated | Main grid |
|---|---|---|---|
| nemotron-3-ultra-550b-a55b | NVIDIA | yes | yes (info) |
| step-5-preview | StepFun | yes | yes (info, reason) |
| nemotron-3.5-super-vl-120b-a12b | NVIDIA | yes | comm_parity with reasoning (n = 20, unmatched) |
| nemotron-3.5-lightning-30b-a3b | NVIDIA | yes | no |
| llama-3.2-11b-vision-instruct | Meta | yes | no |
| glm-5.3-flash | Z.ai | info only (rate-limited out) | no |
| gpt-oss-20b, muse-glimmer-30b, diffusiongemma-26b, deepseek-v4.1-flash | | partly / no | no |

- **Phases:** `calibrate` (n = 30 per difficulty setting), `prepare` (gate on 30 fresh instances
  plus real cost matching at medium delta on 6 instances), main grid at medium delta (all 9 arms,
  n = 30), the H6 sweep (T ∈ {0,1,2,4}, p ∈ {0, 0.1}, n = 20), and the ablations (one model,
  baseline and E, n = 20). Every phase was followed by `ceilings check` and independent read-only
  audits (calibration transcripts, calibration statistics, main-grid integrity, main-grid
  statistics).

## Calibration gates

| Model | info | reason | comm_parity | comm_ip |
|---|---|---|---|---|
| nemotron-3-ultra | pass (k=3, exact probes) | pass d=2 in search, **fail** on fresh instances (baseline 0.60 vs 0.25) | fail at n=2 | fail at n=2 |
| step-5-preview | pass (k=3) | pass (d=2; fresh: baseline 0.27) | fail | fail |
| nemotron-3.5-super-vl | pass (k=3) | pass (d=2) | fail | fail |
| nemotron-3.5-lightning | fail (full fix 0.73) | pass (d=2) | fail | fail |
| llama-3.2-11b | fail | fail | fail | fail |
| super-vl, comm with a reasoning stage | | | **pass up to n=8** (full fix 0.93–1.00) | **pass at n=2, 3, 4** (1.00); n=6 interrupted, resumed in a follow-up |

- **Communication gates fail because of reasoning, not communication.** With B's correct message
  in hand, agent A (answering with no reasoning stage) repeats B's bit as the answer in 80–94% of
  samples. Giving every comm-family call a medium reasoning stage
  (`--harness comm_depth=medium`) makes both comm families pass on Super VL. The design for the paid
  run should do this.
- **Noisy probes (p = 0.1) fail for every model** (full fix 0.60–0.73) although a majority vote
  over the logged reports is right 30/30: again an aggregation (reasoning) failure.
- **comm_ip's search stopped at n=6** when the calibration process reached its 8,000-request
  cap: it started before 429 replies stopped counting against the cap, so rate-limit retries used
  it up. The comm-with-reasoning variant therefore went on with comm_parity only; a follow-up
  resumes comm_ip from its cached settings and runs its `prepare` and main grid.
- **The n = 30 gate is loose.** Its tolerance passes any baseline from about 0.05 to 0.45 at chance
  0.25; Ultra's reason gate passed in the search at 13/30 and failed on fresh instances at 18/30.

## Main grid (medium delta, n = 30 paired instances)

Gain over baseline with a 95% paired bootstrap interval ([gain_table.csv](results/pilot-2026-10/gain_table.csv)):

| Arm | Ultra, info | Step 5, info | Step 5, reason |
|---|---|---|---|
| E | **+0.87** [+0.73, +0.97] | **+0.90** [+0.77, +1.00] | +0.27 [+0.13, +0.43] |
| K | −0.03 | −0.03 | **+0.90** [+0.80, +1.00] |
| M | −0.03 | −0.03 | **+0.83** [+0.70, +0.97] |
| A1 | −0.03 | −0.07 | +0.30 [+0.10, +0.50] |
| A2 | −0.03 | −0.07 | **+0.80** [+0.60, +0.97] |
| A2_single_ref | −0.03 | 0.00 | +0.13 [0.00, +0.30] |
| P1 | −0.07 | −0.03 | +0.30 [+0.13, +0.50] |
| P2 | −0.03 | −0.07 | +0.27 [+0.07, +0.47] |
| baseline accuracy | 0.10 (chance 0.125) | 0.10 | 0.10 (chance 0.25) |

- **Information family:** only E helps, as H1 predicts, on both models. At this setting the team
  holds no information about the answer without probes, so the null for K, M and A is true by
  construction; it confirms the harness, not the theory. A partial-information setting is needed
  for those nulls to be informative.
- **Reasoning family:** K helps, as H2 predicts. M and A2 help almost as much. In this family B
  sees all the evidence, so its message is computation done on A's behalf; the message channel is
  acting as compute, which is the paper's own "only total compute counts" point but cuts against a
  literal reading of "M does not help". The +0.27 to +0.30 from E, A1, P1 and P2 is most likely an
  artefact of the depressed baseline (below): any change to the prompt moves the uninformed default
  answer.
- **Verdicts** ([predictions.json](results/pilot-2026-10/predictions.json)), with the pilot's
  n = 30 as the minimum and 2 models required to agree: H1 inconclusive (the equivalence tests need
  more data), H2 inconclusive (one evaluable model), H3 not evaluable (no comm gate passed in the
  main pilot; see the communication section for a first look), H4 inconclusive, H5 inconclusive (see below), **H6 supported** (no cell is below its
  Fano floor; both models), H7 inconclusive.
- **H5 cannot be read from this pilot.** Step 5's A2 beats its single-agent reference on reason
  (+0.80 vs +0.13), which the analysis scores as "contradicted" for that model. The reference got
  A2's pooled reasoning depth (none, at A2's matched strength) and was then topped up to the cost
  target with neutral filler, while A2's relay messages are themselves reasoning. The reference
  never received the Δ compute the prediction is about. The paid run should cost-match the
  reference with reasoning tokens, not filler.

## Communication with a reasoning stage (Nemotron Super VL, comm_parity, n = 20)

Every call in the communication families gets a medium reasoning stage
(`--harness comm_depth=medium`). The gate passed on 30 fresh instances at n = 8 bits (baseline
0.57 vs chance 0.50, full fix 0.97). The grid was trimmed to fit the free tier: 20 instances,
7 arms (no A2 or A2_single_ref), and **arms not cost-matched** (default strengths; P1 and P2 carry
much more text than the others), so read effect directions, not equivalence
([gains](results/pilot-2026-10/comm_parity_with_reasoning_gains.txt),
[checks](results/pilot-2026-10/comm_parity_with_reasoning_checks.json)).

| Arm | Accuracy | Gain over baseline [95% CI] | Mean tokens per run |
|---|---|---|---|
| baseline | 0.35 | | 4.5k |
| **M** (1 bit from B) | **0.95** | **+0.60** [+0.35, +0.85] | 5.0k |
| E (8 lines of same-kind private evidence) | 0.45 | +0.10 [−0.20, +0.40] | 5.3k |
| K (high) | 0.40 | +0.05 [−0.20, +0.30] | 9.0k |
| P1 | 0.45 | +0.10 [−0.15, +0.35] | 14.9k |
| A1 | 0.25 | −0.10 [−0.35, +0.15] | 4.3k |
| P2 | 0.25 | −0.10 [−0.35, +0.15] | 17.1k |

Only the message helps, as H3 predicts, once reasoning no longer binds; the other arms' intervals
all include zero at this n. The comm_ip search reached n = 4 (all passing) before the pilot was
cut down, and the H7 bit sweep was not run. All 140 runs are valid. (`ceilings check` first flagged
every run because its one-resource check compared arms with the default baseline instead of the
baseline under the reasoning floor; fixed, and the checks pass.)

## Ablations (Nemotron Ultra, info family, baseline and E, n = 20)

| Option | baseline | E | Main grid on the same instances |
|---|---|---|---|
| probe=agent | 0/20 | 19/20 | baseline 1/20, E 19/20 |
| schema=answer_first | 0/20 | 19/20 | |
| shuffle=off | 0/20 | 15/20 | |
| temp=0 | 0/20 | 19/20 | |
| sender=encode | not run (no comm family passed for Ultra) | | |

E's effect does not depend on the answer schema, the temperature or who picks the probes.
Unshuffled options cost E four instances (15/20), all confident misreads of one attribute, not
leaks (McNemar p = 0.22). Two caveats: probe=agent was cost-matched with the rotation table and
costs about 5.7× its target, and its run used the old probe parser (see below), so 35% of its
"choices" were made by the parser. It should be rerun before it is read.

## Harness problems found and fixed during the pilot

All fixed with tests (156 tests pass); the runs affected are noted.

- **Hosted-provider support:** built-in reasoning switched off per model, `<think>` and
  `reasoning_content` stripped and logged, a requests-per-minute limiter, `Retry-After`-aware
  backoff with a separate allowance for 429s, a request cap (`--max-requests`), served-name
  aliases, and provider profiles.
- **Parsing:** servers that ignore `response_format` get the JSON format in every answer prompt; a
  reply that names several options without choosing, a reply cut off by the length cap, and a
  probabilities-only reply with a tied top are parse failures rather than guesses.
- **Reasoning stage:** text before a stray `</think>` was dropped from the turn passed to the answer
  (10/120 Ultra K-arm samples in calibration).
- **Cost matching:** a negative measured slope turned into a ~250M-token filler and exhausted memory
  (Step 5's reason `prepare`, OOM-killed six times); matching is skipped for a family whose gate
  failed; strengths are binary-searched.
- **A2_single_ref** got a 1024-token reasoning stage against four A2 agents with none; it now gets
  the pooled token budget, and its info-family runs were re-matched and rerun. (The remaining H5
  problem above is the filler top-up.)
- **probe=agent parsing** picked the first attribute in table order mentioned anywhere; now strict.
- **Parity M arm** allowed n characters against a one-bit message format; now 1 bit.
- **Run records** now include every request ID (message and probe calls were missing) and the
  config actually run after harness options.
- **Resumability:** calibration settings, `prepare`'s gate and every matching measurement are
  cached, so interrupted phases resume.

## Recommendations for the paid full run

Items 1–6 are implemented in the harness (see the README's "Design changes after the October 2026
pilot"); item 7 is a model-selection rule for the run itself.

1. **Communication families:** give every call a reasoning stage (`comm_depth=medium` or deeper),
   so the comm gate and H3/H7 test communication.
2. **Floor scoring:** uninformed answers default to the first option (host-00 in 24–30 of 30
   runs), because the host table is always listed in the same order. Shuffle the table order per
   call, randomize ties with a seed, balance true labels across instances, and report p(true) or
   a log score alongside accuracy.
3. **Gate:** replace the n = 30 gate with an equivalence test at ±0.05 plus a one-sided bound on the
   full fix ≥ 0.90 (about n ≥ 300 per cell), and confirm the chosen setting on fresh instances.
4. **Compute arms:** the 256-token reasoning stage at medium is cut off in 80–100% of calls and K
   costs 1.8–2× its target; recalibrate the reasoning budgets so K can be matched. Cost-match
   A2_single_ref with reasoning tokens, not filler.
5. **Ablations:** cost-match each ablation separately (probe=agent especially) and rerun
   probe=agent with the fixed parser.
6. **Information family:** add a partial-information setting so the K/M/A nulls are informative;
   use the strong Fano bound and add a T = 3 cell so H6 has power.
7. **Models:** pick models whose built-in reasoning can be switched off (or cap it); Step 5 still
   reasons in its hidden channel in the K arm, and gpt-oss and Muse Glimmer always do.

## Size and cost of the paid run

- `ceilings power` on these data gives 372 per cell for Ultra (info) and 5,259 for Step 5, where
  the reason family's intermediate gains (P1, P2, E around +0.3) dominate. With the floor
  artefact removed, the main-grid reviewer estimates about **1,200 paired instances per cell**
  for the equivalence tests, against the 850 planned.
- Measured tokens per team run at medium delta: info about 1.9k at baseline and 3.7–5.3k on the
  matched arms; reason about 0.9k and 1.6–3.8k; communication with a reasoning stage about 4.3k
  at baseline. Inputs are 83% of all tokens.
- At about 1,200 instances per cell, 4 families, 9 arms and 3 delta levels, that is roughly
  **600–900M tokens per model** including calibration and matching. At typical hosted prices for
  open-weight models (assumed, not quoted: about $0.05–0.15 per million tokens for small models
  and $0.40–0.80 for large ones) that is about **$40–130 per small model and $250–700 per large
  model**, or roughly **$1,000–2,000 for three models**. With 500+ requests per minute the run
  takes about 1–3 days.
- The whole free-tier pilot used about 10M logged tokens (3.8M in the grids, sweeps and ablations;
  5.8M in the gates) and about 18,000 logged model calls, plus unlogged cost-matching calls and
  rate-limit retries, over about two days.
