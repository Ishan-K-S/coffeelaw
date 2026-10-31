# Paid run on OpenRouter (October 2026)

The confirmatory run of the ceiling-discrimination experiment: ten models, the pre-registered
design frozen before the main grid (hash `7961e5c2…050f0e`, [prereg/](prereg/)), 850 paired
instances per cell. Machine-readable outputs are in [results/paid-2026-10/](results/paid-2026-10/);
raw runs and transcripts are in `data/paid/` (not committed). Departures from the plan are in
[prereg/DEVIATIONS.md](prereg/DEVIATIONS.md) and in the frozen config's `run_plan.deviations`.

## Verdicts

| Hypothesis | Verdict | Supported for | Inconclusive for | Not evaluable |
|---|---|---|---|---|
| H1 information: E helps; K, M, A1, P1, P2 do not | **supported** | Llama 3.3, Mistral Small 3.2 | DeepSeek, Gemma, Nemotron, Qwen, MiMo | Haiku, gpt-4o-mini, GLM |
| H2 reasoning: K helps; E, M, A1, P1, P2 do not | inconclusive | none | DeepSeek, Mistral, gpt-4o-mini, MiMo, GLM | Haiku, Gemma, Llama, Nemotron, Qwen |
| H3 communication: M helps; K, E, A1, P1, P2 do not | inconclusive | Nemotron | DeepSeek, Qwen, MiMo | 6 others (comm_ip gate failed) |
| H4 A1 (more agents, same totals) gives no gain | inconclusive | Llama | DeepSeek, Mistral, gpt-4o-mini, MiMo, GLM | 4 |
| H5 A2 gain equals the same compute in one agent | inconclusive | Llama | DeepSeek, Mistral, gpt-4o-mini, MiMo, GLM | 4 |
| H6 information error never significantly below the Fano floor | **supported** | DeepSeek, Nemotron, Qwen, MiMo | | |
| H7 inner product flat then rising; parity jumps at 1 bit | **supported** | DeepSeek, Nemotron, Qwen, MiMo | | |

The rule is pre-registered: a model supports a hypothesis only when every one of its support tests
rejects after Holm (superiority by exact McNemar for the matching arm, equivalence within ±0.05 by
Newcombe TOST for every other arm); a hypothesis needs two agreeing evaluable models. No hypothesis
was contradicted: no falsifier test rejected for any model.

**What the inconclusive verdicts mean.** In every family the matching arm helps on every model and
the other arms barely move accuracy, but at n = 850 many of those near-zero differences do not
pass equivalence at ±0.05 after Holm correction across the model's tests. The exceptions that are
real effects, not power, are listed below.

## Main grid (medium delta, 850 paired instances, gains over baseline)

| Family | Matching arm, gain range | Other arms | Models |
|---|---|---|---|
| info | E +0.76 to +0.87 | K, M, A1, A2, P1, P2 all within −0.03 to +0.03 | 7 |
| reason | K +0.12 to +0.30 | E, A1, P1, P2 within −0.04 to +0.06; **M +0.26 to +0.68** | 9 |
| comm_parity | M +0.44 to +0.53 | K, E, A1, P1, P2 within −0.05 to +0.05 | 8 |
| comm_ip | M +0.45 to +0.50 | all within −0.02 to +0.03 | 4 |

Every baseline is at chance (info 2^known/2^k, reason 0.25, comm 0.5). Full table: `tests.csv`,
`gain_table.csv`.

Findings that bear on the verdicts:

- **M on reason is a design flaw, not a refutation.** Agent B writes the worked chain into its
  message, so M delivers the sender's compute as text: B states the target shift in 407 to 782 of
  850 messages, and M beats K on all 9 models. Under the frozen rule this blocks H2's support for
  every model; it is reported as a limitation and not corrected after the fact.
- **A2 is worse than the same compute in one agent on reason** by 0.11 to 0.35 on 8 of 9 models
  (Haiku +0.00). H5 predicts equality; the frozen falsifier tests only A2 > single agent, so this is
  inconclusive, not contradicted. It is the clearest effect against a prediction in the run. (Qwen's
  single-agent reference got 59% less added compute than its target and still beat A2.)
- **A1 on comm can only lose.** Agent B is one of the four answerers and sees its own ready-made
  message, while no agent sees both halves, so A1 shifts accuracy below baseline by up to 0.04
  (Qwen parity 0.425 vs 0.465). This works against H4's equivalence.
- **MiMo K on comm_parity is a small harm** (−0.05, 95% CI −0.09 to −0.02).

## Calibration gates

Two stages: the fix's lower bound ≥ 0.90 and the baseline near chance on 300 fresh gate instances,
then the baseline equivalent to chance within ±0.05 on the gate and main-grid baselines pooled
(about 1,150 instances).

| Model | info | reason | comm_parity | comm_ip |
|---|---|---|---|---|
| Claude Haiku 5.5 | fail | fail (stage 2, 0.290) | pass | fail |
| DeepSeek v4 flash | pass | pass | pass | pass |
| Gemma 3 27B | pass | fail (stage 2, 0.313) | pass | fail |
| GLM 4.7 flash | fail | pass | fail | fail |
| gpt-4o-mini | fail | pass | fail | fail |
| Llama 3.3 70B | pass | fail | pass | fail |
| MiMo v2.6 flash | pass | pass | pass | pass |
| Mistral Small 3.2 24B | pass | pass | pass | fail |
| Nemotron 3.5 super | pass | fail (stage 2, 0.282) | pass | pass |
| Qwen3 235B A22B | pass | fail (stage 2, 0.278) | pass | pass |

Qwen comm_parity was the closest stage-2 pass (0.475 against chance 0.5).

## Delta curves (small and large delta on Gemma, Mistral, MiMo)

The curves are wired correctly (8 arms × 2 levels × 850 instances per group, the same instances as
the main grid), and the gains are flat: E on info +0.74 to +0.78, K on reason +0.17 to +0.22, M on
comm +0.46 to +0.50 at every level, and the other arms stay within about ±0.06.

**They cannot show whether gains depend on cost level.** The matching arm already sits at its
useful maximum at small delta: E has 16 probes (10 for Gemma at small) on tasks that need far
fewer, M sends 1 bit on parity and the full 8-bit list on comm_ip. The extra budget at medium and large
for E and M is therefore filler. K does deepen one sample's reasoning (medium, high, xhigh; for
Mistral small and medium are the same setting) and its gain stays flat, so deeper reasoning on one
of four samples adds nothing past medium. What the curves do show is that filler and
the wrong kind of resource add nothing at any level (Mistral comm_parity M even drops from 0.987 to
0.962, about 3 standard errors).

At small delta, 24 cells miss the ±5% cost target because the arm's cheapest setting already costs
more than half the baseline; the worst are Mistral reason A2_single_ref (+275%) and K (+97%). The
Gemma reason curve was stopped at 887 runs because its gate failed; those rows appear in
`gain_table.csv` as small and large cells of about 55 and should be ignored.

## H6 and H7 sweeps (DeepSeek, Nemotron, Qwen, MiMo; 100 instances per cell)

**H6** (info, k = 4, T ∈ {0…16} probes, flip rate p ∈ {0, 0.1, 0.2}): no cell's error is
significantly below the Fano floor after Holm, and every row used exactly T tool calls. The floor
is far from tight for these models: with noisy probes and many of them (p = 0.1 at T ≥ 6, p = 0.2
at T ≥ 8) error stays at 0.35 to 0.80 where the floor is 0.25 or below, so H6's support says the floor holds, not that models reach
it. Answer truncation weakens this sweep: on 16 options models spend the 400-token answer cap on
long probabilities, and Nemotron loses 11.9% of samples (up to 46% in one cell), which only adds
error.

**H7** (comm, 16 bits, message budget L ∈ {0…16}): parity jumps from 0.47–0.55 at L = 0 to
0.97–1.00 at L = 1 on all four models. The inner product stays at 0.40 to 0.58 through L = 12 to 14
and rises only at L = 16 (Qwen 0.61, MiMo 0.81, DeepSeek 0.82, Nemotron 0.88). Every message
carried at most L bits. The location of the rise is set by the protocol (B sends the first L bits
of its list, so the best possible accuracy below L = 16 is 0.5 + 0.5·2^−(16−L)), so it says nothing
about an optimal protocol. 78 of Qwen's 900 non-empty-budget messages were empty after retries.

## Integrity and quality

- `ceilings check` passed on all 356,287 rows: no answer leaks, private evidence stays private,
  message budgets and step counts hold, ledgers add up. `ceilings audit` passes everything except
  cost matching (52 cells off ±5%: 24 at medium, 24 at small, 4 at large). The medium misses are
  mostly 5 to 8%; the large ones are Qwen reason A2_single_ref −59%, Nemotron comm_ip K +30.5%,
  Nemotron info A2_single_ref −14.5%, Qwen comm_ip K −13%, Mistral reason A2_single_ref −12%.
- No invalid or refused runs. Temperature 0.7 is logged on every row, but Haiku's provider accepts
  neither temperature nor seed, so Haiku used its default sampling.
- Quality warnings: the comm reasoning stage hits its 512-token cap in most calls (up to 99% for
  Qwen); M messages needed retries for format on Haiku parity (50%) and Qwen comm_ip (46%), and the
  delivered message was wrong or empty in 52 of 850 Haiku parity runs; Gemma's info family at k = 2
  has only 24 distinct prompts across 850 instances, so its outcomes are correlated.
- Comm baselines track the parity of the agent's own list (53 to 94% of answers); the instance set
  has P(parity even) = 0.467, which explains Qwen's 0.465 baseline. Labels are balanced, so this
  cannot fake a gain.
- The main grid and the curves and sweeps each had an independent read-only audit; their findings
  are folded in above.

## Spend

| Phase | Cost |
|---|---|
| Calibration | $5.84 |
| Prepare (gates and cost matching) | $25.75 |
| Reason re-gate at d = 3 | $2.73 |
| Main grid | $182.22 |
| Delta curves and H6/H7 sweeps | $88.99 |
| **Total (key usage)** | **$305.60** |
