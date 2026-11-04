# Arm design 3: realistic agent teams (draft design)

This is a design draft. It is not part of the frozen pre-registration (`config.json`, `HASH`),
nothing here has been implemented, and no paid call depends on it. The design-2 A2 result stays
in RESULTS.md as the result for a naive relay. Design 3 adds teams; it does not replace that
result.

## Decisions made (2026-11-03)

- Same compute means the same **total cost**, every token included.
- The lead must use **at least 2 workers**.
- The isolated-swarm control **W0 is in, but optional**: it is the first thing cut if the cost
  runs too high.
- The **Claude Code swarm comes later**, after the main arms.

## Why

Design 2's A2 tested one hand-made protocol:
- a fixed chain D → C → B → A;
- one message per agent, written in a single turn;
- the final agent sees only the last message.

The agents sent bare guesses (median 7-40 tokens per model), and weak models copied wrong ones:
Mistral copied 43% of the wrong answers it received, GLM 26%. Real multi-agent systems don't work like that.
Design 3 tests H5 on teams built the way deployed agent systems are built, coding agents in
particular. The goal is agents that behave realistically, with no idealised protocol and no
straw man.

## How real coding-agent teams work, and what each point means here

| How coding agents work | Design choice |
|---|---|
| **An agentic loop.** An agent acts in turns: think, call a tool, read the result, decide what to do next, and stop when it judges it is done or hits a turn cap. | Every agent runs a loop with tools and a `max_turns` cap, instead of making one fixed call. |
| **A lead delegates.** The lead reads the task, decides how to split it, and writes each worker a brief. Each worker starts with a fresh context (its brief, plus access to the workspace) and returns a final report. The lead never sees a worker's reasoning. | The lead must use at least 2 workers (up to 3) and decides what each one does. The harness never splits the task. |
| **They coordinate through a workspace, not chat.** Agents mostly coordinate through files (the repo, notes, a plan), not direct messages. | A shared notes file that agents read and append to. |
| **Reports are free-form and lossy.** What crosses between agents is what the sender chooses to write. | Reports are free text with a token cap. Reasoning stays hidden. |
| **Agents choose their own effort.** Agents decide how long to work. A spend cap stops them, not a fixed number of steps. | A team-level token budget, enforced by the harness, with the remaining budget shown to each agent. |
| **Checking is weak without tests.** Coding agents catch errors by running tests. A reasoning task has no tests, so the only check is re-deriving. | No checking tool in the core arms. A separate exploratory arm adds tools (below). |
| **Parallel workers.** Workers run at the same time and don't see each other's work until it lands. | Workers in a round see a snapshot of the notes from the round's start. Ordering is seeded and logged. |
| **Mixed models.** Teams often use a strong lead with cheap workers. | Out of scope for the core design (listed under extensions). |

## The task has to be divisible

The current reason family is one chain of d = 2-3 shifts, with nothing to divide, so a realistic
lead would solve it alone. Code is different: it is large and modular, which is why delegation
pays off there.

Design 3 uses the same chain generator, `tasks/reason.py`, with long chains (d = 6-30). The net
offset of any stretch of links can be summed without knowing where the chain starts ("Zola works
3 shifts after Mira"), so workers can each resolve one stretch and the lead combines the results.

**Layout: facts grouped into sections, like files in a repo.** If all d facts were fully
shuffled, a lead would have to trace the whole chain just to split it, and by then it might as
well finish the job alone. So the chain is cut into 3 consecutive sections ("Rota, part 1" to
"part 3"), each one's facts shuffled within the section. A lead can then delegate by section,
the way coding agents delegate by module, and each section still takes real reasoning. Every
agent, the solo agent included, sees the same layout, so it changes the task but not the
comparison. It is a new difficulty setting of the same family version (`layout="sections"`,
part of every instance ID and of the instance folder name), so it never mixes with design-2
instances, which regenerate unchanged.

- **Chain length per model.** Each model gets the largest d in {6, 9, 12, 18, 24, 30} at which
  the solo agent's pilot accuracy (100 instances, turn cap only, no token cap, so its natural
  cost is measured too) is between 0.50 and 0.85. That leaves room for a team to land above or
  below the solo agent.
  - To keep the pilot cheap, the search starts at d = 12 and moves one step up if accuracy is
    above 0.85, or one step down if it is below 0.50, with at most 3 steps per model.
  - If no d qualifies, the model is left out of H5 and reported as such: at a ceiling of 1.0 a
    team can't beat the solo agent, and equivalence would be trivial.
  - The rule is fixed before the pilot is looked at.
- **Sections hold consecutive links.** Part 1 holds the starting fact and the first third of the
  links, and so on. Sections appear in chain order, and only the facts inside each section are
  shuffled.
- **A new instance split, `design3`.** It needs its own seed offset in `generate_many` (the offset
  table knows only calibration, main and heldout_blind), so no prompt repeats an earlier run.
- **4 options, chance 0.25.** There is no baseline arm. H5 compares teams with the solo agent,
  so the chain-length rule replaces the reason gate.

## Arms

All arms use the same model, the same tools and the same team budget B. Only the topology
differs.

| Arm | Topology | What happens |
|---|---|---|
| **S** (reference) | one agent | An agentic loop with a private notes file and `submit`. Same budget B. |
| **L** lead and workers (Claude Code subagent style) | 1 lead + 2-3 workers | The lead calls `delegate(brief)` (workers run in parallel and see the task facts plus the brief), reads the reports (capped at 300 tokens), may delegate again if budget is left, and calls `submit`. Calling `submit` before 2 workers have reported is refused with an error, so every L run actually splits the work. |
| **W** swarm (shared workspace) | 4 peers, no lead | Each round, every peer reads the notes, works, and may `append_note`. A peer calls `submit` when it is confident. The team answer is the majority of submitted answers, with ties broken by mean probability. The run stops when all peers have submitted or B is spent. |
| **R** author and reviewer (code-review style) | 2 agents | The author writes a worked solution. The reviewer checks it against the facts and writes a review. The author revises and submits. |
| **W0** isolated swarm (**optional**) | 4 peers, no notes | Like W, but no peer can read or write notes, so the team is 4 independent attempts plus a vote. W against W0 separates the effect of sharing from the effect of voting, and shows whether peers copy each other's mistakes. If the pilot shows the grid costs too much, W0 is cut first. |

The design-2 A2 relay is not rerun. It ran on a different task (d = 2-3, one shuffled list), so
it is reported separately as the naive-relay result and never compared with these arms.

**Every agent holds all the information.** Every agent's first prompt contains every fact line.
A worker's brief is extra instruction, never a substitute for the facts, and the harness doesn't
leave it to the lead to copy facts across. (Claude Code's own subagents read the repo
themselves; here the facts are the repo.) So a team can never lose information by
construction, the way an arm with zero information would. `ceilings check` verifies it on every
run, and `TheoryQuantities.I_held_by_agent` is extended to 2 bits for every agent: it currently
lists only A and B.

**How an agent's turns work:**
- **Action format.** Actions use a plain-text JSON protocol, `{"tool": ..., "args": ...}`, ReAct
  style, the same for every model. Native tool calling varies by model and provider, and that
  difference would get mixed into the result.
- **Per-turn effort.** Every turn is the adapter's standard free-text call: a reasoning stage at
  medium depth (512 tokens), then the action reply (at most 600 tokens), the same in every arm.
  Only the action reply stays in the agent's conversation, as in most agent loops. Effort varies only through the number of turns and agents an arm
  chooses to use. Models without native reasoning think in their turn's output, as they do in a
  normal agent loop.
- **Invalid actions.** An invalid action gets one retry with the parse error. If that also fails,
  the turn is used up and logged.
- **The budget is checked before each turn,** so one turn can overshoot B a little; the
  overshoot is in each record's `team.spent`.
- **No run ends without an answer.** Each answering agent has a reserved answer call outside
  the turn budget. If an agent reaches its turn cap, or the team spends B, before calling
  `submit`, the harness asks for a final answer with what the agent has. The run is flagged
  `forced_answer`, and the rate is reported per arm. A run that ends with no answer would be
  counted wrong for reasons that have nothing to do with splitting compute.

## Cost and matching

- **Total cost is the measure** (decided 2026-11-03): ledger total tokens, counting every call by
  every agent. That includes tool-call I/O and the repeated reading of facts, notes and reports.
  This is what a user pays, and team overhead is a real cost of teams, so it counts against them.
  It is also the measure design 2 matched on.
  - Because overhead counts, H5a ("no team beats one agent") is a little easier to hold than it
    would be under a compute-only measure. Generated tokens (output + reasoning), the closest
    thing to the theory's "compute", are reported next to it so readers can see the gap.
- **Budget B per model, the same for every arm:** the smallest of 2×, 3× and 4× the solo agent's
  natural total cost in the pilot at which every team arm's forced-answer rate is below 10%. A
  team of a lead and 3 workers reads the facts 4 times over, so too small a B would mostly
  measure teams running out of budget.
- **Primary comparison:** at equal caps B, which is the quantity a user actually sets.
- **S is never padded.** A solo agent that stops early chose to stop, as it would in real use.
  The design-2 approach of padding the reference up to the team's realized cost has no realistic
  equivalent in an agent loop. Realized total cost and generated tokens per arm are reported, so
  any gap in what was actually spent is visible.

## Hypotheses (draft; to be frozen before any paid run)

- **H5a, upper bound (confirmatory).** No topology beats S at equal cost.
  - Falsifier: team > S, by exact McNemar superiority after Holm across the team arms run for
    that model (L, W, R, and W0 if it is run).
  - H5a is contradicted if 2 models reject.
  - This is the claim the design-2 falsifier already tested. The swarm's vote and the lead's
    divided chain are the strongest realistic challenges to it.
- **H5b, equivalence (confirmatory, secondary).** Each team arm is within ±0.05 of S, by Newcombe
  TOST after Holm.
  - Predicted to hold on strong models (DeepSeek, MiMo, gpt-4o-mini).
  - Predicted to fail on Mistral and GLM, which showed a copying problem in design 2.
- **H5c, mechanism (pre-registered, descriptive).** A team's deficit grows with how often it
  accepts a wrong intermediate result.
  - The measure is the share of wrong worker reports, notes or reviews that the final answer
    agrees with.
  - Ground truth for every stretch of the chain is known, so reports that state a stretch's
    offset or a person's shift are graded automatically.
  - Parse coverage is reported per model, and a hand-checked sample of 100 reports per model
    calibrates the parser. In design 2, simple parsing covered only 366 of 850 GLM runs.
  - If W0 is run, W − W0 (paired) is the effect of sharing notes, reported with its 95% interval.
    The design-2 copying result predicts it is negative on Mistral and GLM.
- **Recorded, not tested:** how many workers the lead used (2 or 3), how the chain was split,
  turns used, forced answers, and budget left at `submit`.

## Exploratory arm: a Claude Code swarm

This arm runs real Claude Code agents, not agents that imitate them. It is outside the verdicts:
these runs can't be seeded, and their cost can't be matched to the other arms.

- **How it runs:**
  - Claude Code runs headless through the Agent SDK in a fresh temporary directory per instance,
    with `task.md` holding the facts.
  - A worker subagent is defined in `.claude/agents/worker.md`, and the answer is written to
    `answer.json`.
  - Two configurations: **solo** (one session, no Agent tool) and **swarm** (the lead is told to
    use at least 2 worker subagents with the Agent tool, matching arm L). Claude Code can't
    enforce this, so a session that used fewer is flagged and reported, not dropped.
  - Cost and tokens come from the SDK's result message.
  - The working directory holds only `task.md` and the agent definitions, never the instance
    record or its answer. The transcript must show every worker reading `task.md` before it
    reports. Otherwise the run is flagged, because a worker that never read the facts had no
    information.
- **Tools:**
  - **tools-off**, the core variant: Read, Write, Agent. No shell, so no code execution.
  - **tools-on**, descriptive only: adds Bash. A real coding agent would simply write a few lines
    of Python to sum the offsets. This turns a reasoning ceiling into a tool-use ceiling, which
    is realistic and worth showing, but it is another resource, not split compute.
- **Model:** Claude Haiku 5.5, the cheapest Claude model.
  - Its reason gate failed in October (stage 2, baseline 0.290), so it needs a new gate at the
    design-3 d.
  - Its provider ignores temperature and seed, which is another reason this arm is exploratory.
- **What it adds:**
  - Claude Code's own system prompt and tool definitions add a large fixed number of input tokens
    to every session, and a swarm pays that per worker.
  - That is exactly the overhead real swarms pay, so it is reported, not removed.
- **Before any spend:**
  - Confirm how Claude Code is billed here. It needs Anthropic API access; the experiment's paid
    key is OpenRouter.
  - Pilot 10 instances per configuration to measure the cost per instance.
  - Run in the existing Docker image, with keys passed as `-e NAME`.

## Integrity checks to add to `ceilings check`

- Every agent's first prompt contains every fact line (no agent ever holds less than the full
  2 bits).
- A worker sees only the task facts, its brief and the notes. It never sees the lead's reasoning.
- Every run ends with an answer, and forced answers are flagged and counted.
- No agent ever sees the answer.
- Budget caps and turn caps hold for every run, and every action is logged with the agent, its
  turn and the notes snapshot it saw.
- Reports and notes respect their caps.
- `ARM_DESIGN = 3` in run IDs and records, and `check_design` refuses to mix design 3 with
  earlier designs.
- SimPolicy gets a deterministic team loop, so the whole pipeline is unit-tested with no API
  calls.

## Scope and cost

- **Models:** the five design-2 reason models.
- **Cells:** 4 arms (S, L, W, R) per model, plus the optional W0. If the pilot shows the grid costs
  too much, W0 is cut first, then n is lowered.
- **Cost:** not yet estimable without a pilot.
  - In design 2, A2 and its reference cost $3.50 for 5 models at n = 850, with one call per
    agent at d = 2-3.
  - Agentic loops at d = 6-30 make several calls per agent and re-read context, so a cost per
    instance 5-15× higher is plausible. That puts the full grid well above the roughly $14 of
    credit left.
- **Plan:**
  - A 20-instance pilot per arm (W0 included) on Mistral and gpt-4o-mini to measure cost.
  - Then choose n per cell from the pilot's discordance and the budget. H5a's superiority test
    needs far fewer instances than H5b's ±0.05 equivalence.
  - Freeze n before the main run.

## Order of work

1. Implement the team loop, the arms (S, L, W, R, W0), the sectioned task layout, the ledger
   additions and the checks, with SimPolicy tests. Free.
2. Pilot to choose d (solo only, 100 instances per model), then pilot cost (20 per arm).
3. Fill in B, d and n per model, add a DEVIATIONS.md entry, then `ceilings prereg freeze`.
4. Main run with per-phase spending caps, then `check`, `audit` and `analyze`.
5. Later, after the main arms: the Claude Code arm (decided 2026-11-03).

## Open decisions

- How much budget to add for the main grid, after the pilot cost is known.
- Which credentials the Claude Code arm uses, when it comes up.
- Whether the tools-on (Bash) variant is wanted.

## Extensions (not in this design)

- Mixed teams: a strong lead with cheap workers.
- Teams on info and comm. Delegation can't add probes or private bits, so these only matter if
  agents get different private evidence there.
- A checking tool, such as a calculator charged as tool calls. This gives the reasoning tasks a
  "test suite" like the ones coding agents rely on.
