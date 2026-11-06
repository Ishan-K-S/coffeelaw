# Arm design 3: realistic agent teams

This design is frozen in `config.json` and `HASH` (run_plan, arm_design 3) before the main run;
the design-2 freeze is archived in `prereg/design2/`. The design-2 A2 result stays in RESULTS.md
as the result for a naive relay. Design 3 adds teams; it does not replace that result.

## Decisions made (2026-11-03)

- Same compute means the same **total cost**, every token included.
- The lead must use **at least 2 workers**.
- The isolated-swarm control **W0 is in, but optional**: it is the first thing cut if the cost
  runs too high.
- The **Claude Code swarm comes later**, after the main arms.
- **2026-11-04, after the pilot: compare at equal spend, not equal caps.** Given the same budget,
  the lone agent stopped after 1-2 turns while teams spent 2.3-15.5× as much. Equal caps would
  compare unequal compute, so each team is compared with S_ref, one agent matched to that team's
  spend (below). This replaces the 2026-11-03 rule that S is never padded up to a team's cost.
- **2026-11-04, scope for the credit left (about $12.80):** 4 models (DeepSeek, Mistral, MiMo,
  GLM), n = 300 per cell, W0 cut. gpt-4o-mini is dropped: it was about a third of the cost and
  scored 0.05 alone at d = 12. Estimated total about $12.40, so n is re-checked against the
  measured cost after matching and lowered before the main run if needed.
- **2026-11-04, after the chain-length pilot** (one agent alone, 100 instances per step, $0.43):
  - | Model | d = 6 | d = 9 | d = 12 | d = 18 | d = 24 |
    |---|---|---|---|---|---|
    | DeepSeek | | | 1.00 | 0.94 | 0.99 |
    | MiMo | | | 0.66 | | |
    | GLM | 0.47 | 0.33 | 0.28 | | |
    | Mistral | 0.35 | 0.32 | 0.28 | | |
  - Only MiMo met the 0.50-0.85 rule. The final answers follow the agents' own reasoning (Mistral
    88 of 89 at d = 6), so the low scores are real reasoning errors, mostly on wrap-around steps.
  - Asked for the setup most like real coding agents, we kept tasks that divide into parts (at
    least 6 links, 2 per part; at 4 there is little to split) and **lowered the rule's floor from
    0.50 to 0.35** (chance + 0.10). At a floor of 0.30 the rule would pick d = 9 for Mistral and
    GLM, only 0.07-0.08 above chance, which leaves too little room to see a team do worse. The
    matched S_ref spends more than S, so it will score above these figures.
  - **Models and chain lengths: MiMo d = 12, GLM d = 6, Mistral d = 6. DeepSeek is dropped**: at
    0.94-1.00 up to d = 24, no team could beat its S_ref, and equivalence there is trivial.
  - **The lead delegates first**: in the pilot, leads free to choose mostly worked alone (forced
    answers in 70% of Mistral's L runs, 30% of MiMo's, 25% of GLM's), so L did not test
    splitting. Until 2 workers have reported, delegate is now the lead's only action.

- **2026-11-05, fixes found by the matching runs** (all before the freeze):
  - **Turn prompts.** A turn's reasoning stage reused the generic prompt that says "you will be
    asked for the final answer", so leads wrote their whole answer there and then submitted
    without delegating. Turns now say the action comes next and nothing has been done yet.
  - **An action written only in the reasoning counts.** If the reply is not a valid action but
    the reasoning ends with one, that action is taken (logged as `actions_from_reasoning`), as an
    agent loop that reads the model's whole output would.
  - **Arguments beside `"tool"`.** MiMo wrote `{"tool": "delegate", "brief": ...}` instead of
    putting the arguments under `"args"`; the parser now accepts both. Only MiMo did this, so
    the GLM and Mistral matching runs are unchanged.
  - **MiMo is served by Xiaomi.** On DeepInfra, MiMo's replies were garbled in 6 of 8 calls when
    one instance was replayed, and 0 of 6 on Xiaomi.
  - **Denser ladder and half rungs.** The ladder now has every attempt count up to 8, then 10,
    12 and 16. A team more than 10% from every rung and lying between two neighbouring attempt
    counts gets the half rung between them: `max:3.5` makes 3 or 4 attempts, by a coin fixed per
    instance, so the expected cost is halfway. Half rungs run only when needed. If the half
    still misses (on 20 instances its coin can fall well off half: MiMo's R measured `max:1.5`
    at -16%), the tenth rung whose expected cost, interpolated between the two neighbours, is the
    team's is run instead (MiMo R: `max:1.7`, one extra attempt on 70% of instances). A tenth's
    extra attempts include every instance a half's would, so the half rungs' runs are unchanged.
  - **The matching was rerun with these fixes** on the same 20 instances (data/design3). Every
    team of every model is matched (below), and no team's forced-answer rate is above 5%, so
    B stays as set: MiMo 64,000 tokens, GLM and Mistral 32,000.
- **2026-11-05, sample size from the measured cost.** The matched grid costs $0.0091 per
  instance on GLM, $0.0070 on Mistral and about $0.014 on MiMo (logged cost; billing ran about 11%
  higher). With about $10.5 of credit left, n = 300 on all three would leave under 5% margin, so
  **GLM and Mistral run n = 300 and MiMo n = 250**.

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
  cost is measured too) is between 0.35 and 0.85 (the floor was 0.50 until the pilot; see
  "Decisions made"). That leaves room for a team to land above or below the solo agent.
  - To keep the pilot cheap, the search starts at d = 12 and moves one step up if accuracy is
    above 0.85, or one step down if it is below the floor, with at most 3 steps per model.
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
| **S** | one agent | An agentic loop with a private notes file and `submit`, at the same depth as every team agent. Same budget B. Reported, not the H5 reference. |
| **S_ref** (H5 reference, one per team arm) | one agent | S raised until it spends what that team spends: deeper reasoning per turn, then several independent attempts and a majority vote (see "Cost and matching"). |
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
  medium depth (512 tokens), then the action reply (at most 1,200 tokens), the same in every arm.
  The cap was 600 until the chain-length pilot, where MiMo's long notes were cut off mid-JSON,
  rejected and retried, and one solo run grew to 101,619 tokens; agents are also told to keep
  each action short.
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
  It is also the measure design 2 matched on. Generated tokens (output + reasoning), the closest
  thing to the theory's "compute", are reported next to it.
- **Equal spend, not equal caps** (decided 2026-11-04). In the pilot (d = 12, no cap), teams spent
  this much more than S, median across models: R 3.3×, W0 4×, L 4.5×, W 5.8×, and up to 15.5× on
  MiMo. S chose to stop early. So H5 compares each team with its own **S_ref**, matched to the
  team's realized cost the way design 2 matched K and A2_single_ref:
  - `ceilings team-match` runs the teams on 20 matching instances (seeds apart from the main
    grid's), then S_ref up the ladder `medium, high, xhigh, max, max:2, ..., max:8, max:10,
    max:12, max:16`, with a half rung (e.g. `max:3.5`), then if needed an interpolated tenth (e.g.
    `max:1.7`), between neighbouring attempt counts when a team falls between them. A depth is per-turn reasoning, and `max:3` is three independent
    attempts at max depth with a majority vote. The climb stops once a rung spends more than
    every team.
  - Each team gets the rung whose mean total tokens are nearest its own. It is matched if that
    rung is within ±10%; otherwise it is recorded as unmatched with the gap, which is reported
    next to its H5 tests.
  - Several attempts and a vote count as one agent's compute here, as in every arm's baseline
    (four samples of agent A) and design 2's K. No filler padding is used, since in an agent loop
    it would only make the reference worse.
  - Deeper reasoning can finish in fewer turns, so a higher rung does not always cost more; the
    matcher picks by measured cost, not by rung order.
- **Budget B per model, the same for every arm,** is now a safety cap, not the matching device:
  the smallest of 4×, 8×, 12× and 16× S's natural cost at which every team arm's forced-answer
  rate in the pilot is below 10%. (The pilot showed teams at up to 15.5× S, so the earlier rule of
  at most 4× would have cut off every swarm run.)

## Hypotheses (frozen 2026-11-05)

- **H5a, upper bound (confirmatory).** No team beats one agent that spends as much.
  - Falsifier: team > its S_ref, by exact McNemar superiority after Holm across the team arms run
    for that model (L, W, R, and W0 if it is run).
  - H5a is contradicted if 2 models reject.
  - This is the claim the design-2 falsifier already tested. The swarm's vote and the lead's
    divided chain are the strongest realistic challenges to it.
- **H5b, equivalence (confirmatory, secondary).** Each team arm is within ±0.05 of its S_ref, by
  Newcombe TOST after Holm.
  - Predicted to hold on the strong model (MiMo; DeepSeek and gpt-4o-mini were dropped).
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

- **Models:** MiMo v2.6 flash (d = 12), GLM 4.7 flash (d = 6) and Mistral Small 3.2 (d = 6),
  decided 2026-11-04. gpt-4o-mini was dropped for cost, and DeepSeek because it is at ceiling.
- **Cells:** S, L, W, R and each team's S_ref per model, plus the optional W0 and its S_ref. If
  the grid costs too much, W0 is cut first, then n is lowered.
- **Pilot (2026-11-04):** 5 arms × 20 instances × 5 models at d = 12 with no cap, $0.72 (an
  earlier run, $0.56, found the bug where agents answered without their own last reasoning, and
  was discarded). Measured cost per instance, all 5 models: S $0.0018, L + W + R $0.0258,
  W0 $0.0082.
- **Cost with matched references:** each S_ref costs about what its team costs, so matching
  roughly doubles the team cost:
  - per instance, 5 models: about $0.053 without W0, $0.070 with it;
  - n = 300: about $16 without W0, $21 with it; n = 850: about $45 and $59;
  - plus about $1-1.5 for matching and $0.2-0.5 for the chain-length pilot.
- **Plan:** choose n per cell from the pilot's discordance and the budget (H5a's superiority test
  needs far fewer instances than H5b's ±0.05 equivalence), and freeze n before the main run.

## Order of work

1. Implement the team loop, the arms (S, L, W, R, W0), the sectioned task layout, the ledger
   additions and the checks, with SimPolicy tests. Free.
2. Pilot to choose d (solo only, 100 instances per model), then pilot cost (20 per arm). Done
   for cost at d = 12; the chain-length pilot is still to run.
3. Match S_ref per model and team at the chosen d and B (`ceilings team-match`).
4. Fill in B, d and n per model, add a DEVIATIONS.md entry, then `ceilings prereg freeze`.
5. Main run (`ceilings team-run --with-refs`) with per-phase spending caps, then `check`, `audit`
   and `analyze`.
6. Later, after the main arms: the Claude Code arm (decided 2026-11-03).

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
