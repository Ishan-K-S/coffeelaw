# Arm design 2 rerun: plan and pre-registration amendments

This is a draft for the design-2 rerun. It is not part of the frozen pre-registration
(`config.json`, `HASH`), which records the October 2026 paid run (arm design 1). Before any
paid call, archive this folder's frozen files with the paid data, apply the amendments below
to `config.json`, and run `ceilings prereg freeze`.

The scope covers only the tests that design 1 got wrong: H2 (K vote dilution, M on reason
carrying compute) and H5 (A2 never splitting compute).

## Scope

- **Data root:** `data/design2`, new and empty apart from the gate files copied in step 1.
  `ceilings run` refuses to mix designs in one root.
- **Family:** reason only.
- **Models:** the 5 whose reason gate passed in the paid run. Each uses its gated difficulty
  from `data/paid/calibration/reason__<model>.json`, not the defaults (d=12) and not
  `difficulty/*.json` (which says d=2 for all 5; three models were re-gated at d=3 on
  2026-10-28).

  | Model | Difficulty |
  |---|---|
  | `openai/deepseek/deepseek-v4-flash` | d=3 |
  | `openai/mistralai/mistral-small-3.2-24b-instruct` | d=3 |
  | `openai/xiaomi/mimo-v2.6-flash` | d=3 |
  | `openai/openai/gpt-4o-mini` | d=2 |
  | `openai/z-ai/glm-4.7-flash` | d=2 |

- **Arms and delta:** all 9 arms at the medium delta only, with n = 850 paired instances per
  cell. A2 cannot meet any level's target (below), so extra levels would only repeat the same
  A2 configuration.
- **Budget:** about $15-20, run with per-phase caps.

## Amendments to config.json

1. **K** (`design.K`): "strength is a depth plus how many answer samples the team uses
   (e.g. 'medium:1' is one sample, reasoning at medium); every sample reasons; candidates are
   ordered by sample count, then depth." Fewer votes works against K, so this is conservative
   for H2's superiority test.
2. **M on reason** (`design`): "M's channel on reason carries only lines copied word for word
   from the task, in task order. Both agents already hold every fact, so M is a consistency
   check (null by construction), not a test that could have gone either way."
3. **H5** (`predictions.H5` and `analysis`):
   - H5 is tested on reason only. On info and comm, A2 and its reference both sit at chance,
     because extra agents add no probes or private lists, so equivalence there cannot fail.
   - A2 is not matched to the level's target. Four reasoning agents add about 1.9-2.7× the
     medium target (paid matching measurements), so `ceilings audit` delta_matching flags on A2
     are expected.
   - A2_single_ref is matched to A2's realized added cost, mostly through padding (one agent
     at max reasons for only about 100-900 extra tokens), and H5 compares the two at that cost.
4. **Run plan** (`run_plan`): the scope above, with this file's date.

## Steps

1. Reuse the paid reason gates, since gates don't depend on the arm design (baseline and K at
   xhigh on all 4 samples):
   - copy `data/paid/calibration/reason__<model>.json` to `data/design2/calibration/`;
   - copy `data/paid/calibration/runs/reason__<model>.jsonl` to
     `data/design2/calibration/runs/` (stage 2 pools these with the new main baselines).

   `prepare` reuses a gate whose n (300), difficulty, harness and rule (two_stage) all match.
2. Run `ceilings --root data/design2 prepare --model <m> --family reason --difficulty d=<d>
   --n-pilot 300 --n-main 850 --n-match 10 --delta medium --gate two_stage`. This makes fresh
   cost matching for every arm.
3. Pilot: about 20 instances of the new A2 and A2_single_ref on one model, to check cost before
   the full grid.
4. Run `ceilings --root data/design2 run --model <m> --family reason --delta medium
   --require-frozen --max-cost <cap>`.
5. Run `ceilings --root data/design2 check`, then `audit`, then `analyze`.
