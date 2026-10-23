"""`ceilings` command-line interface (contracts/cli.md). Exit codes: 0 ok, 2 gate/validation
failure, 3 pre-registration hash mismatch, 4 budget exceeded."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ceilings import arms, experiment, prereg, tasks
from ceilings.adapters.base import RequestCapExceeded
from ceilings.agents import make_policy
from ceilings.audit import audit
from ceilings.options import HarnessOptions
from ceilings.runner import BudgetExceeded, load_runs, run_grid

ALL_FAMILIES = "info,reason,comm_parity,comm_ip"


def _difficulty(text: str | None, family: str) -> dict | None:
    if not text:
        return None
    out = {}
    for part in text.split(","):
        key, value = part.split("=")
        out[key] = float(value) if "." in value else int(value)
    return out


def _levels(text: str) -> dict[str, float]:
    return {lvl: arms.DELTA_MULTIPLIERS[lvl] for lvl in text.split(",")}


def _progress(label: str):
    start = time.time()

    def report(done: int, total: int) -> None:
        if done == total or done % max(1, total // 10) == 0:
            print(f"  {label}: {done}/{total} runs ({time.time() - start:.0f}s)", flush=True)

    return report


def _policy(a):
    """The model policy with `--harness` options, and `--max-requests` applied to adapters that
    count requests."""
    policy = make_policy(a.model, HarnessOptions.parse(getattr(a, "harness", None)))
    adapter = getattr(policy, "adapter", None)
    if getattr(a, "max_requests", None) is not None and adapter is not None:
        adapter.max_requests = a.max_requests
    return policy


def cmd_generate(a) -> int:
    for fam in a.family.split(","):
        diff = _difficulty(a.difficulty, fam) or experiment.DEFAULT_DIFFICULTY[fam]
        insts = tasks.generate_many(fam, a.n, a.split, a.seed, **diff)
        print(tasks.save_instances(insts, Path(a.root)))
    return 0


def cmd_calibrate(a) -> int:
    """Search each family's difficulty grid for a setting that passes the gate on this model."""
    from ceilings.calibration import save_json, search_difficulty

    policy = _policy(a)
    failed = False
    for fam in a.family.split(","):
        out = experiment.difficulty_path(a.root, fam, policy.model_id)
        res = search_difficulty(fam, policy, n=a.n, concurrency=a.concurrency,
                                grid="quick" if a.grid == "quick" else None,
                                cache_path=out.with_suffix(".settings.json"))
        save_json(res, experiment.difficulty_path(a.root, fam, policy.model_id))
        for r in res["results"]:
            print(f"{fam} {r['difficulty']}: baseline {r['baseline_acc']:.2f} (chance "
                  f"{r['chance_acc']:.2f}), full {r['matched_full_acc']:.2f}, "
                  f"{'PASS' if r['passed'] else 'fail'}")
        print(f"{fam}: chosen {res['chosen']}")
        failed |= res["chosen"] is None
    return 2 if failed else 0


def cmd_prepare(a) -> int:
    policy = _policy(a)
    failed = False
    for fam in a.family.split(","):
        res = experiment.prepare(
            fam, policy, a.root, a.n_main, a.n_pilot, _difficulty(a.difficulty, fam),
            concurrency=a.concurrency, levels=_levels(a.delta), n_match=a.n_match,
            match=not a.no_match,
        )
        cal = res["calibration"]
        unmatched = [f"{arm}/{lvl}" for arm, v in res["table"]["arms"].items()
                     for lvl, e in v.items() if not e["matched"]]
        print(
            f"{fam} {res['difficulty']}: gate {'PASS' if cal.passed else 'FAIL'} (baseline "
            f"{cal.baseline_acc:.2f}, chance {cal.chance_acc:.2f}, full "
            f"{cal.matched_full_acc:.2f}); unmatched cells: {unmatched or 'none'}"
        )
        failed |= not cal.passed
    return 2 if failed else 0


def cmd_run(a) -> int:
    policy = _policy(a)
    total = 0
    for fam in a.family.split(","):
        table = experiment.load_table(a.root, fam, policy.model_id)
        insts = experiment.load_main(a.root, fam, policy.model_id, a.split)[: a.limit]
        cells = experiment.main_cells(
            insts, table, deltas=tuple(a.delta.split(",")), arm_ids=tuple(a.arms.split(","))
        )
        total += len(run_grid(cells, policy, a.root, a.max_cost,
                              require_frozen=a.split == "main" and a.require_frozen,
                              concurrency=a.concurrency, progress=_progress(fam)))
    print(f"new runs: {total}")
    return 0


def cmd_sweep(a) -> int:
    policy = _policy(a)
    n_new = 0
    if a.family in ("info", "all"):
        Ts = [int(t) for t in a.T.split(",")]
        ps = [float(p) for p in a.flip_rate.split(",")]
        cells, floors = experiment.info_sweep_cells(a.k, Ts, ps, a.n, root=a.root)
        n_new += len(run_grid(cells, policy, a.root, a.max_cost, concurrency=a.concurrency,
                              progress=_progress("info sweep")))
        Path(a.root).mkdir(parents=True, exist_ok=True)
        (Path(a.root) / "floors.json").write_text(
            json.dumps([{"T": T, "p": p, "floor": f} for (T, p), f in floors.items()])
        )
    if a.family in ("comm", "all"):
        Ls = [int(x) for x in a.L.split(",")]
        for variant in ("parity", "ip"):
            n_new += len(run_grid(experiment.comm_sweep_cells(variant, a.n_bits, Ls, a.n, root=a.root),
                                  policy, a.root, a.max_cost, concurrency=a.concurrency,
                                  progress=_progress(f"{variant} sweep")))
    print(f"new sweep runs: {n_new}")
    return 0


def _floors(root: Path) -> dict:
    path = root / "floors.json"
    if not path.exists():
        return {}
    return {(r["T"], r["p"]): r["floor"] for r in json.loads(path.read_text())}


def cmd_analyze(a) -> int:
    from ceilings.analysis.gains import runs_frame
    from ceilings.analysis.report import load_gates, write_reports

    df = runs_frame(load_runs(a.root, a.model))
    min_n = a.min_n
    if min_n is None:
        per_cell = json.loads(Path("prereg/config.json").read_text()).get(
            "sample_sizes", {}).get("per_cell") if Path("prereg/config.json").exists() else None
        min_n = int(per_cell or 0)
    res = write_reports(df, a.out, delta=a.delta, floors=_floors(Path(a.root)), n_boot=a.n_boot,
                        gates=load_gates(a.root), min_n=min_n, min_models=a.min_models)
    for p in res["predictions"]:
        why = p.detail.get("reasons", {})
        note = "; ".join(f"{m}: {', '.join(r)}" for m, r in why.items() if r) if why else ""
        print(f"{p.prediction_id}: {p.verdict}" + (f"  ({note})" if note else ""))
    return 0


def cmd_power(a) -> int:
    from ceilings.analysis.gains import runs_frame
    from ceilings.analysis.power import sample_sizes
    from ceilings.analysis.report import load_gates

    df = runs_frame(load_runs(a.root, a.model))
    if df.empty:
        print("no runs")
        return 2
    res = sample_sizes(df, delta=a.delta, margin=a.margin, alpha=a.alpha, power=a.power,
                       gates=load_gates(a.root))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "power.json").write_text(json.dumps(res, indent=2, default=str))
    for model, m in res["models"].items():
        print(f"{model}: per-cell n for equivalence = {m['per_cell']}")
        for r in m["rows"]:
            print(f"  {r['family']:12s} {r['arm']:3s} vs {r['reference']:13s} {r['test']:7s} "
                  f"pilot n={r['pilot_n']:4d} pi_d={r['pi_d']:.2f} gain={r['gain']:+.2f} "
                  f"-> n={r['n_required']}" + ("" if r["gate_passed"] else "  (gate failed)"))
    print(f"recommended sample_sizes.per_cell: {res['per_cell']}")
    return 0


def cmd_check(a) -> int:
    from ceilings.checks import run_checks, summary_lines

    res = run_checks(a.root, load_runs(a.root, a.model), delta=a.delta)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "checks.json").write_text(json.dumps(res, indent=2, default=str))
    print("\n".join(summary_lines(res)))
    print("checks", "PASSED" if res["passed"] else "FAILED (integrity)")
    return 0 if res["passed"] else 2


def cmd_audit(a) -> int:
    records = load_runs(a.root, a.model)
    tables = {}
    for path in Path(a.root).glob("arms/*/*.json"):
        t = json.loads(path.read_text())
        tables[f"{t['model_id']}/{t['family']}"] = t
    res = audit(records, prereg.current_hash(), tables)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "audit.json").write_text(json.dumps(res, indent=2))
    print("audit", "PASSED" if res["passed"] else "FAILED",
          {k: len(v) for k, v in res["checks"].items()})
    return 0 if res["passed"] else 2


def _gain_rows(a) -> list[dict]:
    from ceilings.analysis.gains import gain_table, runs_frame

    table = gain_table(runs_frame(load_runs(a.root, a.model)))
    if table.empty:
        return []
    for col, val in (("family", a.family), ("arm", a.arm), ("delta", a.delta)):
        if val:
            table = table[table[col] == val]
    return table.sort_values(["model", "family", "arm", "delta"]).to_dict("records")


def cmd_show(a) -> int:
    from ceilings.numbers import show_line

    rows = _gain_rows(a)
    for row in rows:
        print(show_line(row))
    return 0 if rows else 2


def cmd_numbers(a) -> int:
    from ceilings.analysis.gains import gain_table, runs_frame
    from ceilings.numbers import entries

    rows = _gain_rows(a)
    table = gain_table(runs_frame(load_runs(a.root, a.model)))
    out = entries(table, a.root, rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {len(out)} registry entries to {a.out}")
    return 0


def cmd_prereg(a) -> int:
    print(prereg.freeze() if a.action == "freeze" else prereg.check_hash())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ceilings")
    p.add_argument("--root", default="data", help="where instances, runs and tables live")
    sub = p.add_subparsers(dest="cmd", required=True)

    def model_opt(sp, default="sim-bounded"):
        sp.add_argument("--model", default=default,
                        help="sim-bounded | claude-<m> | ollama/<m> | openai/<m>")

    def conc_opt(sp):
        sp.add_argument("--concurrency", type=int, default=1,
                        help="parallel team runs (model calls are I/O-bound)")
        sp.add_argument("--max-requests", type=int, default=None,
                        help="stop after this many HTTP requests (openai/ models; exit code 4)")
        sp.add_argument("--harness", default=None,
                        help="ablation options, e.g. probe=agent,sender=encode,"
                             "schema=answer_first,shuffle=off,temp=0,reason=0:128:512:1024:2048 "
                             "(use a separate --root per ablation)")

    g = sub.add_parser("generate", help="generate instances only")
    g.add_argument("--family", required=True)
    g.add_argument("--difficulty")
    g.add_argument("--n", type=int, default=50)
    g.add_argument("--split", default="main")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_generate)

    c = sub.add_parser("calibrate", help="search each family's difficulty for this model")
    c.add_argument("--family", default=ALL_FAMILIES)
    model_opt(c)
    conc_opt(c)
    c.add_argument("--n", type=int, default=30, help="instances per difficulty setting")
    c.add_argument("--grid", choices=["full", "quick"], default="full")
    c.set_defaults(fn=cmd_calibrate)

    pr = sub.add_parser("prepare", help="instances + calibration gate + cost matching")
    pr.add_argument("--family", default=ALL_FAMILIES)
    model_opt(pr)
    conc_opt(pr)
    pr.add_argument("--difficulty", help="override, e.g. k=3,p=0.1 (default: searched setting)")
    pr.add_argument("--n-main", type=int, default=100)
    pr.add_argument("--n-pilot", type=int, default=40)
    pr.add_argument("--n-match", type=int, default=None, help="pilot instances used for matching")
    pr.add_argument("--delta", default="small,medium,large", help="cost levels to match")
    pr.add_argument("--no-match", action="store_true",
                    help="skip cost matching (smoke tests only; cells are marked unmatched)")
    pr.set_defaults(fn=cmd_prepare)

    r = sub.add_parser("run", help="main grid")
    r.add_argument("--family", default=ALL_FAMILIES)
    model_opt(r)
    conc_opt(r)
    r.add_argument("--arms", default=",".join(arms.ARMS))
    r.add_argument("--delta", default="small,medium,large")
    r.add_argument("--split", default="main")
    r.add_argument("--limit", type=int, default=None, help="use only the first N instances")
    r.add_argument("--max-cost", type=float, default=None)
    r.add_argument("--require-frozen", action="store_true")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("sweep", help="H6 (tool calls) and H7 (message bits) sweeps")
    s.add_argument("--family", default="all", choices=["info", "comm", "all"])
    model_opt(s)
    conc_opt(s)
    s.add_argument("--T", default="0,1,2,3,4,6,8,12,16")
    s.add_argument("--flip-rate", default="0.0,0.1,0.2")
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--L", default="0,1,2,4,6,8,10,12,14,16")
    s.add_argument("--n-bits", type=int, default=16)
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--max-cost", type=float, default=None)
    s.set_defaults(fn=cmd_sweep)

    for name, fn, helptext in (("analyze", cmd_analyze, "gain table, tests, verdicts"),
                               ("check", cmd_check, "did the runs go as designed?"),
                               ("audit", cmd_audit, "integrity audit")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--model", default=None, help="restrict to one model (default: all)")
        sp.add_argument("--out", default="reports")
        sp.add_argument("--delta", default="medium")
        sp.add_argument("--n-boot", type=int, default=10_000)
        if name == "analyze":
            sp.add_argument("--min-n", type=int, default=None,
                            help="paired instances needed per cell (default: prereg per_cell)")
            sp.add_argument("--min-models", type=int, default=2,
                            help="evaluable models that must agree (pre-registered: 2)")
        sp.set_defaults(fn=fn)

    pw = sub.add_parser("power", help="per-cell sample size from pilot runs (FR-017)")
    pw.add_argument("--model", default=None)
    pw.add_argument("--out", default="reports")
    pw.add_argument("--delta", default="medium")
    pw.add_argument("--margin", type=float, default=0.05)
    pw.add_argument("--alpha", type=float, default=0.05)
    pw.add_argument("--power", type=float, default=0.8)
    pw.set_defaults(fn=cmd_power)

    for name, fn, helptext in (("show", cmd_show, "print gain rows in a stable format"),
                               ("numbers", cmd_numbers, "write paper/numbers.json entries")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--model", default=None)
        sp.add_argument("--family", default=None)
        sp.add_argument("--arm", default=None)
        sp.add_argument("--delta", default=None)
        if name == "numbers":
            sp.add_argument("--out", default="reports/numbers_entries.json")
        sp.set_defaults(fn=fn)

    pg = sub.add_parser("prereg", help="freeze or check the pre-registration")
    pg.add_argument("action", choices=["freeze", "check"])
    pg.set_defaults(fn=cmd_prereg)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except prereg.PreregHashMismatch as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except (BudgetExceeded, RequestCapExceeded) as e:
        print(f"error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
