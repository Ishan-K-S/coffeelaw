"""End to end at small scale: stub dry run (quickstart Scenario 1) and a simulated-agent suite.

The simulated agent's ceilings are known by construction, so the pipeline must recover the
diagonal. This checks the harness and the analysis, not the theory.
"""

import json

from ceilings import experiment, tasks
from ceilings.agents import SimPolicy, make_policy
from ceilings.analysis.gains import runs_frame
from ceilings.analysis.report import write_reports
from ceilings.arms import build
from ceilings.audit import audit
from ceilings.prereg import current_hash
from ceilings.runner import Cell, load_runs, run_grid


def test_stub_dry_run(tmp_path):
    policy = make_policy("stub-oracle")
    insts = tasks.generate_many("info", 20, "calibration", k=3, p=0.1)
    cells = [
        Cell(i, build(a, "info", s), None if a == "baseline" else "small")
        for i in insts
        for a, s in [("baseline", None), ("E", 3), ("K", "high"), ("P2", None)]
    ]
    records = run_grid(cells, policy, tmp_path)
    assert len(records) == 80
    assert run_grid(cells, policy, tmp_path) == []  # resumable: nothing re-run
    loaded = load_runs(tmp_path)
    assert audit(loaded, current_hash())["checks"]["ledger_totals"] == []


def test_sim_suite_recovers_the_diagonal(tmp_path):
    policy = SimPolicy()
    for fam in ["info", "reason", "comm_ip"]:
        prep = experiment.prepare(fam, policy, tmp_path, n_main=150, n_pilot=200, gate="screen")
        assert prep["calibration"].passed, fam
        run_grid(
            experiment.main_cells(prep["main"], prep["table"], deltas=("medium",)),
            policy,
            tmp_path,
            save_transcripts=False,
        )
    df = runs_frame(load_runs(tmp_path))
    res = write_reports(df, tmp_path / "reports", n_boot=500, min_models=1)
    verdicts = {p.prediction_id: p.verdict for p in res["predictions"]}
    # Small n: equivalence on noisy control arms may stay inconclusive, but nothing may be
    # contradicted, and the two cleanest diagonals must come out supported.
    assert verdicts["H1"] == verdicts["H2"] == "supported", verdicts
    assert all(verdicts[h] != "contradicted" for h in ["H1", "H2", "H3", "H4", "H5"]), verdicts
    table = res["gain_table"].set_index(["family", "arm"])
    assert table.loc[("info", "E"), "gain"] > 0.4
    assert table.loc[("reason", "K"), "gain"] > 0.5
    assert table.loc[("comm_ip", "M"), "gain"] > 0.3
    assert abs(table.loc[("reason", "M"), "gain"]) < 0.15
    preds = json.loads((tmp_path / "reports" / "predictions.json").read_text())
    assert {p["prediction_id"] for p in preds} >= {"H1", "H2", "H3", "H4", "H5"}
