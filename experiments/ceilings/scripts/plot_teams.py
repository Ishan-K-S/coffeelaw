"""Figures for the design-3 agent-team results, from the committed outputs in results/design3-2026-11/.

Reads team_tests.csv and team_mechanism.csv only (no raw runs). Uses matplotlib's default
style, colors, markers and fonts.

    python scripts/plot_teams.py --results results/design3-2026-11
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr

p = argparse.ArgumentParser()
p.add_argument("--results", default="results/design3-2026-11")
p.add_argument("--out", default=None, help="default: <results>/figures")
a = p.parse_args()
res = Path(a.results)
out = Path(a.out) if a.out else res / "figures"
out.mkdir(parents=True, exist_ok=True)

MODELS = {  # weakest first, as in RESULTS.md
    "openai/mistralai/mistral-small-3.2-24b-instruct": "Mistral Small 3.2",
    "openai/z-ai/glm-4.7-flash": "GLM 4.7 flash",
    "openai/xiaomi/mimo-v2.6-flash": "MiMo v2.6 flash",
}
ARMS = {  # arm: (label, default color cycle entry)
    "L": ("Lead + workers", "C0"),
    "W": ("Swarm (shared notes, vote)", "C1"),
    "R": ("Author + reviewer", "C2"),
}
MARGIN = 0.05

tests = pd.read_csv(res / "team_tests.csv")
mech = pd.read_csv(res / "team_mechanism.csv")

# 1. team minus its spend-matched single agent, with the paired 95% interval
fig, ax = plt.subplots(figsize=(8, 5))
ax.axvspan(-MARGIN, MARGIN, color="gray", alpha=0.15, label="±0.05 margin (H5b)")
ax.axvline(0, color="gray", linewidth=1)
ticks, labels, y = [], [], 0
for model, mname in MODELS.items():
    for arm, (alabel, color) in ARMS.items():
        r = tests[(tests.model == model) & (tests.arm == arm)].iloc[0]
        lo, hi = json.loads(r.ci95)
        ax.errorbar(r["diff"], y, xerr=[[r["diff"] - lo], [hi - r["diff"]]], fmt="o",
                    color=color, capsize=3, label=alabel if model == next(iter(MODELS)) else None)
        ax.annotate(f"{r['diff']:+.2f}", (hi, y), xytext=(5, 0), textcoords="offset points",
                    va="center")
        ticks.append(y)
        labels.append(f"{mname}, {arm}")
        y += 1
    y += 0.6
ax.set_yticks(ticks, labels)
ax.invert_yaxis()
ax.set_xlim(-0.25, 0.17)
ax.set_xlabel("Team accuracy minus one agent spending the same tokens (95% paired interval)")
ax.set_title("How each team did against one agent with the same token budget")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
fig.savefig(out / "1_team_vs_single.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 2. deficit against how often the answer agrees with a teammate's wrong claim
fig, ax = plt.subplots(figsize=(7, 5))
ax.axhline(0, color="gray", linewidth=1)
short = {"Mistral Small 3.2": "Mistral", "GLM 4.7 flash": "GLM", "MiMo v2.6 flash": "MiMo"}
for arm, (alabel, color) in ARMS.items():
    sub = mech[mech.arm == arm]
    ax.scatter(sub.accepted_share, sub.deficit, color=color, label=alabel)
    for _, r in sub.iterrows():
        ax.annotate(f"{short[MODELS[r.model]]} {arm}", (r.accepted_share, r.deficit),
                    xytext=(6, -3), textcoords="offset points")
rho, pval = spearmanr(mech.accepted_share, mech.deficit)
ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_xlim(0, 0.9)
ax.set_xlabel("Answer agrees with a teammate's wrong claim about the target")
ax.set_ylabel("Deficit to the single agent (accuracy)")
ax.set_title("Teams that went along with wrong claims tended to fall further behind\n"
             f"(Spearman ρ = {rho:.2f}, p = {pval:.2f}, over 9 teams)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
fig.savefig(out / "2_copying_vs_deficit.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", *sorted(x.name for x in out.glob("*.png")))
