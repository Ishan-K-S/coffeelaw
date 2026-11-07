"""The five final figures, one per experiment, from the committed results only (no raw runs).

Only the current version of each experiment is drawn:
- H1, H3, H4 (info and comm families), H6 and H7: results/paid-2026-10/
- H2 and H5 (reason family, arm design 2 rerun, replacing the October reason results):
  results/design2-2026-11/
- H5a-H5c (agent teams, arm design 3): results/design3-2026-11/

Plain matplotlib: default style, colors, markers and fonts.

    python scripts/plot_final.py --out results/final-figures
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr

p = argparse.ArgumentParser()
p.add_argument("--results", default="results")
p.add_argument("--out", default="results/final-figures")
a = p.parse_args()
res = Path(a.results)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
PAID, RERUN, TEAMS = res / "paid-2026-10", res / "design2-2026-11", res / "design3-2026-11"

NAMES = {
    "anthropic/claude-haiku-5.5": "Haiku 5.5", "deepseek/deepseek-v4-flash": "DeepSeek v4 flash",
    "google/gemma-3-27b-it": "Gemma 3 27B", "z-ai/glm-4.7-flash": "GLM 4.7 flash",
    "openai/gpt-4o-mini": "gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct": "Llama 3.3 70B",
    "xiaomi/mimo-v2.6-flash": "MiMo v2.6 flash",
    "mistralai/mistral-small-3.2-24b-instruct": "Mistral Small 3.2",
    "nvidia/nemotron-3.5-super-vl": "Nemotron 3.5 super", "qwen/qwen3-235b-a22b-2507": "Qwen3 235B",
}


def name(model: str) -> str:
    return NAMES.get(model.removeprefix("openai/"), model)


def save(fig, filename):
    fig.savefig(out / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


# 1. Gains over baseline by family and arm (H1-H4). Reason comes from the rerun.
paid = pd.read_csv(PAID / "tests.csv")
rerun = pd.read_csv(RERUN / "tests.csv")
tests = pd.concat([paid[paid.family != "reason"], rerun])
vs_base = tests[tests.reference == "baseline"]
FAMILIES = [("info", "Information"), ("reason", "Reasoning"),
            ("comm_parity", "Communication:\nparity"), ("comm_ip", "Communication:\ninner product")]
ARMS = [("E", "E: evidence"), ("K", "K: reasoning"), ("M", "M: messages"),
        ("A1", "A1: more agents"), ("P1", "P1: irrelevant text"), ("P2", "P2: blank filler")]
fig, ax = plt.subplots(figsize=(10, 5))
width = 0.13
x = np.arange(len(FAMILIES))
for i, (arm, label) in enumerate(ARMS):
    means, lows, highs = [], [], []
    for fam, _ in FAMILIES:
        g = vs_base[(vs_base.family == fam) & (vs_base.arm == arm)].gain
        means.append(g.mean())
        lows.append(g.mean() - g.min())
        highs.append(g.max() - g.mean())
    xs = x + (i - (len(ARMS) - 1) / 2) * width
    ax.bar(xs, means, width, label=label)
    ax.errorbar(xs, means, yerr=[lows, highs], fmt="none", ecolor="black", capsize=2,
                linewidth=1)
n_models = [vs_base[vs_base.family == f].model.nunique() for f, _ in FAMILIES]
ax.set_xticks(x, [f"{lab}\n({n} models)" for (_, lab), n in zip(FAMILIES, n_models, strict=True)])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Gain in accuracy over the baseline")
ax.set_title("Each task improves only with the resource it is missing")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
fig.text(0.5, -0.2, "Bars: mean over the models that passed calibration; whiskers: lowest to "
         "highest model. Medium budget, 850 paired instances per model.", ha="center")
save(fig, "1_gains_by_family.png")

# 2. Reasoning (H2 and H5): K against baseline, and A2 against one agent with the same compute
g = pd.read_csv(RERUN / "gain_table.csv")
g = g[g.delta == "medium"]
acc = g.pivot(index="model", columns="arm", values="arm_acc")
acc["baseline"] = g.groupby("model").baseline_acc.first()
acc = acc.sort_values("K")
fig, ax = plt.subplots(figsize=(9, 5.5))
y = np.arange(len(acc))
h = 0.2
for k, (col, label) in enumerate([("baseline", "Baseline (4 quick answers)"),
                                  ("K", "K: reasoning, same cost"),
                                  ("A2", "A2: reasoning split across 4 agents"),
                                  ("A2_single_ref", "One agent with A2's compute")]):
    ax.barh(y + (1.5 - k) * h, acc[col], h, label=label)
for yi, (_, r) in zip(y, acc.iterrows(), strict=True):
    gap = r.A2 - r.A2_single_ref
    ax.annotate(f"A2 − one agent: {gap:+.3f}", (max(r.A2, r.A2_single_ref), yi - h),
                xytext=(5, 0), textcoords="offset points", va="center")
ax.axvline(0.25, color="gray", linestyle="--", linewidth=1, label="Chance (25%)")
ax.set_yticks(y, [name(m) for m in acc.index])
ax.set_xlim(0, 1.35)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Accuracy on the reasoning task (850 paired instances per model)")
ax.set_title("Reasoning lifts every model; splitting it across agents costs the weaker ones")
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
save(fig, "2_reasoning_and_splitting.png")

# 3. H6: error against probes, mean of the sweep models, with the Fano floor
pred = {x["prediction_id"]: x for x in json.loads((PAID / "predictions.json").read_text())}
cells = pd.DataFrame(pred["H6"]["detail"]["cells"])
fig, ax = plt.subplots(figsize=(8, 5))
for k, (flip, label) in enumerate([(0.0, "clean probes"), (0.1, "10% of probes flipped"),
                                   (0.2, "20% of probes flipped")]):
    c = cells[cells.p == flip].groupby("T").agg(error=("error", "mean"), lo=("error", "min"),
                                                hi=("error", "max"), floor=("floor", "first"))
    ax.plot(c.index, c.error, marker="o", color=f"C{k}", label=f"Models, {label}")
    ax.fill_between(c.index, c.lo, c.hi, color=f"C{k}", alpha=0.15)
    ax.plot(c.index, c.floor, linestyle="--", color=f"C{k}", label=f"Fano floor, {label}")
ax.set_xticks(sorted(cells["T"].unique()))
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Number of probes (16 hosts, so 4 bits are needed)")
ax.set_ylabel("Error rate")
ax.set_title("No model's error was significantly below the information floor")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.text(0.5, -0.2, "Lines: mean of DeepSeek, Nemotron, Qwen and MiMo; shading: lowest to highest "
         "model. 100 instances per point.", ha="center")
save(fig, "3_error_vs_probes.png")

# 4. H7: accuracy against message bits
h7 = {k: v for k, v in pred["H7"]["detail"].items() if k.count("/") >= 2}
SWEEP = ["deepseek/deepseek-v4-flash", "nvidia/nemotron-3.5-super-vl",
         "qwen/qwen3-235b-a22b-2507", "xiaomi/mimo-v2.6-flash"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, fam, title in ((axes[0], "comm_parity", "Parity"), (axes[1], "comm_ip", "Inner product")):
    for model in SWEEP:
        accs = h7[f"openai/{model}/{fam}"]["accuracy_by_L"]
        ax.plot([float(L) for L in accs], list(accs.values()), marker="o", label=name(model))
    if fam == "comm_ip":
        L = np.arange(0, 17)
        ax.plot(L, 0.5 + 0.5 * 2.0 ** -(16 - L), color="gray", linestyle="--",
                label="Best possible with this message format")
    ax.set_title(title)
    ax.set_xlabel("Message size (bits, out of 16)")
    ax.set_xticks([0, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
axes[0].set_ylabel("Accuracy")
axes[1].legend(loc="upper left")
fig.suptitle("Parity is solved with a 1-bit message; the inner product needs the whole list")
fig.tight_layout()
save(fig, "4_accuracy_vs_message_bits.png")

# 5. Agent teams (H5a-H5c): each team against one agent spending the same tokens
tt = pd.read_csv(TEAMS / "team_tests.csv")
mech = pd.read_csv(TEAMS / "team_mechanism.csv")
MODELS = {
    "openai/mistralai/mistral-small-3.2-24b-instruct": "Mistral Small 3.2",
    "openai/z-ai/glm-4.7-flash": "GLM 4.7 flash",
    "openai/xiaomi/mimo-v2.6-flash": "MiMo v2.6 flash",
}
TEAM_ARMS = {"L": ("Lead + workers", "C0"), "W": ("Swarm (shared notes, vote)", "C1"),
             "R": ("Author + reviewer", "C2")}
fig, (left, right) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.15, 1]})
left.axvspan(-0.05, 0.05, color="gray", alpha=0.15, label="Within ±0.05")
left.axvline(0, color="gray", linewidth=1)
ticks, labels, y = [], [], 0
for model, mname in MODELS.items():
    for arm, (alabel, color) in TEAM_ARMS.items():
        r = tt[(tt.model == model) & (tt.arm == arm)].iloc[0]
        lo, hi = json.loads(r.ci95)
        left.errorbar(r["diff"], y, xerr=[[r["diff"] - lo], [hi - r["diff"]]], fmt="o",
                      color=color, capsize=3, label=alabel if model == next(iter(MODELS)) else None)
        left.annotate(f"{r['diff']:+.2f}", (hi, y), xytext=(5, 0), textcoords="offset points",
                      va="center")
        ticks.append(y)
        labels.append(f"{mname}, {arm}")
        y += 1
    y += 0.6
left.set_yticks(ticks, labels)
left.invert_yaxis()
left.set_xlim(-0.25, 0.17)
left.set_xlabel("Team accuracy minus one agent with the same token budget (95% interval)")
left.set_title("No team beat one agent with the same token budget")
short = {"Mistral Small 3.2": "Mistral", "GLM 4.7 flash": "GLM", "MiMo v2.6 flash": "MiMo"}
right.axhline(0, color="gray", linewidth=1)
for arm, (alabel, color) in TEAM_ARMS.items():
    sub = mech[mech.arm == arm]
    right.scatter(sub.accepted_share, sub.deficit, color=color)
    for _, r in sub.iterrows():
        right.annotate(f"{short[MODELS[r.model]]} {arm}", (r.accepted_share, r.deficit),
                       xytext=(6, -3), textcoords="offset points")
rho, pval = spearmanr(mech.accepted_share, mech.deficit)
right.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
right.set_xlim(0, 0.9)
right.set_xlabel("How often the answer agreed with a teammate's wrong claim")
right.set_ylabel("How far the team fell behind (accuracy)")
right.set_title("Teams that went along with wrong claims tended to fall further behind\n"
                f"(Spearman ρ = {rho:.2f}, p = {pval:.2f}, over 9 teams)")
fig.legend(*left.get_legend_handles_labels(), ncol=4, loc="upper center",
           bbox_to_anchor=(0.5, 0.0))
fig.tight_layout()
save(fig, "5_agent_teams.png")

print("\n".join(str(f) for f in sorted(out.glob("*.png"))))
