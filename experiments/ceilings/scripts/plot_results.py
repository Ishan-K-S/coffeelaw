"""Figures for RESULTS.md from the committed outputs in results/paid-2026-10/.

Reads tests.csv, gain_table.csv and predictions.json only (no raw runs), so it runs anywhere
matplotlib and pandas are installed.

    python scripts/plot_results.py --results results/paid-2026-10
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--results", default="results/paid-2026-10")
p.add_argument("--out", default=None, help="default: <results>/figures")
a = p.parse_args()
res = Path(a.results)
out = Path(a.out) if a.out else res / "figures"
out.mkdir(parents=True, exist_ok=True)

# fixed categorical order; marker shapes back up color
BLUE, ORANGE, AQUA, YELLOW, VIOLET, MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e87ba4")
GRAY, INK, MUTED, GRID = "#b4b2a9", "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
    "savefig.bbox": "tight", "lines.linewidth": 2, "lines.markersize": 6,
})
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


def pct(ax, axis="y"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


tests = pd.read_csv(res / "tests.csv")
gains = pd.read_csv(res / "gain_table.csv")
pred = {x["prediction_id"]: x for x in json.loads((res / "predictions.json").read_text())}
FAMILIES = [("info", "Information"), ("reason", "Reasoning"),
            ("comm_parity", "Comm: parity"), ("comm_ip", "Comm: inner product")]
MATCH = {"info": "E", "reason": "K", "comm_parity": "M", "comm_ip": "M"}

# 1. which resource lifts which ceiling: mean gain over baseline, medium delta
arms = [("E", "E  evidence"), ("K", "K  reasoning"), ("M", "M  messages"),
        ("A1", "A1  more agents"), ("P1", "P1  irrelevant text"), ("P2", "P2  blank filler")]
vs_base = tests[tests["reference"] == "baseline"]
grid = np.array([[vs_base[(vs_base.family == f) & (vs_base.arm == arm)].gain.mean()
                  for f, _ in FAMILIES] for arm, _ in arms])
n_models = [vs_base[vs_base.family == f].model.nunique() for f, _ in FAMILIES]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f0efec", BLUE])
ax.imshow(grid, cmap=cmap, vmin=0, vmax=0.85, aspect="auto")
ax.grid(False)
for i, (arm, _) in enumerate(arms):
    for j, (f, _) in enumerate(FAMILIES):
        g = grid[i, j]
        ax.text(j, i, f"{g:+.2f}".replace("-", "−"), ha="center", va="center",
                color="white" if g > 0.4 else INK, weight="bold" if MATCH[f] == arm else "normal")
        if f == "reason" and arm == "M":
            ax.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False,
                                       edgecolor=ORANGE, linewidth=2.5))
ax.set_xticks(range(4), [f"{lab}\n{n} models" for (_, lab), n in zip(FAMILIES, n_models, strict=True)])
ax.set_yticks(range(len(arms)), [lab for _, lab in arms])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Each ceiling lifts only with its own resource\n"
             "mean gain over baseline, medium budget, 850 paired instances per model",
             loc="left", fontsize=10.5)
fig.text(0.01, -0.02, "Orange box: M on reasoning helps because agent B writes the solved chain "
         "into its message (a design flaw, see RESULTS.md).", fontsize=8.5, color="#52514e")
fig.savefig(out / "1_gain_fingerprint.png")
plt.close(fig)

# 2. H7: accuracy against message bits
h7 = {k: v for k, v in pred["H7"]["detail"].items() if k.count("/") >= 2}
SWEEP = [("deepseek/deepseek-v4-flash", BLUE, "o"), ("nvidia/nemotron-3.5-super-vl", ORANGE, "s"),
         ("qwen/qwen3-235b-a22b-2507", AQUA, "^"), ("xiaomi/mimo-v2.6-flash", YELLOW, "D")]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, fam, title in ((axes[0], "comm_parity", "Parity"), (axes[1], "comm_ip", "Inner product")):
    for model, color, marker in SWEEP:
        acc = h7[f"openai/{model}/{fam}"]["accuracy_by_L"]
        L = [float(x) for x in acc]
        ax.plot(L, list(acc.values()), color=color, marker=marker, label=name(model),
                markeredgecolor="white", markeredgewidth=1)
    if fam == "comm_ip":
        Ls = np.arange(0, 17)
        ax.plot(Ls, 0.5 + 0.5 * 2.0 ** -(16 - Ls), color=MUTED, ls="--", lw=1.5,
                label="best possible, prefix message")
    ax.set_title(f"{title}: accuracy vs message bits", loc="left", fontsize=10.5)
    ax.set_xlabel("message budget L (bits, of 16)")
    ax.set_xticks([0, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    ax.set_ylim(0.3, 1.03)
    ax.grid(axis="x", visible=False)
    pct(ax)
axes[0].set_ylabel("accuracy")
axes[1].legend(loc="upper left", fontsize=8.5)
fig.suptitle("H7: parity needs 1 bit; the inner product needs the whole list",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(out / "2_h7_message_bits.png")
plt.close(fig)

# 3. H6: error against probes, mean of the sweep models, with the Fano floor
cells = pd.DataFrame(pred["H6"]["detail"]["cells"])
fig, ax = plt.subplots(figsize=(7.2, 4))
for flip, color, marker, lab in ((0.0, BLUE, "o", "clean probes"),
                                 (0.1, VIOLET, "s", "10% of probes flipped"),
                                 (0.2, MAGENTA, "^", "20% flipped")):
    c = cells[cells.p == flip].groupby("T").agg(error=("error", "mean"), floor=("floor", "first"))
    ax.plot(c.index, c.error, color=color, marker=marker, label=f"{lab}: models",
            markeredgecolor="white", markeredgewidth=1)
    ax.plot(c.index, c.floor, color=color, ls="--", lw=1.5, label=f"{lab}: Fano floor")
ax.set_xlabel("probes T (16 hosts, 4 bits to identify)")
ax.set_ylabel("error")
ax.set_xticks(sorted(cells["T"].unique()))
ax.set_ylim(-0.02, 1.0)
ax.grid(axis="x", visible=False)
pct(ax)
ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
ax.set_title("H6: clean probes reach the floor at T = 4; noisy probes stay far above it\n"
             "mean error of DeepSeek, Nemotron, Qwen, MiMo; 100 instances per cell",
             loc="left", fontsize=10.5)
fig.savefig(out / "3_h6_probes_vs_fano.png")
plt.close(fig)

# 4. A2: the same reasoning compute split across agents vs given to one agent
rs = gains[(gains.family == "reason") & (gains.delta == "medium")]
acc = rs.pivot(index="model", columns="arm", values="arm_acc")
acc["baseline"] = rs.groupby("model").baseline_acc.first()
acc = acc.assign(gap=acc.A2_single_ref - acc.A2).sort_values("gap")
y = np.arange(len(acc))
fig, ax = plt.subplots(figsize=(7.2, 4.6))
h = 0.26
for off, col, color, lab in ((h, "A2_single_ref", BLUE, "one agent (A2 single ref)"),
                             (0, "A2", ORANGE, "split across agents (A2)"),
                             (-h, "baseline", GRAY, "baseline")):
    ax.barh(y + off, acc[col], height=h * 0.9, color=color, label=lab)
for yi, (_, r) in zip(y, acc.iterrows(), strict=True):
    ax.text(max(r.A2_single_ref, r.A2) + 0.015, yi + h / 2, f"−{r.gap * 100:.0f} pts"
            if r.gap > 0.005 else "no gap", va="center", fontsize=8.5, color="#52514e")
ax.set_yticks(y, [name(m) for m in acc.index])
ax.set_xlim(0, 1.12)
ax.grid(axis="y", visible=False)
pct(ax, "x")
ax.set_xlabel("accuracy on the reasoning task")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=8.5, ncol=3, loc="upper center",
          bbox_to_anchor=(0.45, -0.12))
ax.set_title("H5: splitting the same compute across agents loses up to 35 points\n"
             "reasoning family, medium budget, 850 paired instances", loc="left", fontsize=10.5)
fig.savefig(out / "4_a2_split_vs_single.png")
plt.close(fig)

# 5. delta curves: the matching arm's gain at each budget level
curve_models = ["google/gemma-3-27b-it", "mistralai/mistral-small-3.2-24b-instruct",
                "xiaomi/mimo-v2.6-flash"]
STYLE = {"info": (BLUE, "o"), "reason": (ORANGE, "^"), "comm_parity": (AQUA, "s"),
         "comm_ip": (AQUA, "D")}
levels = ["small", "medium", "large"]
GROUP = {"info": "E on information", "reason": "K on reasoning",
         "comm_parity": "M on communication", "comm_ip": "M on communication"}
fig, ax = plt.subplots(figsize=(7.2, 4))
ends = {}
for model in curve_models:
    m = f"openai/{model}"
    for fam, _ in FAMILIES:
        c = gains[(gains.model == m) & (gains.family == fam) & (gains.arm == MATCH[fam])]
        c = c.set_index("delta").reindex(levels)
        # Gemma's reason curve was stopped at about 55 instances per cell (see DEVIATIONS.md)
        if c.gain.isna().any() or (c.n < 850).any():
            continue
        color, marker = STYLE[fam]
        ax.plot(levels, c.gain, color=color, marker=marker, markeredgecolor="white",
                markeredgewidth=1, alpha=0.9)
        ends.setdefault(GROUP[fam], []).append((c.gain.iloc[-1], name(model).split()[0]))
for group, pts in ends.items():
    who = ", ".join(sorted({w for _, w in pts}))
    ax.annotate(f"{group}\n{len(pts)} curves: {who}", (2, np.mean([g for g, _ in pts])),
                xytext=(10, 0), textcoords="offset points", va="center", fontsize=8,
                color="#52514e")
ax.set_xticks(range(3), ["small (0.5×)", "medium (1×)", "large (2×)"])
ax.set_xlim(-0.2, 2.9)
ax.set_ylim(0, 0.9)
ax.set_ylabel("gain over baseline")
ax.grid(axis="x", visible=False)
ax.set_title("Delta curves: the gain is the same at half and double the budget\n"
             "the useful resource is already at its maximum at small; the rest is filler",
             loc="left", fontsize=10.5)
fig.savefig(out / "5_delta_curves.png")
plt.close(fig)

print("\n".join(str(f) for f in sorted(out.glob("*.png"))))
