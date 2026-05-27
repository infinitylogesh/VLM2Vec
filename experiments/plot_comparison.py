"""
Generates comparison figures matching the benchmark charts, with our
Qwen3-0.8B VLM2Vec checkpoint results added.

Usage:
    uv run python experiments/plot_comparison.py
Outputs:
    experiments/figures/chart1_task_comparison.png
    experiments/figures/chart2_modality_bars.png
    experiments/figures/chart3_scatter.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs("experiments/figures", exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────
C_SMALL  = "#C0392B"   # v5-omni-small  (dark red)
C_NANO   = "#E8998D"   # v5-omni-nano   (salmon)
C_GRAY1  = "#B0B0B0"   # LanguageBind
C_GRAY2  = "#999999"   # Nemotron-3B
C_GRAY3  = "#777777"   # LCO-3B
C_GRAY4  = "#555555"   # LCO-7B
C_OURS   = "#2980B9"   # VLM2Vec-Qwen3 (blue)

# ── Exact values from the benchmark charts ─────────────────────────────────
# image.png  /  screenshot_2.png bottom panels
MODALITY_DATA = {
    #                              Text    Image   Video   Audio
    #                             (MMTEB) (MIEB)  (MMEBv) (MAEB)
    "v5-omni-small\n(1.57B)":  (67.0,   56.0,   41.2,   51.5,  C_SMALL),
    "v5-omni-nano\n(0.95B)":   (65.5,   44.4,   26.9,   44.0,  C_NANO),
    "LanguageBind\n(1.14B)":   (27.3,   47.8,   48.1,   20.1,  C_GRAY1),
    "Nemotron-3B\n(4.70B)":    (47.6,   44.5,   24.5,   48.3,  C_GRAY2),
    "LCO-3B\n(4.07B)":         (57.5,   58.4,   46.8,   52.5,  C_GRAY3),
    "LCO-7B\n(8.93B)":         (59.3,   58.6,   47.4,   52.4,  C_GRAY4),
    # Our model — Image and Video only (text/audio not evaluated)
    "VLM2Vec-Qwen3\n(0.8B, ours)": (None, 59.1,  28.9,  None,  C_OURS),
}

MODALITY_TITLES = ["Text\n(MMTEB)", "Image\n(MIEB)", "Video\n(MMEB-Video)", "Audio\n(MAEB)"]
MODALITY_IDX    = [0, 1, 2, 3]

# ── Figure 1 – 4-panel modality bar chart (replicates image.png) ───────────
fig1, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=False)
fig1.subplots_adjust(wspace=0.35)

model_keys  = list(MODALITY_DATA.keys())
short_labels = [k.split("\n")[0] for k in model_keys]
x = np.arange(len(model_keys))
bar_w = 0.65

for col, (ax, title, mi) in enumerate(zip(axes, MODALITY_TITLES, MODALITY_IDX)):
    for xi, key in enumerate(model_keys):
        row = MODALITY_DATA[key]
        score = row[mi]
        color = row[4]
        if score is None:
            # draw a hatched placeholder bar to indicate "not evaluated"
            ax.bar(xi, 0, bar_w, color="none",
                   edgecolor=color, linewidth=1.5, linestyle="--", zorder=3)
            ax.text(xi, 1.5, "N/A", ha="center", va="bottom",
                    fontsize=8, color=color, style="italic")
        else:
            bar = ax.bar(xi, score, bar_w, color=color, zorder=3,
                         edgecolor="white", linewidth=0.5)
            ax.text(xi, score + 0.8, f"{score}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#222222")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([k.split("\n")[0] + "\n" + k.split("\n")[1]
                        if "\n" in k else k
                        for k in model_keys],
                       rotation=35, ha="right", fontsize=7.5)
    ax.set_ylim(0, 80)
    ax.set_ylabel("Score", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    # Highlight our bar with a subtle border
    ax.bar(len(model_keys) - 1,
           MODALITY_DATA["VLM2Vec-Qwen3\n(0.8B, ours)"][mi] or 0,
           bar_w + 0.05, color="none",
           edgecolor=C_OURS, linewidth=2, zorder=4)

fig1.suptitle("Per-Modality Benchmark Scores — VLM2Vec-Qwen3-0.8B vs Baselines",
              fontsize=13, fontweight="bold", y=1.01)
fig1.savefig("experiments/figures/chart2_modality_bars.png",
             dpi=180, bbox_inches="tight")
print("Saved → experiments/figures/chart2_modality_bars.png")


# ── Figure 2 – Task-level horizontal bar chart (replicates screenshot_1 top)
# Values read directly from screenshot_1.png
TASK_DATA = {
    # task: {model: score}   — None means not evaluated by that model
    "Image Classification": {
        "v5-omni-small (1.57B)": 68.5,
        "v5-omni-nano (0.95B)":  None,
        "Best Baseline":          64.3,
        "VLM2Vec-Qwen3\n(0.8B, ours)": 54.0,
    },
    "Image Clustering": {
        "v5-omni-small (1.57B)": 84.6,
        "v5-omni-nano (0.95B)":  None,
        "Best Baseline":          80.2,
        "VLM2Vec-Qwen3\n(0.8B, ours)": None,   # not in our eval suite
    },
    "Visual STS": {
        "v5-omni-small (1.57B)": 58.8,
        "v5-omni-nano (0.95B)":  None,
        "Best Baseline":          79.6,
        "VLM2Vec-Qwen3\n(0.8B, ours)": None,
    },
    "Image Retrieval": {
        "v5-omni-small (1.57B)": 38.5,
        "v5-omni-nano (0.95B)":  21.6,
        "Best Baseline":          46.3,
        "VLM2Vec-Qwen3\n(0.8B, ours)": 63.2,
    },
    "Document Retrieval": {
        "v5-omni-small (1.57B)": 79.1,
        "v5-omni-nano (0.95B)":  70.0,
        "Best Baseline":          85.6,
        "VLM2Vec-Qwen3\n(0.8B, ours)": None,   # visdoc eval pending
    },
    "Compositional / VQA": {
        "v5-omni-small (1.57B)": 44.2,
        "v5-omni-nano (0.95B)":  None,
        "Best Baseline":          53.4,
        "VLM2Vec-Qwen3\n(0.8B, ours)": 57.6,
    },
    "Video Classification": {
        "v5-omni-small (1.57B)": 42.7,
        "v5-omni-nano (0.95B)":  37.8,
        "Best Baseline":          78.4,
        "VLM2Vec-Qwen3\n(0.8B, ours)": 29.7,
    },
    "Video QA": {
        "v5-omni-small (1.57B)": 44.5,
        "v5-omni-nano (0.95B)":  36.8,
        "Best Baseline":          71.3,
        "VLM2Vec-Qwen3\n(0.8B, ours)": 31.2,
    },
    "Video Retrieval": {
        "v5-omni-small (1.57B)": 27.8,
        "v5-omni-nano (0.95B)":  14.5,
        "Best Baseline":          58.7,
        "VLM2Vec-Qwen3\n(0.8B, ours)": 19.3,
    },
    "Video Moment Ret.": {
        "v5-omni-small (1.57B)": 47.2,
        "v5-omni-nano (0.95B)":  43.0,
        "Best Baseline":          58.1,
        "VLM2Vec-Qwen3\n(0.8B, ours)": 39.5,
    },
    "Audio Classification": {
        "v5-omni-small (1.57B)": 56.9,
        "v5-omni-nano (0.95B)":  22.3,
        "Best Baseline":          53.4,
        "VLM2Vec-Qwen3\n(0.8B, ours)": None,
    },
    "Audio Retrieval": {
        "v5-omni-small (1.57B)": 61.7,
        "v5-omni-nano (0.95B)":  39.3,
        "Best Baseline":          None,
        "VLM2Vec-Qwen3\n(0.8B, ours)": None,
    },
    "Audio Text Match": {
        "v5-omni-small (1.57B)": 62.0,
        "v5-omni-nano (0.95B)":  46.5,
        "Best Baseline":          47.3,
        "VLM2Vec-Qwen3\n(0.8B, ours)": None,
    },
}

TASK_MODEL_ORDER = [
    "v5-omni-small (1.57B)",
    "v5-omni-nano (0.95B)",
    "Best Baseline",
    "VLM2Vec-Qwen3\n(0.8B, ours)",
]
TASK_COLORS = {
    "v5-omni-small (1.57B)":        C_SMALL,
    "v5-omni-nano (0.95B)":         C_NANO,
    "Best Baseline":                 "#95A5A6",
    "VLM2Vec-Qwen3\n(0.8B, ours)":  C_OURS,
}

tasks     = list(TASK_DATA.keys())
n_tasks   = len(tasks)
n_models  = len(TASK_MODEL_ORDER)
bar_h     = 0.16
gap       = 0.03
group_h   = n_models * (bar_h + gap)

fig2, ax2 = plt.subplots(figsize=(13, 9))

for ti, task in enumerate(tasks):
    y_base = ti * (group_h + 0.12)
    for mi, model in enumerate(TASK_MODEL_ORDER):
        score = TASK_DATA[task].get(model)
        y     = y_base + mi * (bar_h + gap)
        color = TASK_COLORS[model]
        if score is not None:
            ax2.barh(y, score, height=bar_h, color=color,
                     edgecolor="white", linewidth=0.4, zorder=3)
            ax2.text(score + 0.5, y, f"{score}",
                     va="center", ha="left", fontsize=7.5, color="#333333")
        else:
            # dashed outline to indicate "N/A"
            ax2.barh(y, 2, height=bar_h, color="none",
                     edgecolor=color, linewidth=1, linestyle=":", zorder=3)

y_ticks = [ti * (group_h + 0.12) + (n_models * (bar_h + gap) - bar_h) / 2
           for ti in range(n_tasks)]
ax2.set_yticks(y_ticks)
ax2.set_yticklabels(tasks, fontsize=10, fontweight="bold")
ax2.set_xlabel("Score", fontsize=11)
ax2.set_xlim(0, 105)
ax2.set_title("Task-Level Performance — VLM2Vec-Qwen3-0.8B vs Baselines\n"
              "(dotted = not evaluated)",
              fontsize=13, fontweight="bold", pad=12)
ax2.spines[["top", "right"]].set_visible(False)
ax2.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax2.set_axisbelow(True)

legend_handles = [mpatches.Patch(color=TASK_COLORS[m], label=m.replace("\n", " "))
                  for m in TASK_MODEL_ORDER]
ax2.legend(handles=legend_handles, loc="lower right", fontsize=9,
           framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()
fig2.savefig("experiments/figures/chart1_task_comparison.png",
             dpi=180, bbox_inches="tight")
print("Saved → experiments/figures/chart1_task_comparison.png")


# ── Figure 3 – Scatter: params vs avg score ────────────────────────────────
# Avg = (Text + Image + Video + Audio) / 4  for reference models
# For our model: (Image + Video) / 2 — clearly labelled
SCATTER_MODELS = [
    # (params_B, avg_score, label, color, marker, alpha)
    (1.57, 53.9, "v5-omni-small\n(53.9)",  C_SMALL,  "D", 1.0),
    (0.95, 45.2, "v5-omni-nano\n(45.2)",   C_NANO,   "D", 1.0),
    (1.14, 36.3, "LanguageBind\n(36.3)",   C_GRAY1,  "o", 1.0),
    (4.70, 41.2, "Nemotron-3B\n(41.2)",    C_GRAY2,  "o", 1.0),
    (4.07, 53.8, "LCO-3B\n(53.8)",         C_GRAY3,  "o", 1.0),
    (8.93, 54.4, "LCO-7B\n(54.4)",         C_GRAY4,  "o", 1.0),
    # Ours: (59.1 + 28.9) / 2 = 44.0  [Image + Video only]
    (0.80, 44.0, "VLM2Vec-Qwen3-0.8B\n(44.0, img+vid only)", C_OURS, "s", 1.0),
]

fig3, ax3 = plt.subplots(figsize=(9, 6))

# Pareto frontier (reference models, excluding ours for a clean line)
ref_pts = [(p, s) for (p, s, *_) in SCATTER_MODELS if _ [1] != C_OURS]
pareto = []
best = -1
for pt in sorted(ref_pts):
    if pt[1] > best:
        pareto.append(pt)
        best = pt[1]
ax3.plot([p[0] for p in pareto], [p[1] for p in pareto],
         "D--", color=C_NANO, alpha=0.45, linewidth=1.5, zorder=2,
         label="Pareto frontier")

for (params, avg, label, color, marker, alpha) in SCATTER_MODELS:
    ax3.scatter(params, avg, s=130, color=color, marker=marker,
                zorder=5, edgecolors="white", linewidths=0.9, alpha=alpha)
    nudge = (0.15, 3) if "Qwen3" in label else (0.12, 3)
    ax3.annotate(label, (params, avg),
                 textcoords="offset points", xytext=(nudge[0]*50, nudge[1]),
                 fontsize=8,
                 color=color,
                 fontweight="bold" if "Qwen3" in label else "normal")

ax3.set_xlabel("Total Parameters (B)", fontsize=11)
ax3.set_ylabel("Avg Score  (Text + Image + Video + Audio)", fontsize=11)
ax3.set_title("Parameter Efficiency\n"
              "* VLM2Vec-Qwen3 avg computed over Image + Video only (text/audio not evaluated)",
              fontsize=11, fontweight="bold")
ax3.spines[["top", "right"]].set_visible(False)
ax3.grid(True, linestyle="--", alpha=0.35)
ax3.set_xlim(-0.5, 10.5)
ax3.set_ylim(30, 62)

fig3.savefig("experiments/figures/chart3_scatter.png",
             dpi=180, bbox_inches="tight")
print("Saved → experiments/figures/chart3_scatter.png")

plt.close("all")
print("\nAll 3 charts saved to experiments/figures/")
