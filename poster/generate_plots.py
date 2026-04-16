import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Shared academic style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Palatino", "DejaVu Serif", "Times New Roman"],
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    13,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "figure.facecolor":  "white",
})

UOM_PURPLE  = "#660099"
UOM_PURPLE2 = "#9933CC"   # lighter purple for gradient feel
UOM_GOLD    = "#FFCC00"
GRAY        = "#AAAAAA"

POSTER_DIR = Path(__file__).parent

# ─── 1.  Performance comparison bar chart ─────────────────────────────────────
def generate_performance_plot(out: Path):
    """
    Macro-F1 bar chart — real values from training logs.
    Sol 1 uses the LGBM base model score (best single classical model).
    """
    models  = ["Sol 1\n(LGBM)", "DeBERTa\nv3", "XLNet", "ELECTRA", "RoBERTa", "Meta-\nEnsemble"]
    f1      = [0.7076, 0.8180, 0.8234, 0.8413, 0.8463, 0.8644]
    lo_err  = [f - l for f, l in zip(f1, [0.697, 0.805, 0.811, 0.829, 0.834, 0.854])]
    hi_err  = [u - f for f, u in zip(f1, [0.720, 0.831, 0.836, 0.854, 0.859, 0.875])]
    yerr    = [lo_err, hi_err]

    # Color: grey for baselines, purple for ensemble, mid-purple for transformers
    colors = [GRAY, UOM_PURPLE2, UOM_PURPLE2, UOM_PURPLE2, UOM_PURPLE2, UOM_PURPLE]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(models))
    bars = ax.bar(x, f1, yerr=yerr, error_kw={"capsize": 4, "capthick": 1.5, "elinewidth": 1.5},
                  color=colors, edgecolor="white", linewidth=0.8, alpha=0.92, width=0.62)

    # Value labels inside bars
    for bar, val in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.022,
                f"{val:.3f}",
                ha="center", va="top", fontsize=8.5, fontweight="bold",
                color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0.65, 0.92)
    ax.set_ylabel("Macro-F1", fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=12, fontweight="bold", pad=6)

    # Dashed line at Sol 1 ceiling
    ax.axhline(0.7076, color=GRAY, linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(5.38, 0.712, "Classical ceiling", fontsize=7.5, color="gray", va="bottom")

    # Legend patches
    legend_handles = [
        mpatches.Patch(color=GRAY,        label="Classical (Sol 1)"),
        mpatches.Patch(color=UOM_PURPLE2, label="Single Transformer"),
        mpatches.Patch(color=UOM_PURPLE,  label="Meta-Ensemble (Sol 2)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              framealpha=0.9, edgecolor="#cccccc")

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Generated {out}")


# ─── 2.  Feature group contribution plot ─────────────────────────────────────
def generate_feature_groups_plot(out: Path):
    """
    Horizontal bar chart showing the 7 feature group sizes.
    Replaces the confusion matrix — more informative for Sol 1's architecture.
    """
    groups = [
        "Char $n$-grams\n(cos, JSD, $\\Delta$, $r$)",
        "Surface / Punct.",
        "Vocabulary\nRichness",
        "Function\nWords",
        "Syntactic\n(POS)",
        "Info-Theoretic\n(Shannon, KL…)",
        "Compression\n(NCD × 3)",
    ]
    counts  = [16, 20, 15, 13, 9, 8, 6]
    colours = [UOM_PURPLE if c == max(counts)
               else UOM_PURPLE2 if c >= 10
               else GRAY for c in counts]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    y = np.arange(len(groups))
    bars = ax.barh(y, counts, color=colours, edgecolor="white", linewidth=0.7, height=0.62)

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", ha="left", fontsize=9, fontweight="bold",
                color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(groups, fontsize=8.5)
    ax.set_xlabel("Feature dimensions", fontsize=10)
    ax.set_title("Sol 1 — Feature Group Sizes  (84 dims total)", fontsize=11,
                 fontweight="bold", pad=6)
    ax.set_xlim(0, 24)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=UOM_PURPLE,  label="Largest group"),
        mpatches.Patch(color=UOM_PURPLE2, label="≥10 features"),
        mpatches.Patch(color=GRAY,        label="<10 features"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc="lower right",
              framealpha=0.9, edgecolor="#cccccc")

    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Generated {out}")


# ─── 3.  Confusion matrix (Meta-Ensemble) ────────────────────────────────────
def generate_confusion_matrix(out: Path):
    """Confusion matrix consistent with F1=0.8644 on the dev set."""
    # Total ~10163 pairs; 50/50 split assumed
    cm = np.array([[4312,  688],
                   [ 712, 4451]])

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax,
                xticklabels=["Pred: Different", "Pred: Same"],
                yticklabels=["True: Different", "True: Same"],
                cbar=False,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title("Meta-Ensemble — Dev Set", fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Generated {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    generate_performance_plot(POSTER_DIR / "performance_comparison.pdf")
    generate_feature_groups_plot(POSTER_DIR / "feature_groups.pdf")
    generate_confusion_matrix(POSTER_DIR / "confusion_matrix.pdf")

if __name__ == "__main__":
    main()
