"""
Component 2C - Baseline Degradation Curves

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C2C-5

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C2C-5: Plot baseline degradation curves
# ========================================================================================

# Cell C2C-5: Plot baseline degradation curves
# Purpose:
# - Visualize macro-F1 degradation for Gaussian noise, masking, and feature dropout

plot_df = c2c_degradation_summary_df[
    c2c_degradation_summary_df["degradation_type"] != "clean"
].copy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, degradation_type in zip(
    axes,
    ["gaussian_noise", "random_masking", "feature_dropout"]
):
    subset = plot_df[plot_df["degradation_type"] == degradation_type].copy()
    
    ax.errorbar(
        subset["level"],
        subset["macro_f1_mean"],
        yerr=subset["macro_f1_std"].fillna(0),
        marker="o",
        capsize=4
    )
    ax.axhline(
        c2c_clean_baseline_df.loc[0, "macro_f1"],
        linestyle="--",
        color="gray",
        label="clean baseline"
    )
    ax.set_title(degradation_type)
    ax.set_xlabel("Degradation level")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

