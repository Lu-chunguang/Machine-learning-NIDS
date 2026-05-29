"""
Component 3C - Degradation Curves

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3C-7

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3C-7: Plot original vs augmented degradation curves
# ========================================================================================

# Cell C3C-7: Plot original vs augmented degradation curves
# Purpose:
# - Visualize Strategy 5 performance under all degradation types

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, degradation_type in zip(
    axes,
    ["gaussian_noise", "random_masking", "feature_dropout"]
):
    subset = c3c_stress_comparison_df[
        c3c_stress_comparison_df["degradation_type"] == degradation_type
    ].copy()
    
    ax.plot(
        subset["level"],
        subset["baseline_macro_f1"],
        marker="o",
        label="Original baseline"
    )
    ax.plot(
        subset["level"],
        subset["augmented_macro_f1"],
        marker="o",
        label=f"Augmented: {c3c_primary_augmented_model_name}"
    )
    ax.set_title(degradation_type)
    ax.set_xlabel("Degradation level")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

