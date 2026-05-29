"""
Component 3C - Original vs Augmented Degradation Comparison

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3C-5

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3C-5: Compare original and augmented models under Stress Test C
# ========================================================================================

# Cell C3C-5: Compare original and augmented models under Stress Test C
# Purpose:
# - Report the proposal-required original vs augmented degradation comparison
# - Calculate macro-F1 gain for the primary augmented classifier

c3c_stress_comparison_df = (
    c2c_degradation_summary_df[[
        "degradation_type",
        "level",
        "macro_f1_mean",
        "weighted_f1_mean"
    ]]
    .rename(columns={
        "macro_f1_mean": "baseline_macro_f1",
        "weighted_f1_mean": "baseline_weighted_f1"
    })
    .merge(
        c3c_primary_summary_df[[
            "degradation_type",
            "level",
            "macro_f1_mean",
            "weighted_f1_mean"
        ]].rename(columns={
            "macro_f1_mean": "augmented_macro_f1",
            "weighted_f1_mean": "augmented_weighted_f1"
        }),
        on=["degradation_type", "level"],
        how="inner"
    )
)

c3c_stress_comparison_df["macro_f1_gain"] = (
    c3c_stress_comparison_df["augmented_macro_f1"] -
    c3c_stress_comparison_df["baseline_macro_f1"]
)

c3c_stress_comparison_df["weighted_f1_gain"] = (
    c3c_stress_comparison_df["augmented_weighted_f1"] -
    c3c_stress_comparison_df["baseline_weighted_f1"]
)

display(c3c_stress_comparison_df.round(4))

