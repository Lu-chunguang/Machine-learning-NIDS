"""
Component 3C - Augmented Degradation Stress Test

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3C-4

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3C-4: Re-run full Stress Test C on the primary augmented model
# ========================================================================================

# Cell C3C-4: Re-run full Stress Test C on the primary augmented model
# Purpose:
# - Satisfy Strategy 5 by re-running Stress Test C after augmented training
# - Compare the augmented classifier against the original baseline under identical degradation

c3c_primary_detail_df, c3c_primary_summary_df = run_degradation_suite(
    c3c_primary_augmented_model,
    c3c_primary_augmented_model_name
)

display(c3c_primary_summary_df.round(4))

