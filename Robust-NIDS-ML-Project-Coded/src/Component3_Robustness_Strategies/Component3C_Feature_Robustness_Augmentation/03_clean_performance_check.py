"""
Component 3C - Clean Performance Check

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3C-3

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3C-3: Clean performance check for augmented models
# ========================================================================================

# Cell C3C-3: Clean performance check for augmented models
# Purpose:
# - Verify whether augmentation hurts clean test performance
# - Compare all variants against the original baseline

c3c_clean_rows = [{
    "model": "baseline",
    "accuracy": c2c_clean_baseline_df.loc[0, "accuracy"],
    "macro_f1": c2c_clean_baseline_df.loc[0, "macro_f1"],
    "weighted_f1": c2c_clean_baseline_df.loc[0, "weighted_f1"]
}]

for model_name, model in c3c_augmented_models.items():
    metrics = evaluate_multiclass_prediction(model, X_test_processed)
    c3c_clean_rows.append({
        "model": model_name,
        **metrics
    })

c3c_clean_performance_df = pd.DataFrame(c3c_clean_rows)
display(c3c_clean_performance_df.round(4))

