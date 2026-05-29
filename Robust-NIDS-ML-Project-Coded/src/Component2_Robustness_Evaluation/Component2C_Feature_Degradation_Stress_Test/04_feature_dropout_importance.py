"""
Component 2C - Feature Dropout Importance

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C2C-4

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C2C-4: Identify single-feature dropout impact
# ========================================================================================

# Cell C2C-4: Identify single-feature dropout impact
# Purpose:
# - Drop one raw feature group at a time
# - Rank features by clean-to-drop macro F1 loss

clean_macro_f1 = c2c_clean_baseline_df.loc[0, "macro_f1"]
single_feature_dropout_rows = []

for raw_feature in dropout_candidate_features:
    X_degraded, dropped_processed_count = make_feature_dropout_matrix(
        X_test_processed,
        [raw_feature]
    )
    
    metrics = evaluate_multiclass_prediction(
        selected_baseline_model,
        X_degraded
    )
    
    single_feature_dropout_rows.append({
        "dropped_raw_feature": raw_feature,
        "dropped_processed_features": dropped_processed_count,
        "macro_f1_after_dropout": metrics["macro_f1"],
        "macro_f1_drop": clean_macro_f1 - metrics["macro_f1"],
        "accuracy_after_dropout": metrics["accuracy"],
        "weighted_f1_after_dropout": metrics["weighted_f1"]
    })
    
    del X_degraded
    gc.collect()

c2c_single_feature_dropout_df = (
    pd.DataFrame(single_feature_dropout_rows)
    .sort_values(by="macro_f1_drop", ascending=False)
)

display(c2c_single_feature_dropout_df.head(15).round(4))

