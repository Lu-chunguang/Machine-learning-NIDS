"""
Component 3B - Few-Shot Holdout Evaluation

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3B-9, C3B-10

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3B-9: Evaluate few-shot calibrated model on CIC holdout
# ========================================================================================

# Cell C3B-9: Evaluate few-shot calibrated model on CIC holdout
# Purpose:
# - Test the few-shot calibrated model on CIC holdout data
# - Compare with source-only and cost-sensitive results
# - Keep holdout samples completely unseen during training

y_cic_holdout_fewshot_pred = c3b_fewshot_xgb.predict(
    X_cic_holdout_processed
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_cic_holdout_binary,
    y_cic_holdout_fewshot_pred,
    labels=[0, 1]
).ravel()

total = tn + fp + fn + tp

cic_fewshot_metrics_df = pd.DataFrame([{
    "scenario": "CIC_fewshot_calibrated_holdout",
    "accuracy": (tn + tp) / total,
    "precision": tp / (tp + fp) if (tp + fp) else 0,
    "recall": tp / (tp + fn) if (tp + fn) else 0,
    "f1": (
        2 * (tp / (tp + fp)) * (tp / (tp + fn)) /
        ((tp / (tp + fp)) + (tp / (tp + fn)))
        if (tp + fp) and (tp + fn) and ((tp / (tp + fp)) + (tp / (tp + fn))) else 0
    ),
    "fpr": fp / (fp + tn) if (fp + tn) else 0,
    "fnr": fn / (fn + tp) if (fn + tp) else 0,
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp)
}])

fewshot_prediction_distribution_df = pd.DataFrame({
    "true_CIC_holdout_labels": pd.Series(y_cic_holdout_binary)
        .map({0: "Benign", 1: "Attack"})
        .value_counts(),
    "fewshot_predictions": pd.Series(y_cic_holdout_fewshot_pred)
        .map({0: "Benign", 1: "Attack"})
        .value_counts()
}).fillna(0).astype(int)

display(cic_fewshot_metrics_df.round(4))
display(fewshot_prediction_distribution_df)


# ========================================================================================
# Notebook Cell C3B-10: Attack-level analysis for few-shot calibrated model
# ========================================================================================

# Cell C3B-10: Attack-level analysis for few-shot calibrated model
# Purpose:
# - Check which CIC attack types are detected after few-shot calibration
# - Compare few-shot recall with source-only and cost-sensitive recall

c3b_fewshot_detail_df = pd.DataFrame({
    "true_attack": cic_holdout_df["Attack"].values,
    "true_binary": y_cic_holdout_binary,
    "fewshot_pred": y_cic_holdout_fewshot_pred
})

fewshot_attack_only_df = c3b_fewshot_detail_df[
    c3b_fewshot_detail_df["true_attack"] != "Benign"
].copy()

c3b_fewshot_attack_df = (
    fewshot_attack_only_df
    .groupby("true_attack")
    .agg(
        holdout_samples=("true_attack", "size"),
        fewshot_recall=("fewshot_pred", "mean")
    )
    .reset_index()
)

# Add source-only and cost-sensitive recall from the full CIC sample
comparison_reference_df = c3b_cost_attack_comparison_df[
    ["true_attack", "source_only_recall", "cost_sensitive_recall"]
].copy()

c3b_fewshot_attack_comparison_df = c3b_fewshot_attack_df.merge(
    comparison_reference_df,
    on="true_attack",
    how="left"
)

c3b_fewshot_attack_comparison_df["fewshot_gain_over_source"] = (
    c3b_fewshot_attack_comparison_df["fewshot_recall"] -
    c3b_fewshot_attack_comparison_df["source_only_recall"]
)

c3b_fewshot_attack_comparison_df["fewshot_gain_over_cost_sensitive"] = (
    c3b_fewshot_attack_comparison_df["fewshot_recall"] -
    c3b_fewshot_attack_comparison_df["cost_sensitive_recall"]
)

c3b_fewshot_attack_comparison_df = c3b_fewshot_attack_comparison_df.sort_values(
    by="holdout_samples",
    ascending=False
)

display(c3b_fewshot_attack_comparison_df.round(4))

