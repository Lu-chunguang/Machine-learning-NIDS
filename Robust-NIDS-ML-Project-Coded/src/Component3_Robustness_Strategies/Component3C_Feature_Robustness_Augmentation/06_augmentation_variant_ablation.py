"""
Component 3C - Augmentation Variant Ablation

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3C-6

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3C-6: Ablation across augmentation variants under random masking
# ========================================================================================

# Cell C3C-6: Ablation across augmentation variants under random masking
# Purpose:
# - Compare augmentation variants under the masking degradation condition
# - Keep this ablation narrower than the full Stress Test C rerun for runtime control

c3c_variant_masking_rows = []

for model_name, model in c3c_augmented_models.items():
    for mask_ratio in c2c_masking_ratios:
        for repeat_id in range(c2c_repeats):
            X_degraded, masked_feature_count = make_random_masked_matrix(
                X_test_processed,
                mask_ratio=mask_ratio,
                seed=RANDOM_STATE + 5000 + repeat_id
            )
            metrics = evaluate_multiclass_prediction(model, X_degraded)
            c3c_variant_masking_rows.append({
                "model": model_name,
                "mask_ratio": mask_ratio,
                "repeat_id": repeat_id,
                "masked_processed_features": masked_feature_count,
                **metrics
            })
            del X_degraded
            gc.collect()

c3c_variant_masking_detail_df = pd.DataFrame(c3c_variant_masking_rows)

c3c_variant_masking_summary_df = (
    c3c_variant_masking_detail_df
    .groupby(["model", "mask_ratio"], as_index=False)
    .agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std")
    )
)

c3c_variant_ablation_table_df = (
    c3c_variant_masking_summary_df
    .pivot(
        index="mask_ratio",
        columns="model",
        values="macro_f1_mean"
    )
    .reset_index()
)

baseline_masking_macro_df = (
    c2c_degradation_summary_df[
        c2c_degradation_summary_df["degradation_type"] == "random_masking"
    ][["level", "macro_f1_mean"]]
    .copy()
)
baseline_masking_macro_df["mask_ratio"] = (
    baseline_masking_macro_df["level"]
    .str.replace("p=", "", regex=False)
    .astype(float)
)
baseline_masking_macro_df = baseline_masking_macro_df[[
    "mask_ratio",
    "macro_f1_mean"
]].rename(columns={"macro_f1_mean": "baseline"})

c3c_variant_ablation_table_df = baseline_masking_macro_df.merge(
    c3c_variant_ablation_table_df,
    on="mask_ratio",
    how="left"
)

variant_columns = [
    col for col in c3c_variant_ablation_table_df.columns
    if col != "mask_ratio"
]

c3c_variant_ablation_table_df["best_strategy"] = (
    c3c_variant_ablation_table_df[variant_columns]
    .idxmax(axis=1)
)

c3c_variant_ablation_table_df["best_macro_f1"] = (
    c3c_variant_ablation_table_df[variant_columns]
    .max(axis=1)
)

display(c3c_variant_masking_summary_df.round(4))
display(c3c_variant_ablation_table_df.round(4))

