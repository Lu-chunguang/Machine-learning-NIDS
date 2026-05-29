"""
Component 2C - Baseline Under Degradation Conditions

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C2C-3

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C2C-3: Evaluate baseline under Gaussian noise, random masking, and feature dropout
# ========================================================================================

# Cell C2C-3: Evaluate baseline under Gaussian noise, random masking, and feature dropout
# Purpose:
# - Run all proposal-required Stress Test C conditions
# - Keep degradation independent from training

def run_degradation_suite(model, model_name, repeats=c2c_repeats):
    rows = []
    
    # Clean condition
    clean_metrics = evaluate_multiclass_prediction(model, X_test_processed)
    rows.append({
        "model": model_name,
        "degradation_type": "clean",
        "level": "clean",
        "repeat_id": 0,
        **clean_metrics
    })
    
    # Gaussian noise
    for sigma in c2c_noise_sigmas:
        for repeat_id in range(repeats):
            X_degraded = make_gaussian_noise_matrix(
                X_test_processed,
                sigma=sigma,
                seed=RANDOM_STATE + 1000 + repeat_id
            )
            metrics = evaluate_multiclass_prediction(model, X_degraded)
            rows.append({
                "model": model_name,
                "degradation_type": "gaussian_noise",
                "level": f"sigma={sigma}",
                "repeat_id": repeat_id,
                **metrics
            })
            del X_degraded
            gc.collect()
    
    # Random masking
    for mask_ratio in c2c_masking_ratios:
        for repeat_id in range(repeats):
            X_degraded, masked_feature_count = make_random_masked_matrix(
                X_test_processed,
                mask_ratio=mask_ratio,
                seed=RANDOM_STATE + 2000 + repeat_id
            )
            metrics = evaluate_multiclass_prediction(model, X_degraded)
            rows.append({
                "model": model_name,
                "degradation_type": "random_masking",
                "level": f"p={mask_ratio}",
                "repeat_id": repeat_id,
                "masked_processed_features": masked_feature_count,
                **metrics
            })
            del X_degraded
            gc.collect()
    
    # Feature dropout
    for feature_count in c2c_dropout_feature_counts:
        for repeat_id in range(repeats):
            rng = np.random.default_rng(RANDOM_STATE + 3000 + repeat_id)
            dropped_features = rng.choice(
                dropout_candidate_features,
                size=feature_count,
                replace=False
            ).tolist()
            
            X_degraded, dropped_processed_count = make_feature_dropout_matrix(
                X_test_processed,
                dropped_features
            )
            metrics = evaluate_multiclass_prediction(model, X_degraded)
            rows.append({
                "model": model_name,
                "degradation_type": "feature_dropout",
                "level": f"k={feature_count}",
                "repeat_id": repeat_id,
                "dropped_raw_features": ", ".join(dropped_features),
                "dropped_processed_features": dropped_processed_count,
                **metrics
            })
            del X_degraded
            gc.collect()
    
    return summarize_degradation_rows(rows)


c2c_degradation_detail_df, c2c_degradation_summary_df = run_degradation_suite(
    selected_baseline_model,
    "baseline"
)

display(c2c_degradation_summary_df.round(4))

