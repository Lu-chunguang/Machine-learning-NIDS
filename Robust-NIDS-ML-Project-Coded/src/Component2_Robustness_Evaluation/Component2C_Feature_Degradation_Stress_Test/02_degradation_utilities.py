"""
Component 2C - Degradation Utilities

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C2C-2

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C2C-2: Build degradation helper functions
# ========================================================================================

# Cell C2C-2: Build degradation helper functions
# Purpose:
# - Keep Gaussian noise, random masking, and feature dropout reproducible
# - Aggregate repeated runs as mean and standard deviation

def processed_name_to_raw_feature(processed_name):
    if processed_name.startswith("log_numeric__"):
        return processed_name.replace("log_numeric__", "", 1)
    if processed_name.startswith("regular_numeric__"):
        return processed_name.replace("regular_numeric__", "", 1)
    if processed_name.startswith("categorical__PROTOCOL_"):
        return "PROTOCOL"
    if processed_name.startswith("categorical__L7_PROTO_"):
        return "L7_PROTO"
    return processed_name


processed_feature_to_raw_feature = {
    idx: processed_name_to_raw_feature(name)
    for idx, name in enumerate(baseline_feature_names)
}

raw_feature_to_processed_indices = {}
for idx, raw_feature in processed_feature_to_raw_feature.items():
    raw_feature_to_processed_indices.setdefault(raw_feature, []).append(idx)

numeric_processed_indices = np.array([
    idx for idx, name in enumerate(baseline_feature_names)
    if name.startswith("log_numeric__") or name.startswith("regular_numeric__")
])

dropout_candidate_features = sorted(raw_feature_to_processed_indices.keys())


def evaluate_multiclass_prediction(model, X_eval):
    y_pred = model.predict(X_eval).astype(int)
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted")
    }


def make_random_masked_matrix(X_source, mask_ratio, seed):
    rng = np.random.default_rng(seed)
    total_features = X_source.shape[1]
    masked_feature_count = int(total_features * mask_ratio)
    
    X_degraded = X_source.copy().tolil() if sparse.issparse(X_source) else X_source.copy()
    
    if masked_feature_count > 0:
        masked_indices = rng.choice(
            total_features,
            size=masked_feature_count,
            replace=False
        )
        X_degraded[:, masked_indices] = 0
    
    if sparse.issparse(X_degraded):
        X_degraded = X_degraded.tocsr()
    
    return X_degraded, masked_feature_count


def make_feature_dropout_matrix(X_source, dropped_raw_features):
    drop_indices = []
    
    for raw_feature in dropped_raw_features:
        drop_indices.extend(raw_feature_to_processed_indices[raw_feature])
    
    X_degraded = X_source.copy().tolil() if sparse.issparse(X_source) else X_source.copy()
    X_degraded[:, drop_indices] = 0
    
    if sparse.issparse(X_degraded):
        X_degraded = X_degraded.tocsr()
    
    return X_degraded, len(drop_indices)


def make_gaussian_noise_matrix(X_source, sigma, seed):
    rng = np.random.default_rng(seed)
    
    X_degraded = (
        X_source.toarray().astype(np.float32, copy=False)
        if sparse.issparse(X_source)
        else X_source.copy().astype(np.float32, copy=False)
    )
    
    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=(X_degraded.shape[0], len(numeric_processed_indices))
    ).astype(np.float32)
    
    X_degraded[:, numeric_processed_indices] += noise
    
    del noise
    gc.collect()
    
    return X_degraded


def summarize_degradation_rows(rows):
    detailed_df = pd.DataFrame(rows)
    
    summary_df = (
        detailed_df
        .groupby(["model", "degradation_type", "level"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_std=("weighted_f1", "std")
        )
    )
    
    return detailed_df, summary_df


print("Degradation helper functions are ready.")
print("Processed features:", X_test_processed.shape[1])
print("Raw feature groups for dropout:", len(dropout_candidate_features))
print("Numeric processed features for Gaussian noise:", len(numeric_processed_indices))

