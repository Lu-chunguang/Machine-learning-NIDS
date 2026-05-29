"""
Component 3C - Train Augmented Models

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3C-2

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3C-2: Train augmentation variants
# ========================================================================================

# Cell C3C-2: Train augmentation variants
# Purpose:
# - Train augmented XGBoost models using proposal-style corrupted samples
# - Mix each augmented subset with the original clean training set

import time
from xgboost import XGBClassifier


def train_multiclass_xgb_on_augmented_data(model_name, augmented_parts, augmented_labels):
    X_train_variant = sparse.vstack([
        X_train_processed,
        *augmented_parts
    ])
    
    y_train_variant = np.concatenate([
        y_train,
        *augmented_labels
    ])
    
    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )
    
    start_time = time.time()
    model.fit(X_train_variant, y_train_variant)
    training_time = time.time() - start_time
    
    summary = {
        "model": model_name,
        "clean_train_samples": X_train_processed.shape[0],
        "augmented_samples": X_train_variant.shape[0] - X_train_processed.shape[0],
        "total_train_samples": X_train_variant.shape[0],
        "processed_features": X_train_variant.shape[1],
        "training_time_seconds": training_time
    }
    
    del X_train_variant, y_train_variant
    gc.collect()
    
    return model, summary


def make_zeroing_augmented_part(row_indices, zero_ratio, seed):
    rng = np.random.default_rng(seed)
    X_part = X_train_processed[row_indices].copy()
    y_part = y_train[row_indices].copy()
    
    zero_count = int(X_train_processed.shape[1] * zero_ratio)
    zero_indices = rng.choice(
        X_train_processed.shape[1],
        size=zero_count,
        replace=False
    )
    
    X_part = X_part.tolil() if sparse.issparse(X_part) else X_part
    X_part[:, zero_indices] = 0
    
    if sparse.issparse(X_part):
        X_part = X_part.tocsr()
    
    return X_part, y_part


def make_gaussian_augmented_part(row_indices, sigma_min, sigma_max, seed, zero_ratio=0.0):
    rng = np.random.default_rng(seed)
    
    X_dense = (
        X_train_processed[row_indices]
        .toarray()
        .astype(np.float32, copy=False)
    )
    
    y_part = y_train[row_indices].copy()
    
    row_sigmas = rng.uniform(
        sigma_min,
        sigma_max,
        size=(len(row_indices), 1)
    ).astype(np.float32)
    
    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(len(row_indices), len(numeric_processed_indices))
    ).astype(np.float32)
    
    X_dense[:, numeric_processed_indices] += noise * row_sigmas
    
    if zero_ratio > 0:
        zero_count = int(X_train_processed.shape[1] * zero_ratio)
        zero_indices = rng.choice(
            X_train_processed.shape[1],
            size=zero_count,
            replace=False
        )
        X_dense[:, zero_indices] = 0
    
    X_part = sparse.csr_matrix(X_dense)
    
    del X_dense, noise
    gc.collect()
    
    return X_part, y_part


c3c_rng = np.random.default_rng(RANDOM_STATE)
c3c_training_summaries = []

# A. Random zeroing augmentation
zeroing_rows = c3c_rng.choice(
    X_train_processed.shape[0],
    size=200_000,
    replace=False
)
X_zeroing_part, y_zeroing_part = make_zeroing_augmented_part(
    zeroing_rows,
    zero_ratio=0.25,
    seed=RANDOM_STATE + 10
)
c3c_zeroing_xgb, zeroing_summary = train_multiclass_xgb_on_augmented_data(
    "random_zeroing",
    [X_zeroing_part],
    [y_zeroing_part]
)
c3c_training_summaries.append(zeroing_summary)
del X_zeroing_part, y_zeroing_part
gc.collect()

# B. Gaussian noise augmentation
gaussian_rows = c3c_rng.choice(
    X_train_processed.shape[0],
    size=200_000,
    replace=False
)
X_gaussian_part, y_gaussian_part = make_gaussian_augmented_part(
    gaussian_rows,
    sigma_min=0.1,
    sigma_max=1.0,
    seed=RANDOM_STATE + 20
)
c3c_gaussian_xgb, gaussian_summary = train_multiclass_xgb_on_augmented_data(
    "gaussian_noise",
    [X_gaussian_part],
    [y_gaussian_part]
)
c3c_training_summaries.append(gaussian_summary)
del X_gaussian_part, y_gaussian_part
gc.collect()

# C. Combined Gaussian noise + zeroing augmentation
combined_rows = c3c_rng.choice(
    X_train_processed.shape[0],
    size=200_000,
    replace=False
)
X_combined_part, y_combined_part = make_gaussian_augmented_part(
    combined_rows,
    sigma_min=0.1,
    sigma_max=1.0,
    seed=RANDOM_STATE + 30,
    zero_ratio=0.25
)
c3c_combined_xgb, combined_summary = train_multiclass_xgb_on_augmented_data(
    "combined_noise_zeroing",
    [X_combined_part],
    [y_combined_part]
)
c3c_training_summaries.append(combined_summary)
del X_combined_part, y_combined_part
gc.collect()

# D. Multi-ratio zeroing augmentation
multi_ratio_parts = []
multi_ratio_labels = []

for ratio_id, zero_ratio in enumerate([0.10, 0.25, 0.50]):
    ratio_rows = c3c_rng.choice(
        X_train_processed.shape[0],
        size=100_000,
        replace=False
    )
    X_ratio_part, y_ratio_part = make_zeroing_augmented_part(
        ratio_rows,
        zero_ratio=zero_ratio,
        seed=RANDOM_STATE + 40 + ratio_id
    )
    multi_ratio_parts.append(X_ratio_part)
    multi_ratio_labels.append(y_ratio_part)

c3c_multiratio_xgb, multiratio_summary = train_multiclass_xgb_on_augmented_data(
    "multi_ratio_zeroing",
    multi_ratio_parts,
    multi_ratio_labels
)
c3c_training_summaries.append(multiratio_summary)

del multi_ratio_parts, multi_ratio_labels
gc.collect()

c3c_training_summary_df = pd.DataFrame(c3c_training_summaries)
display(c3c_training_summary_df.round(4))

c3c_augmented_models = {
    "random_zeroing": c3c_zeroing_xgb,
    "gaussian_noise": c3c_gaussian_xgb,
    "combined_noise_zeroing": c3c_combined_xgb,
    "multi_ratio_zeroing": c3c_multiratio_xgb
}

c3c_primary_augmented_model_name = "random_zeroing"
c3c_primary_augmented_model = c3c_augmented_models[c3c_primary_augmented_model_name]

