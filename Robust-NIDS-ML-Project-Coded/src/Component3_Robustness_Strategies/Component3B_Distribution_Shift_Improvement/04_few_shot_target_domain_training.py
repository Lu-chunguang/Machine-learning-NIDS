"""
Component 3B - Few-Shot Target-Domain Training

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3B-7, C3B-8

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3B-7: Preprocess CIC few-shot and holdout sets
# ========================================================================================

# Cell C3B-7: Preprocess CIC few-shot and holdout sets
# Purpose:
# - Build CIC few-shot and holdout feature matrices
# - Use the UNSW-fitted baseline preprocessor
# - Create binary labels for calibration and holdout evaluation

X_cic_fewshot_raw = cic_fewshot_df.drop(columns=drop_from_features).copy()
X_cic_holdout_raw = cic_holdout_df.drop(columns=drop_from_features).copy()

X_cic_fewshot_raw = X_cic_fewshot_raw[X_raw.columns]
X_cic_holdout_raw = X_cic_holdout_raw[X_raw.columns]

X_cic_fewshot_processed = baseline_preprocessor.transform(X_cic_fewshot_raw)
X_cic_holdout_processed = baseline_preprocessor.transform(X_cic_holdout_raw)

y_cic_fewshot_binary = (cic_fewshot_df["Attack"] != "Benign").astype(int).to_numpy()
y_cic_holdout_binary = (cic_holdout_df["Attack"] != "Benign").astype(int).to_numpy()

fewshot_preprocess_summary_df = pd.DataFrame([{
    "dataset": "CIC_fewshot",
    "samples": X_cic_fewshot_processed.shape[0],
    "features": X_cic_fewshot_processed.shape[1],
    "attack_samples": int((y_cic_fewshot_binary == 1).sum()),
    "benign_samples": int((y_cic_fewshot_binary == 0).sum()),
    "attack_ratio": y_cic_fewshot_binary.mean()
}, {
    "dataset": "CIC_holdout",
    "samples": X_cic_holdout_processed.shape[0],
    "features": X_cic_holdout_processed.shape[1],
    "attack_samples": int((y_cic_holdout_binary == 1).sum()),
    "benign_samples": int((y_cic_holdout_binary == 0).sum()),
    "attack_ratio": y_cic_holdout_binary.mean()
}])

display(fewshot_preprocess_summary_df.round(4))


# ========================================================================================
# Notebook Cell C3B-8: Train few-shot target-calibrated binary XGBoost
# ========================================================================================

# Cell C3B-8: Train few-shot target-calibrated binary XGBoost
# Purpose:
# - Combine UNSW training data with the CIC few-shot calibration set
# - Train a binary XGBoost model on source data plus limited target-domain labels
# - Keep CIC holdout fully separate for final evaluation

from scipy.sparse import vstack
from xgboost import XGBClassifier
import time

X_fewshot_train_combined = vstack([
    X_train_processed,
    X_cic_fewshot_processed
])

y_fewshot_train_combined = np.concatenate([
    y_train_binary,
    y_cic_fewshot_binary
])

fewshot_negative_count = int((y_fewshot_train_combined == 0).sum())
fewshot_positive_count = int((y_fewshot_train_combined == 1).sum())

fewshot_scale_pos_weight = fewshot_negative_count / fewshot_positive_count

c3b_fewshot_xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=fewshot_scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

start_time = time.time()

c3b_fewshot_xgb.fit(
    X_fewshot_train_combined,
    y_fewshot_train_combined
)

fewshot_train_time = time.time() - start_time

fewshot_training_summary_df = pd.DataFrame([{
    "training_source": "UNSW_train + CIC_fewshot",
    "train_samples": X_fewshot_train_combined.shape[0],
    "features": X_fewshot_train_combined.shape[1],
    "benign_samples": fewshot_negative_count,
    "attack_samples": fewshot_positive_count,
    "scale_pos_weight": fewshot_scale_pos_weight,
    "training_time_seconds": fewshot_train_time
}])

display(fewshot_training_summary_df.round(4))

