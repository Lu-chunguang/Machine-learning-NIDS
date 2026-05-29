"""
Component 3B - Cost-Sensitive Binary Learning

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3B-1, C3B-2, C3B-3, C3B-4, C3B-5

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3B-1: Build UNSW binary labels for cost-sensitive learning
# ========================================================================================

# Cell C3B-1: Build UNSW binary labels for cost-sensitive learning
# Purpose:
# - Use the original UNSW Label column as binary target
# - Benign = 0, Attack = 1
# - Keep the same train / validation / test split from Component 1

y_train_binary = raw_df.loc[X_train_raw.index, "Label"].astype(int).to_numpy()
y_val_binary = raw_df.loc[X_val_raw.index, "Label"].astype(int).to_numpy()
y_test_binary = raw_df.loc[X_test_raw.index, "Label"].astype(int).to_numpy()

binary_label_summary_df = pd.DataFrame({
    "split": ["Train", "Validation", "Test", "CICIDS2018 sample"],
    "benign_samples": [
        int((y_train_binary == 0).sum()),
        int((y_val_binary == 0).sum()),
        int((y_test_binary == 0).sum()),
        int((y_cic_binary == 0).sum())
    ],
    "attack_samples": [
        int((y_train_binary == 1).sum()),
        int((y_val_binary == 1).sum()),
        int((y_test_binary == 1).sum()),
        int((y_cic_binary == 1).sum())
    ]
})

binary_label_summary_df["attack_ratio"] = (
    binary_label_summary_df["attack_samples"] /
    (binary_label_summary_df["benign_samples"] + binary_label_summary_df["attack_samples"])
)

display(binary_label_summary_df.round(4))


# ========================================================================================
# Notebook Cell C3B-2: Train cost-sensitive binary XGBoost on UNSW
# ========================================================================================

# Cell C3B-2: Train cost-sensitive binary XGBoost on UNSW
# Purpose:
# - Train a binary XGBoost detector using UNSW Benign vs Attack labels
# - Use scale_pos_weight to penalize missed attacks more heavily
# - Keep preprocessing fixed from Component 1

from xgboost import XGBClassifier
import time

negative_count = int((y_train_binary == 0).sum())
positive_count = int((y_train_binary == 1).sum())

scale_pos_weight_c3b = negative_count / positive_count

c3b_cost_sensitive_xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight_c3b,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

start_time = time.time()

c3b_cost_sensitive_xgb.fit(
    X_train_processed,
    y_train_binary
)

c3b_train_time = time.time() - start_time

c3b_cost_weight_summary_df = pd.DataFrame([{
    "negative_benign_train_samples": negative_count,
    "positive_attack_train_samples": positive_count,
    "scale_pos_weight": scale_pos_weight_c3b,
    "training_time_seconds": c3b_train_time
}])

display(c3b_cost_weight_summary_df.round(4))


# ========================================================================================
# Notebook Cell C3B-3: Evaluate cost-sensitive model on UNSW validation
# ========================================================================================

# Cell C3B-3: Evaluate cost-sensitive model on UNSW validation
# Purpose:
# - Check whether cost-sensitive learning still works on the source dataset
# - Compare binary detection quality before testing on CICIDS2018

from sklearn.metrics import confusion_matrix

y_val_cost_pred = c3b_cost_sensitive_xgb.predict(X_val_processed).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_val_binary,
    y_val_cost_pred,
    labels=[0, 1]
).ravel()

total = tn + fp + fn + tp

unsw_cost_metrics_df = pd.DataFrame([{
    "scenario": "UNSW_validation_cost_sensitive_binary",
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

display(unsw_cost_metrics_df.round(4))


# ========================================================================================
# Notebook Cell C3B-4: Test cost-sensitive binary model on CICIDS2018
# ========================================================================================

# Cell C3B-4: Test cost-sensitive binary model on CICIDS2018
# Purpose:
# - Evaluate whether cost-sensitive learning improves cross-dataset attack detection
# - Compare against the Component 2B source-only baseline

y_cic_cost_pred = c3b_cost_sensitive_xgb.predict(X_cic_processed).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_cic_binary,
    y_cic_cost_pred,
    labels=[0, 1]
).ravel()

total = tn + fp + fn + tp

cic_cost_metrics_df = pd.DataFrame([{
    "scenario": "CIC_cost_sensitive_binary",
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

cic_cost_prediction_distribution_df = pd.DataFrame({
    "true_CIC_labels": pd.Series(y_cic_binary)
        .map({0: "Benign", 1: "Attack"})
        .value_counts(),
    "cost_sensitive_predictions": pd.Series(y_cic_cost_pred)
        .map({0: "Benign", 1: "Attack"})
        .value_counts()
}).fillna(0).astype(int)

display(cic_cost_metrics_df.round(4))
display(cic_cost_prediction_distribution_df)


# ========================================================================================
# Notebook Cell C3B-5: Attack-level analysis for cost-sensitive model on CICIDS2018
# ========================================================================================

# Cell C3B-5: Attack-level analysis for cost-sensitive model on CICIDS2018
# Purpose:
# - Check which CIC attack types improved under cost-sensitive learning
# - Compare attack recall against the Component 2B source-only model

c3b_cost_detail_df = pd.DataFrame({
    "true_attack": cic_raw["Attack"].values,
    "true_binary": y_cic_binary,
    "source_only_pred": y_cic_pred_binary,
    "cost_sensitive_pred": y_cic_cost_pred
})

attack_only_cost_df = c3b_cost_detail_df[
    c3b_cost_detail_df["true_attack"] != "Benign"
].copy()

c3b_cost_attack_comparison_df = (
    attack_only_cost_df
    .groupby("true_attack")
    .agg(
        samples=("true_attack", "size"),
        source_only_recall=("source_only_pred", "mean"),
        cost_sensitive_recall=("cost_sensitive_pred", "mean")
    )
    .reset_index()
)

c3b_cost_attack_comparison_df["recall_gain"] = (
    c3b_cost_attack_comparison_df["cost_sensitive_recall"] -
    c3b_cost_attack_comparison_df["source_only_recall"]
)

c3b_cost_attack_comparison_df = c3b_cost_attack_comparison_df.sort_values(
    by="samples",
    ascending=False
)

display(c3b_cost_attack_comparison_df.round(4))

