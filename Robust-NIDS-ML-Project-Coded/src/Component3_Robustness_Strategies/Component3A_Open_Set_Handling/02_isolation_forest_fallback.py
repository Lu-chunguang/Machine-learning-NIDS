"""
Component 3A - One-Class Anomaly Fallback with Isolation Forest

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3A-7, C3A-8, C3A-9, C3A-10, C3A-10B

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3A-7: Prepare Benign-only training data for Isolation Forest
# ========================================================================================

# Cell C3A-7: Prepare Benign-only training data for Isolation Forest
# Purpose:
# - Train the one-class model only on normal traffic
# - Sample Benign rows to keep training efficient

from scipy import sparse

benign_train_mask = (y_train_text == "Benign").to_numpy()
benign_train_indices = np.where(benign_train_mask)[0]

isoforest_train_size = min(100000, len(benign_train_indices))

rng = np.random.default_rng(RANDOM_STATE)
sampled_benign_indices = rng.choice(
    benign_train_indices,
    size=isoforest_train_size,
    replace=False
)

X_benign_iforest_train = X_train_processed[sampled_benign_indices]

print("Total Benign training samples:", len(benign_train_indices))
print("Sampled Benign samples for Isolation Forest:", X_benign_iforest_train.shape[0])
print("Feature shape:", X_benign_iforest_train.shape)
print("Sparse input:", sparse.issparse(X_benign_iforest_train))


# ========================================================================================
# Notebook Cell C3A-8: Train Isolation Forest on Benign traffic only
# ========================================================================================

# Cell C3A-8: Train Isolation Forest on Benign traffic only
# Purpose:
# - Learn the normal traffic pattern from Benign samples
# - Samples that do not look like Benign can later be flagged as anomalies

from sklearn.ensemble import IsolationForest
import time

X_benign_iforest_train_dense = X_benign_iforest_train.toarray()

iforest = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    max_samples=10000,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

start_time = time.time()
iforest.fit(X_benign_iforest_train_dense)
iforest_train_time = time.time() - start_time

print("Isolation Forest trained on Benign traffic.")
print(f"Training time: {iforest_train_time:.2f} seconds")
print("Training shape:", X_benign_iforest_train_dense.shape)


# ========================================================================================
# Notebook Cell C3A-9: Sanity check Isolation Forest on C1 test split
# ========================================================================================

# Cell C3A-9: Sanity check Isolation Forest on C1 test split
# Purpose:
# - Estimate false positive rate on Benign test traffic
# - Check whether attack traffic is more likely to be flagged as anomaly

X_test_iforest_dense = X_test_processed.toarray()

# IsolationForest prediction:
#  1  = normal / inlier
# -1  = anomaly / outlier
iforest_test_pred = iforest.predict(X_test_iforest_dense)
iforest_test_is_anomaly = (iforest_test_pred == -1)
iforest_test_is_anomaly_series = pd.Series(iforest_test_is_anomaly, index=c3a_a2_detail_df.index)

test_is_benign = (y_test_text == "Benign").to_numpy()
test_is_attack = ~test_is_benign

iforest_sanity_df = pd.DataFrame({
    "group": ["Benign", "Attack"],
    "samples": [
        int(test_is_benign.sum()),
        int(test_is_attack.sum())
    ],
    "anomaly_rate": [
        iforest_test_is_anomaly[test_is_benign].mean(),
        iforest_test_is_anomaly[test_is_attack].mean()
    ],
    "normal_rate": [
        (~iforest_test_is_anomaly[test_is_benign]).mean(),
        (~iforest_test_is_anomaly[test_is_attack]).mean()
    ]
}).round(4)

display(iforest_sanity_df)


# ========================================================================================
# Notebook Cell C3A-10: Test Isolation Forest fallback on A2 unknown samples predicted as Benign
# ========================================================================================

# Cell C3A-10: Test Isolation Forest fallback on A2 unknown samples predicted as Benign
# Purpose:
# - Focus on the dangerous case: unknown attack -> predicted as Benign
# - Check whether Isolation Forest can catch these samples as anomalies

a2_unknown_pred_benign_mask = (
    c3a_a2_detail_df["is_unknown"] &
    (c3a_a2_detail_df["predicted_known_class"] == "Benign")
)

a2_unknown_benign_detail_df = c3a_a2_detail_df.loc[a2_unknown_pred_benign_mask].copy()
a2_unknown_benign_detail_df["iforest_anomaly"] = (
    iforest_test_is_anomaly_series.loc[a2_unknown_benign_detail_df.index].values
)

a2_iforest_fallback_summary_df = (
    a2_unknown_benign_detail_df
    .groupby("true_attack")["iforest_anomaly"]
    .agg(["count", "mean", "sum"])
    .rename(columns={
        "count": "unknown_predicted_benign_samples",
        "mean": "iforest_anomaly_rate",
        "sum": "iforest_caught_samples"
    })
    .round(4)
)

display(a2_iforest_fallback_summary_df)

print("Total A2 unknown samples predicted as Benign:", len(a2_unknown_benign_detail_df))
print("Total caught by Isolation Forest:", int(a2_unknown_benign_detail_df["iforest_anomaly"].sum()))
print("Overall caught rate:", round(a2_unknown_benign_detail_df["iforest_anomaly"].mean(), 4))


# ========================================================================================
# Notebook Cell C3A-10B: Precision-recall curve for Isolation Forest unknown detection
# ========================================================================================

# Cell C3A-10B: Precision-recall curve for Isolation Forest unknown detection
# Purpose:
# - Satisfy the Strategy 3 proposal requirement
# - Evaluate whether the one-class anomaly score separates A2 unknown attacks from known samples

from sklearn.metrics import precision_recall_curve, average_precision_score

# Higher score should mean more anomalous / more likely unknown.
iforest_unknown_score = -iforest.decision_function(X_test_iforest_dense)
iforest_unknown_score_series = pd.Series(
    iforest_unknown_score,
    index=c3a_a2_detail_df.index
)

y_a2_unknown = c3a_a2_detail_df["is_unknown"].astype(int).to_numpy()
a2_unknown_scores = iforest_unknown_score_series.loc[c3a_a2_detail_df.index].to_numpy()

iforest_precision, iforest_recall, iforest_pr_thresholds = precision_recall_curve(
    y_a2_unknown,
    a2_unknown_scores
)

c3a_iforest_average_precision = average_precision_score(
    y_a2_unknown,
    a2_unknown_scores
)

c3a_iforest_pr_summary_df = pd.DataFrame([{
    "scenario": "A2_common_attack_classes",
    "unknown_detector_score": "Isolation Forest anomaly score",
    "known_samples": int((y_a2_unknown == 0).sum()),
    "unknown_samples": int((y_a2_unknown == 1).sum()),
    "average_precision": c3a_iforest_average_precision
}])

display(c3a_iforest_pr_summary_df.round(4))

plt.figure(figsize=(7, 5))
plt.plot(iforest_recall, iforest_precision)
plt.title("Isolation Forest Precision-Recall Curve for A2 Unknown Detection")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()

