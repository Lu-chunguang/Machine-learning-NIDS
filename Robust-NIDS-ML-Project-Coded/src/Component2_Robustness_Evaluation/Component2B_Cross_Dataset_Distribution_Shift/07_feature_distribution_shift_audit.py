"""
Component 2B - Feature Distribution Shift Audit

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C2B-8, C2B-9, C2B-10

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C2B-8: Audit feature distribution shift between UNSW train and CIC sample
# ========================================================================================

# Cell C2B-8: Audit feature distribution shift between UNSW train and CIC sample
# Purpose:
# - Compare raw feature distributions across source and target domains
# - Include proposal-required mean and variance
# - Keep robust statistics such as median, IQR, and zero percentage for interpretation

shift_rows = []

for feature in X_raw.columns:
    if feature not in X_cic_raw.columns:
        continue
    
    if feature in categorical_feature_cols:
        continue
    
    unsw_values = pd.to_numeric(X_train_raw[feature], errors="coerce")
    cic_values = pd.to_numeric(X_cic_raw[feature], errors="coerce")
    
    unsw_q1 = unsw_values.quantile(0.25)
    unsw_median = unsw_values.quantile(0.50)
    unsw_q3 = unsw_values.quantile(0.75)
    unsw_iqr = unsw_q3 - unsw_q1
    
    cic_q1 = cic_values.quantile(0.25)
    cic_median = cic_values.quantile(0.50)
    cic_q3 = cic_values.quantile(0.75)
    cic_iqr = cic_q3 - cic_q1
    
    unsw_zero_pct = (unsw_values == 0).mean() * 100
    cic_zero_pct = (cic_values == 0).mean() * 100
    
    shift_rows.append({
        "feature": feature,
        "UNSW_mean": unsw_values.mean(),
        "CIC_mean": cic_values.mean(),
        "UNSW_variance": unsw_values.var(),
        "CIC_variance": cic_values.var(),
        "UNSW_median": unsw_median,
        "CIC_median": cic_median,
        "median_gap": abs(cic_median - unsw_median),
        "UNSW_IQR": unsw_iqr,
        "CIC_IQR": cic_iqr,
        "IQR_ratio_CIC_over_UNSW": (cic_iqr + 1) / (unsw_iqr + 1),
        "UNSW_zero_%": unsw_zero_pct,
        "CIC_zero_%": cic_zero_pct,
        "zero_gap_%": abs(cic_zero_pct - unsw_zero_pct)
    })

feature_shift_df = pd.DataFrame(shift_rows)

feature_shift_df["shift_score"] = (
    feature_shift_df["median_gap"].rank(pct=True) +
    feature_shift_df["zero_gap_%"].rank(pct=True) +
    abs(np.log(feature_shift_df["IQR_ratio_CIC_over_UNSW"])).rank(pct=True)
)

feature_shift_df = feature_shift_df.sort_values(
    by="shift_score",
    ascending=False
).round(4)

display(feature_shift_df.head(20))


# ========================================================================================
# Notebook Cell C2B-9: Audit categorical feature shift between UNSW and CICIDS2018
# ========================================================================================

# Cell C2B-9: Audit categorical feature shift between UNSW and CICIDS2018
# Purpose:
# - Compare PROTOCOL and L7_PROTO distributions across UNSW and CIC
# - These categorical features are one-hot encoded in the baseline preprocessor

for col in categorical_feature_cols:
    print(f"\nTop values for {col} in UNSW train:")
    unsw_cat_dist = (
        X_train_raw[col]
        .value_counts(normalize=True)
        .head(10)
        .mul(100)
        .round(2)
        .to_frame("UNSW_percent")
    )
    display(unsw_cat_dist)

    print(f"\nTop values for {col} in CICIDS2018 sample:")
    cic_cat_dist = (
        X_cic_raw[col]
        .value_counts(normalize=True)
        .head(10)
        .mul(100)
        .round(2)
        .to_frame("CIC_percent")
    )
    display(cic_cat_dist)


# ========================================================================================
# Notebook Cell C2B-10: Component 2B checkpoint tables
# ========================================================================================

# Cell C2B-10: Component 2B checkpoint tables
# Purpose:
# - Save source-only cross-dataset metrics
# - Report proposal-required relative F1 degradation
# - Save attack-level failure table and top feature-shift evidence

from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(
    y_cic_binary,
    y_cic_pred_binary,
    labels=[0, 1]
).ravel()

total = tn + fp + fn + tp

c2b_accuracy = (tn + tp) / total
c2b_precision = tp / (tp + fp) if (tp + fp) else 0
c2b_recall = tp / (tp + fn) if (tp + fn) else 0
c2b_f1 = (
    2 * c2b_precision * c2b_recall / (c2b_precision + c2b_recall)
    if (c2b_precision + c2b_recall)
    else 0
)
c2b_fpr = fp / (fp + tn) if (fp + tn) else 0
c2b_fnr = fn / (fn + tp) if (fn + tp) else 0

c2b_final_metrics_df = pd.DataFrame([{
    "scenario": "C2B_source_only_cross_dataset",
    "train_dataset": "NF-UNSW-NB15-v2",
    "test_dataset": "NF-CSE-CIC-IDS2018-v2 sample",
    "model": "UNSW-trained XGBoost",
    "accuracy": c2b_accuracy,
    "precision": c2b_precision,
    "recall": c2b_recall,
    "f1": c2b_f1,
    "fpr": c2b_fpr,
    "fnr": c2b_fnr,
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp)
}])

# In-domain binary performance of the selected UNSW model
source_test_true_binary = (pd.Series(y_test_text).to_numpy() != "Benign").astype(int)
source_test_pred_names = attack_label_encoder.inverse_transform(y_test_pred.astype(int))
source_test_pred_binary = (source_test_pred_names != "Benign").astype(int)

source_binary_f1 = f1_score(
    source_test_true_binary,
    source_test_pred_binary,
    zero_division=0
)

c2b_relative_degradation_df = pd.DataFrame([{
    "source_domain": "UNSW test",
    "target_domain": "CICIDS2018 sample",
    "source_binary_f1": source_binary_f1,
    "target_binary_f1": c2b_f1,
    "delta_f1_target_minus_source": c2b_f1 - source_binary_f1,
    "relative_f1_drop": (
        (source_binary_f1 - c2b_f1) / source_binary_f1
        if source_binary_f1 else np.nan
    )
}])

c2b_attack_failure_df = (
    cic_prediction_detail_df[cic_prediction_detail_df["true_attack"] != "Benign"]
    .groupby("true_attack")
    .agg(
        samples=("true_attack", "size"),
        detected_as_attack_rate=("predicted_binary", "mean")
    )
    .reset_index()
)

c2b_attack_failure_df["missed_as_benign_rate"] = (
    1 - c2b_attack_failure_df["detected_as_attack_rate"]
)

c2b_attack_failure_df = c2b_attack_failure_df.sort_values(
    by="samples",
    ascending=False
)

c2b_top_shift_features_df = feature_shift_df.head(10).copy()

display(c2b_final_metrics_df.round(4))
display(c2b_relative_degradation_df.round(4))
display(c2b_attack_failure_df.round(4))
display(c2b_top_shift_features_df.round(4))

