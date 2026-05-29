"""
Component 2A - Scenario A2 Common Attack Classes as Unknown

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C2A-10, C2A-11, C2A-12, C2A-13, C2A-14, C2A-15, C2A-16, C2A-17, C2A-18

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C2A-10: Prepare known-class training data for Scenario A2
# ========================================================================================

# Cell C2A-10: Prepare known-class training data for Scenario A2

c2a_active_scenario = "A2_common_attack_classes"
c2a_held_out_classes = c2a_scenarios[c2a_active_scenario]

print("Active scenario:", c2a_active_scenario)
print("Held-out unknown classes:", c2a_held_out_classes)

# Known classes are all classes except the held-out classes
c2a_known_classes = [
    cls for cls in attack_class_names
    if cls not in c2a_held_out_classes
]

print("Known classes:")
print(c2a_known_classes)

# Training and validation masks: keep only known classes
c2a_train_known_mask = ~y_train_text.isin(c2a_held_out_classes)
c2a_val_known_mask = ~y_val_text.isin(c2a_held_out_classes)

# Test masks: separate known and unknown test samples
c2a_test_known_mask = ~y_test_text.isin(c2a_held_out_classes)
c2a_test_unknown_mask = y_test_text.isin(c2a_held_out_classes)

X_train_c2a_raw = X_train_raw.loc[c2a_train_known_mask].copy()
y_train_c2a_text = y_train_text.loc[c2a_train_known_mask].copy()

X_val_c2a_raw = X_val_raw.loc[c2a_val_known_mask].copy()
y_val_c2a_text = y_val_text.loc[c2a_val_known_mask].copy()

X_test_c2a_raw = X_test_raw.copy()
y_test_c2a_text = y_test_text.copy()

c2a_split_summary_df = pd.DataFrame({
    "split": ["train_known", "validation_known", "test_known", "test_unknown", "test_all"],
    "samples": [
        len(X_train_c2a_raw),
        len(X_val_c2a_raw),
        int(c2a_test_known_mask.sum()),
        int(c2a_test_unknown_mask.sum()),
        len(X_test_c2a_raw)
    ]
})

display(c2a_split_summary_df)

print("C2A Scenario A2 data split is ready.")


# ========================================================================================
# Notebook Cell C2A-11: Encode labels and preprocess features for Scenario A2
# ========================================================================================

# Cell C2A-11: Encode labels and preprocess features for Scenario A2

from sklearn.preprocessing import LabelEncoder

# Label encoder sees only known classes
c2a_label_encoder = LabelEncoder()
y_train_c2a = c2a_label_encoder.fit_transform(y_train_c2a_text)
y_val_c2a = c2a_label_encoder.transform(y_val_c2a_text)

c2a_known_class_names = c2a_label_encoder.classes_.tolist()

print("C2A Scenario A2 known-class label mapping:")
display(pd.DataFrame({
    "encoded_label": range(len(c2a_known_class_names)),
    "known_attack_class": c2a_known_class_names
}))

# Build a fresh preprocessor with the same preprocessing design
c2a_preprocessor = ColumnTransformer(
    transformers=[
        ("log_numeric", log_numeric_pipeline, log_numeric_feature_cols),
        ("regular_numeric", regular_numeric_pipeline, regular_numeric_feature_cols),
        ("categorical", categorical_pipeline, categorical_feature_cols)
    ],
    remainder="drop",
    sparse_threshold=1.0
)

# Fit only on known-class training data
X_train_c2a_processed = c2a_preprocessor.fit_transform(X_train_c2a_raw)

# Apply the learned preprocessing to known validation and full test data
X_val_c2a_processed = c2a_preprocessor.transform(X_val_c2a_raw)
X_test_c2a_processed = c2a_preprocessor.transform(X_test_c2a_raw)

c2a_feature_names = c2a_preprocessor.get_feature_names_out()

print("C2A Scenario A2 processed train shape:", X_train_c2a_processed.shape)
print("C2A Scenario A2 processed validation shape:", X_val_c2a_processed.shape)
print("C2A Scenario A2 processed test shape:", X_test_c2a_processed.shape)
print("Number of C2A Scenario A2 processed features:", len(c2a_feature_names))


# ========================================================================================
# Notebook Cell C2A-12: Train Scenario A2 known-class XGBoost model
# ========================================================================================

# Cell C2A-12: Train Scenario A2 known-class XGBoost model

from xgboost import XGBClassifier
import time

c2a_xgb = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

start_time = time.time()

c2a_xgb.fit(
    X_train_c2a_processed,
    y_train_c2a
)

c2a_train_time = time.time() - start_time

print("C2A Scenario A2 known-class XGBoost trained.")
print(f"Training time: {c2a_train_time:.2f} seconds")
print("Number of known classes:", len(c2a_known_class_names))


# ========================================================================================
# Notebook Cell C2A-13: Generate Scenario A2 validation and test predictions
# ========================================================================================

# Cell C2A-13: Generate Scenario A2 validation and test predictions

# Predict on known-class validation data
y_val_c2a_pred = c2a_xgb.predict(X_val_c2a_processed).astype(int)
y_val_c2a_pred_names = c2a_label_encoder.inverse_transform(y_val_c2a_pred)

# Predict on full test data, including held-out unknown classes
y_test_c2a_pred = c2a_xgb.predict(X_test_c2a_processed).astype(int)
y_test_c2a_pred_names = c2a_label_encoder.inverse_transform(y_test_c2a_pred)

# Predicted probabilities for confidence analysis
y_test_c2a_proba = c2a_xgb.predict_proba(X_test_c2a_processed)
y_test_c2a_confidence = y_test_c2a_proba.max(axis=1)

print("C2A Scenario A2 predictions generated.")
print("Validation known samples:", len(y_val_c2a_pred))
print("Full test samples:", len(y_test_c2a_pred))
print("Mean test confidence:", round(y_test_c2a_confidence.mean(), 4))


# ========================================================================================
# Notebook Cell C2A-14: Evaluate Scenario A2 performance on known validation classes
# ========================================================================================

# Cell C2A-14: Evaluate Scenario A2 performance on known validation classes

from sklearn.metrics import accuracy_score, f1_score, classification_report

c2a_val_accuracy = accuracy_score(y_val_c2a, y_val_c2a_pred)
c2a_val_macro_f1 = f1_score(y_val_c2a, y_val_c2a_pred, average="macro")
c2a_val_weighted_f1 = f1_score(y_val_c2a, y_val_c2a_pred, average="weighted")

c2a_known_validation_summary_df = pd.DataFrame({
    "accuracy": [c2a_val_accuracy],
    "macro_f1": [c2a_val_macro_f1],
    "weighted_f1": [c2a_val_weighted_f1]
}, index=[c2a_active_scenario])

display(c2a_known_validation_summary_df.round(4))

print("Known-class validation report:")
print(
    classification_report(
        y_val_c2a,
        y_val_c2a_pred,
        target_names=c2a_known_class_names,
        digits=4,
        zero_division=0
    )
)


# ========================================================================================
# Notebook Cell C2A-15: Analyze Scenario A2 unknown-class predictions
# ========================================================================================

# Cell C2A-15: Analyze Scenario A2 unknown-class predictions

c2a_test_detail_df = pd.DataFrame({
    "true_attack": y_test_c2a_text.values,
    "is_unknown": c2a_test_unknown_mask.values,
    "predicted_known_class": y_test_c2a_pred_names,
    "confidence": y_test_c2a_confidence
})

c2a_a2_detail_df = c2a_test_detail_df.copy()

unknown_detail_df = c2a_test_detail_df[c2a_test_detail_df["is_unknown"]].copy()

print("Unknown test samples:", len(unknown_detail_df))

unknown_mapping_df = pd.crosstab(
    unknown_detail_df["true_attack"],
    unknown_detail_df["predicted_known_class"],
    normalize="index"
).mul(100).round(2)

display(unknown_mapping_df)

unknown_confidence_summary_df = (
    unknown_detail_df
    .groupby("true_attack")["confidence"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(4)
)

display(unknown_confidence_summary_df)


# ========================================================================================
# Notebook Cell C2A-16: Compare Scenario A2 confidence between known and unknown test samples
# ========================================================================================

# Cell C2A-16: Compare Scenario A2 confidence between known and unknown test samples

confidence_group_summary_df = (
    c2a_test_detail_df
    .assign(group=np.where(c2a_test_detail_df["is_unknown"], "Unknown held-out", "Known classes"))
    .groupby("group")["confidence"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(4)
)

display(confidence_group_summary_df)

plt.figure(figsize=(10, 5))

sns.histplot(
    data=c2a_test_detail_df,
    x="confidence",
    hue=np.where(c2a_test_detail_df["is_unknown"], "Unknown held-out", "Known classes"),
    bins=50,
    stat="density",
    common_norm=False,
    alpha=0.5
)

plt.title("Scenario A2 Confidence Distribution: Known vs Held-Out Unknown Classes")
plt.xlabel("Prediction confidence")
plt.ylabel("Density")
plt.show()


# ========================================================================================
# Notebook Cell C2A-17: Summarize Scenario A2 open-set stress-test findings
# ========================================================================================

# Cell C2A-17: Summarize Scenario A2 open-set stress-test findings

unknown_main_mapping_rows = []

for true_attack, group_df in unknown_detail_df.groupby("true_attack"):
    top_pred = group_df["predicted_known_class"].value_counts(normalize=True).idxmax()
    top_percent = group_df["predicted_known_class"].value_counts(normalize=True).max() * 100
    
    unknown_main_mapping_rows.append({
        "held_out_class": true_attack,
        "unknown_samples": len(group_df),
        "most_common_predicted_known_class": top_pred,
        "mapped_percent": top_percent,
        "mean_confidence": group_df["confidence"].mean(),
        "median_confidence": group_df["confidence"].median()
    })

c2a_a2_unknown_summary_df = (
    pd.DataFrame(unknown_main_mapping_rows)
    .sort_values(by="held_out_class")
    .round(4)
)

c2a_a2_overall_summary_df = pd.DataFrame({
    "scenario": [c2a_active_scenario],
    "held_out_classes": [", ".join(c2a_held_out_classes)],
    "known_validation_macro_f1": [c2a_val_macro_f1],
    "known_validation_weighted_f1": [c2a_val_weighted_f1],
    "unknown_samples": [len(unknown_detail_df)],
    "unknown_mean_confidence": [unknown_detail_df["confidence"].mean()],
    "unknown_median_confidence": [unknown_detail_df["confidence"].median()]
}).round(4)

display(c2a_a2_overall_summary_df)
display(c2a_a2_unknown_summary_df)


# ========================================================================================
# Notebook Cell C2A-18: Combine Scenario A1 and A2 summaries
# ========================================================================================

# Cell C2A-18: Combine Scenario A1 and A2 summaries

c2a_overall_summary_df = pd.concat(
    [c2a_a1_overall_summary_df, c2a_a2_overall_summary_df],
    ignore_index=True
)

c2a_unknown_summary_df = pd.concat(
    [c2a_a1_unknown_summary_df, c2a_a2_unknown_summary_df],
    ignore_index=True
)

display(c2a_overall_summary_df)
display(c2a_unknown_summary_df)

