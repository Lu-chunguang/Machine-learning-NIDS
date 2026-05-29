"""
Component 1 - Baseline Model Training

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C1-6A, C1-6B, C1-6C, C1-6D, C1-6E

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C1-6A: Prepare shared baseline evaluation utilities
# ========================================================================================

# Cell C1-6A: Prepare shared baseline evaluation utilities

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import time

baseline_results = []
baseline_predictions = {}
baseline_class_reports = {}
baseline_confusion_matrices = {}
baseline_model_objects = {}


def record_baseline_result(model_name, model, y_true, y_pred, train_time, inference_time):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    
    baseline_results.append({
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "training_time_seconds": train_time,
        "inference_time_seconds": inference_time
    })
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=attack_class_names,
        output_dict=True,
        zero_division=0
    )
    
    class_report_df = (
        pd.DataFrame(report)
        .transpose()
        .loc[attack_class_names]
        .round(4)
    )
    class_report_df["support"] = class_report_df["support"].astype(int)
    
    normalized_cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(attack_class_names)),
        normalize="true"
    )
    
    baseline_predictions[model_name] = y_pred
    baseline_class_reports[model_name] = class_report_df
    baseline_confusion_matrices[model_name] = pd.DataFrame(
        normalized_cm,
        index=attack_class_names,
        columns=attack_class_names
    )
    baseline_model_objects[model_name] = model
    
    return pd.DataFrame([baseline_results[-1]]).set_index("model").round(4)


print("Baseline evaluation utilities are ready.")
print("Classes:", attack_class_names)


# ========================================================================================
# Notebook Cell C1-6B: Train and evaluate majority-class baseline
# ========================================================================================

# Cell C1-6B: Train and evaluate majority-class baseline

from sklearn.dummy import DummyClassifier

majority_baseline = DummyClassifier(
    strategy="most_frequent"
)

start_time = time.time()
majority_baseline.fit(X_train_processed, y_train)
majority_train_time = time.time() - start_time

start_time = time.time()
y_val_pred_majority = majority_baseline.predict(X_val_processed)
majority_inference_time = time.time() - start_time

display(
    record_baseline_result(
        "Majority-class",
        majority_baseline,
        y_val,
        y_val_pred_majority,
        majority_train_time,
        majority_inference_time
    )
)

print("Majority-class baseline finished.")


# ========================================================================================
# Notebook Cell C1-6C: Train and evaluate Logistic Regression baseline
# ========================================================================================

# Cell C1-6C: Train and evaluate Logistic Regression baseline

from sklearn.linear_model import LogisticRegression

logreg_baseline = LogisticRegression(
    solver="saga",
    penalty="l2",
    C=1.0,
    max_iter=100,
    tol=0.01,
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
logreg_baseline.fit(X_train_processed, y_train)
logreg_train_time = time.time() - start_time

start_time = time.time()
y_val_pred_logreg = logreg_baseline.predict(X_val_processed)
logreg_inference_time = time.time() - start_time

display(
    record_baseline_result(
        "Logistic Regression",
        logreg_baseline,
        y_val,
        y_val_pred_logreg,
        logreg_train_time,
        logreg_inference_time
    )
)

print("Logistic Regression baseline finished.")


# ========================================================================================
# Notebook Cell C1-6D: Train and evaluate Random Forest baseline
# ========================================================================================

# Cell C1-6D: Train and evaluate Random Forest baseline

from sklearn.ensemble import RandomForestClassifier

rf_baseline = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
rf_baseline.fit(X_train_processed, y_train)
rf_train_time = time.time() - start_time

start_time = time.time()
y_val_pred_rf = rf_baseline.predict(X_val_processed)
rf_inference_time = time.time() - start_time

display(
    record_baseline_result(
        "Random Forest",
        rf_baseline,
        y_val,
        y_val_pred_rf,
        rf_train_time,
        rf_inference_time
    )
)

print("Random Forest baseline finished.")


# ========================================================================================
# Notebook Cell C1-6E: Train and evaluate XGBoost chosen model
# ========================================================================================

# Cell C1-6E: Train and evaluate XGBoost chosen model

from xgboost import XGBClassifier

baseline_xgb = XGBClassifier(
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
baseline_xgb.fit(X_train_processed, y_train)
xgb_train_time = time.time() - start_time

start_time = time.time()
y_val_pred_xgb = baseline_xgb.predict(X_val_processed)
xgb_inference_time = time.time() - start_time

display(
    record_baseline_result(
        "XGBoost",
        baseline_xgb,
        y_val,
        y_val_pred_xgb.astype(int),
        xgb_train_time,
        xgb_inference_time
    )
)

print("XGBoost chosen model finished.")

