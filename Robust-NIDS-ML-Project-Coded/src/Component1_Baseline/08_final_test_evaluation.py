"""
Component 1 - Final Test Evaluation

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C1-7A

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C1-7A: Evaluate selected baseline model on held-out test split
# ========================================================================================

# Cell C1-7A: Evaluate selected baseline model on held-out test split

from sklearn.metrics import accuracy_score, f1_score, classification_report
import time

# Use the best validation model selected in C1-6F
selected_baseline_name = best_baseline_name
selected_baseline_model = best_baseline_model

start_time = time.time()
y_test_pred = selected_baseline_model.predict(X_test_processed).astype(int)
test_inference_time = time.time() - start_time

test_accuracy = accuracy_score(y_test, y_test_pred)
test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")
test_weighted_f1 = f1_score(y_test, y_test_pred, average="weighted")

baseline_test_summary_df = pd.DataFrame({
    "accuracy": [test_accuracy],
    "macro_f1": [test_macro_f1],
    "weighted_f1": [test_weighted_f1],
    "test_inference_time_seconds": [test_inference_time]
}, index=[selected_baseline_name])

display(baseline_test_summary_df.round(4))

baseline_test_report = classification_report(
    y_test,
    y_test_pred,
    target_names=attack_class_names,
    output_dict=True,
    zero_division=0
)

baseline_test_class_report_df = (
    pd.DataFrame(baseline_test_report)
    .transpose()
    .loc[attack_class_names]
    .round(4)
)

baseline_test_class_report_df["support"] = baseline_test_class_report_df["support"].astype(int)

display(baseline_test_class_report_df.sort_values(by="f1-score", ascending=True))

