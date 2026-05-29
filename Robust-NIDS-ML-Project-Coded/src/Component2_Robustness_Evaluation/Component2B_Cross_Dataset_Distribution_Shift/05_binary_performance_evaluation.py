"""
Component 2B - Binary Performance Evaluation

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C2B-6

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C2B-6: Evaluate source-only binary performance on CICIDS2018
# ========================================================================================

# Cell C2B-6: Evaluate source-only binary performance on CICIDS2018
# Purpose:
# - Evaluate Benign vs Attack performance on CIC
# - Focus on recall, FPR, and FNR under cross-dataset shift

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

tn, fp, fn, tp = confusion_matrix(
    y_cic_binary,
    y_cic_pred_binary,
    labels=[0, 1]
).ravel()

cic_source_only_metrics = {
    "accuracy": accuracy_score(y_cic_binary, y_cic_pred_binary),
    "precision": precision_score(y_cic_binary, y_cic_pred_binary, zero_division=0),
    "recall": recall_score(y_cic_binary, y_cic_pred_binary, zero_division=0),
    "f1": f1_score(y_cic_binary, y_cic_pred_binary, zero_division=0),
    "fpr": fp / (fp + tn) if (fp + tn) else 0,
    "fnr": fn / (fn + tp) if (fn + tp) else 0,
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp)
}

cic_source_only_metrics_df = pd.DataFrame(
    [cic_source_only_metrics],
    index=["CIC_source_only_UNSW_XGBoost"]
)

display(cic_source_only_metrics_df.round(4))

