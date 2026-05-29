"""
Component 2B - Source-Only Prediction

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C2B-5

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C2B-5: Source-only UNSW model prediction on CICIDS2018
# ========================================================================================

# Cell C2B-5: Source-only UNSW model prediction on CICIDS2018
# Purpose:
# - Use the selected Component 1 XGBoost model directly on CIC
# - Convert UNSW multi-class predictions into binary Benign vs Attack
# - No retraining or adaptation is applied

# Multi-class prediction using the UNSW-trained selected model
y_cic_pred_encoded = selected_baseline_model.predict(X_cic_processed).astype(int)

# Convert encoded UNSW labels back to class names
y_cic_pred_names = attack_label_encoder.inverse_transform(y_cic_pred_encoded)

# Binary conversion:
# predicted Benign = 0
# predicted any attack class = 1
y_cic_pred_binary = (y_cic_pred_names != "Benign").astype(int)

cic_prediction_detail_df = pd.DataFrame({
    "true_attack": cic_raw["Attack"].values,
    "true_binary": y_cic_binary,
    "predicted_unsw_class": y_cic_pred_names,
    "predicted_binary": y_cic_pred_binary
})

print("CIC source-only predictions generated.")

print("\nTrue binary distribution:")
display(
    pd.Series(y_cic_binary)
    .map({0: "Benign", 1: "Attack"})
    .value_counts()
    .to_frame("samples")
)

print("\nPredicted binary distribution:")
display(
    pd.Series(y_cic_pred_binary)
    .map({0: "Benign", 1: "Attack"})
    .value_counts()
    .to_frame("samples")
)

print("\nPredicted UNSW class distribution:")
display(
    cic_prediction_detail_df["predicted_unsw_class"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .to_frame("percent")
)

