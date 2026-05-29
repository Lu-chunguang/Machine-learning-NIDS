"""
Component 2B - Source-Domain Preprocessing on CICIDS2018

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C2B-3, C2B-4

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C2B-3: Prepare CICIDS2018 binary labels and raw features
# ========================================================================================

# Cell C2B-3: Prepare CICIDS2018 binary labels and raw features
# Purpose:
# - Convert CIC labels to binary detection labels
# - Build CIC raw feature matrix using the same feature columns as UNSW
# - Do not fit any preprocessing on CIC

# Binary label:
# Benign = 0
# Attack = 1
y_cic_binary = (cic_raw["Attack"] != "Benign").astype(int).to_numpy()

# Drop the same non-feature columns used in Component 1
X_cic_raw = cic_raw.drop(columns=drop_from_features).copy()

# Reorder CIC raw features to match UNSW raw feature order
X_cic_raw = X_cic_raw[X_raw.columns]

print("CIC raw feature shape:", X_cic_raw.shape)
print("UNSW raw feature shape:", X_raw.shape)

print("\nColumn alignment correct:")
print(list(X_cic_raw.columns) == list(X_raw.columns))

print("\nCIC binary label distribution:")
display(
    pd.Series(y_cic_binary)
    .map({0: "Benign", 1: "Attack"})
    .value_counts()
    .to_frame("samples")
)


# ========================================================================================
# Notebook Cell C2B-4: Transform CICIDS2018 using UNSW-fitted preprocessing pipeline
# ========================================================================================

# Cell C2B-4: Transform CICIDS2018 using UNSW-fitted preprocessing pipeline
# Purpose:
# - Apply the Component 1 preprocessing pipeline to CIC
# - Do not fit preprocessing on CIC
# - Keep this as a source-only distribution-shift test

X_cic_processed = baseline_preprocessor.transform(X_cic_raw)

print("CIC processed feature shape:", X_cic_processed.shape)
print("UNSW processed train shape:", X_train_processed.shape)

print("\nProcessed feature alignment correct:")
print(X_cic_processed.shape[1] == X_train_processed.shape[1])

print("\nReady for source-only prediction.")

