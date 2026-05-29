"""
Component 1 - Baseline Feature Preprocessing

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C1-5A, C1-5B, C1-5C, C1-5D

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C1-5A: Inspect categorical protocol features before encoding
# ========================================================================================

# Cell C1-5A: Inspect categorical protocol features before encoding

categorical_audit_rows = []

for col in categorical_feature_cols:
    train_values = X_train_raw[col]
    
    categorical_audit_rows.append({
        "feature": col,
        "dtype": str(train_values.dtype),
        "train_unique_values": train_values.nunique(),
        "missing_values": train_values.isna().sum()
    })

categorical_audit = pd.DataFrame(categorical_audit_rows)

print("Categorical feature audit on the training split:")
display(categorical_audit)

print("PROTOCOL values in training data:")
print(sorted(X_train_raw["PROTOCOL"].dropna().unique()))

l7_train_values = X_train_raw["L7_PROTO"].dropna()

non_integer_l7_count = (
    (l7_train_values % 1 != 0)
    .sum()
)

print("\nL7_PROTO unique values in training data:", l7_train_values.nunique())
print("L7_PROTO values with a fractional part:", int(non_integer_l7_count))

print("\nFirst 30 L7_PROTO values by frequency:")
display(
    X_train_raw["L7_PROTO"]
    .value_counts()
    .head(30)
    .to_frame("samples")
)


# ========================================================================================
# Notebook Cell C1-5B: Verify selected log1p features are non-negative
# ========================================================================================

# Cell C1-5B: Verify selected log1p features are non-negative

log_feature_minimums = (
    X_train_raw[log_numeric_feature_cols]
    .min()
    .rename("train_min")
    .reset_index()
    .rename(columns={"index": "feature"})
)

negative_log_features = log_feature_minimums[
    log_feature_minimums["train_min"] < 0
]

print("Minimum values of selected log1p features on the training split:")
display(log_feature_minimums)

print("\nSelected log1p features with negative training values:")
if negative_log_features.empty:
    print("None. All selected log1p features are non-negative in the training split.")
else:
    display(negative_log_features)


# ========================================================================================
# Notebook Cell C1-5C: Build the baseline preprocessing pipeline
# ========================================================================================

# Cell C1-5C: Build the baseline preprocessing pipeline

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
import numpy as np

# 1. Numeric features that need log1p first
log_numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", StandardScaler())
])

# 2. Numeric features that do not need log1p
regular_numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 3. Categorical features: PROTOCOL and L7_PROTO
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32
    ))
])

# 4. Combine all preprocessing rules
baseline_preprocessor = ColumnTransformer(
    transformers=[
        ("log_numeric", log_numeric_pipeline, log_numeric_feature_cols),
        ("regular_numeric", regular_numeric_pipeline, regular_numeric_feature_cols),
        ("categorical", categorical_pipeline, categorical_feature_cols)
    ],
    remainder="drop",
    sparse_threshold=1.0
)

print("Baseline preprocessor is ready.")
print("Log1p numeric features:", len(log_numeric_feature_cols))
print("Regular numeric features:", len(regular_numeric_feature_cols))
print("Categorical features:", categorical_feature_cols)


# ========================================================================================
# Notebook Cell C1-5D: Fit preprocessing on training data and transform all splits
# ========================================================================================

# Cell C1-5D: Fit preprocessing on training data and transform all splits

from scipy import sparse

# Learn preprocessing rules from the training split only
X_train_processed = baseline_preprocessor.fit_transform(X_train_raw)

# Apply the same learned rules to validation and test splits
X_val_processed = baseline_preprocessor.transform(X_val_raw)
X_test_processed = baseline_preprocessor.transform(X_test_raw)

# Save processed feature names for later interpretation
baseline_feature_names = baseline_preprocessor.get_feature_names_out()

print("Processed train shape:", X_train_processed.shape)
print("Processed validation shape:", X_val_processed.shape)
print("Processed test shape:", X_test_processed.shape)

print("\nSparse matrix output:")
print("Train:", sparse.issparse(X_train_processed))
print("Validation:", sparse.issparse(X_val_processed))
print("Test:", sparse.issparse(X_test_processed))

print("\nNumber of processed features:", len(baseline_feature_names))
print("First 20 processed feature names:")
print(baseline_feature_names[:20])

