"""
Component 1 - Baseline Preparation and Split

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C1-3A, C1-3B, C1-4A, C1-4B, C1-4C

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C1-3A: Define baseline targets and feature groups
# ========================================================================================

# Cell C1-3A: Define baseline targets and feature groups

# Component 1 target:
# Attack is the multi-class label for the baseline classifier
y_multiclass_raw = raw_df["Attack"].copy()

# Label is kept for later binary experiments such as Stress Test B
y_binary_raw = raw_df["Label"].astype(int).copy()

# Columns that must not enter the model feature matrix
drop_from_features = [
    "IPV4_SRC_ADDR",   # IP address: host identifier, not a robust traffic pattern
    "IPV4_DST_ADDR",   # IP address: host identifier, not a robust traffic pattern
    "Attack",          # multi-class target label
    "Label"            # binary target label; including it would cause label leakage
]

# Raw feature table for Component 1
X_raw = raw_df.drop(columns=drop_from_features).copy()

# Protocol columns are categorical even though they are stored as numbers
categorical_feature_cols = [
    "PROTOCOL",
    "L7_PROTO"
]

categorical_feature_cols = [
    col for col in categorical_feature_cols
    if col in X_raw.columns
]

numeric_feature_cols = [
    col for col in X_raw.columns
    if col not in categorical_feature_cols
]

print("Component 1 target:", "Attack multi-class label")
print("Raw feature shape:", X_raw.shape)
print("Number of categorical feature columns:", len(categorical_feature_cols))
print("Categorical feature columns:", categorical_feature_cols)
print("Number of numeric feature columns:", len(numeric_feature_cols))

print("\nColumns excluded from model features:")
print(drop_from_features)


# ========================================================================================
# Notebook Cell C1-3B: Define numeric feature groups for baseline preprocessing
# ========================================================================================

# Cell C1-3B: Define numeric feature groups for baseline preprocessing

# Non-negative traffic-volume features selected for log1p transformation
log_numeric_feature_cols = [
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "DNS_TTL_ANSWER"
]

# Keep only columns present in the numeric feature set
log_numeric_feature_cols = [
    col for col in log_numeric_feature_cols
    if col in numeric_feature_cols
]

# Numeric columns not selected for log1p
regular_numeric_feature_cols = [
    col for col in numeric_feature_cols
    if col not in log_numeric_feature_cols
]

print("Log1p numeric feature count:", len(log_numeric_feature_cols))
print("Log1p numeric features:")
print(log_numeric_feature_cols)

print("\nRegular numeric feature count:", len(regular_numeric_feature_cols))
print("Regular numeric features:")
print(regular_numeric_feature_cols)

# Sanity checks: no overlap and no missing numeric features
numeric_overlap = set(log_numeric_feature_cols) & set(regular_numeric_feature_cols)
numeric_covered = set(log_numeric_feature_cols) | set(regular_numeric_feature_cols)

print("\nNumeric feature overlap:", numeric_overlap)
print("All numeric features covered:", numeric_covered == set(numeric_feature_cols))


# ========================================================================================
# Notebook Cell C1-4A: Create stratified train, validation, and test splits
# ========================================================================================

# Cell C1-4A: Create stratified train, validation, and test splits

from sklearn.model_selection import train_test_split

# First split: 70% training, 30% temporary holdout
X_train_raw, X_temp_raw, y_train_text, y_temp_text = train_test_split(
    X_raw,
    y_multiclass_raw,
    test_size=0.30,
    stratify=y_multiclass_raw,
    random_state=RANDOM_STATE
)

# Second split: 10% validation, 20% test from the full dataset
X_val_raw, X_test_raw, y_val_text, y_test_text = train_test_split(
    X_temp_raw,
    y_temp_text,
    test_size=2/3,
    stratify=y_temp_text,
    random_state=RANDOM_STATE
)

print("Train shape:", X_train_raw.shape, y_train_text.shape)
print("Validation shape:", X_val_raw.shape, y_val_text.shape)
print("Test shape:", X_test_raw.shape, y_test_text.shape)

split_summary = pd.DataFrame({
    "split": ["Train", "Validation", "Test"],
    "samples": [len(X_train_raw), len(X_val_raw), len(X_test_raw)],
    "percent_of_full_dataset": [
        len(X_train_raw) / len(X_raw) * 100,
        len(X_val_raw) / len(X_raw) * 100,
        len(X_test_raw) / len(X_raw) * 100
    ]
})

display(split_summary.round(2))


# ========================================================================================
# Notebook Cell C1-4B: Verify class balance across data splits
# ========================================================================================

# Cell C1-4B: Verify class balance across data splits

attack_class_order = attack_distribution["attack_class"].tolist()

split_class_counts = pd.DataFrame({
    "Train": y_train_text.value_counts(),
    "Validation": y_val_text.value_counts(),
    "Test": y_test_text.value_counts()
}).reindex(attack_class_order)

split_class_percent = (
    split_class_counts
    .div(split_class_counts.sum(axis=0), axis=1)
    .mul(100)
)

print("Class counts in each split:")
display(split_class_counts.astype(int))

print("Class percentages in each split:")
display(split_class_percent.round(4))


# ========================================================================================
# Notebook Cell C1-4C: Encode multi-class Attack labels after splitting
# ========================================================================================

# Cell C1-4C: Encode multi-class Attack labels after splitting

from sklearn.preprocessing import LabelEncoder

attack_label_encoder = LabelEncoder()

# Fit only on the training labels
y_train = attack_label_encoder.fit_transform(y_train_text)

# Apply the learned mapping to validation and test labels
y_val = attack_label_encoder.transform(y_val_text)
y_test = attack_label_encoder.transform(y_test_text)

attack_class_names = attack_label_encoder.classes_.tolist()

attack_label_mapping = pd.DataFrame({
    "encoded_label": range(len(attack_class_names)),
    "attack_class": attack_class_names
})

print("Encoded Attack classes:")
display(attack_label_mapping)

print("Encoded label shapes:")
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)
print("y_test:", y_test.shape)

