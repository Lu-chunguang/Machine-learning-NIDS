"""
Component 1 - Exploratory Feature Analysis

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C1-2A, C1-2B, C1-2C, C1-2D

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C1-2A: Summarize representative NetFlow features
# ========================================================================================

# Cell C1-2A: Summarize representative NetFlow features

# Core features selected for early inspection
core_feature_cols = [
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
    "MIN_TTL",
    "MAX_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT"
]

core_feature_cols = [
    col for col in core_feature_cols
    if col in raw_df.columns
]

core_feature_summary = (
    raw_df[core_feature_cols]
    .describe()
    .T
    .round(3)
)

print("Representative NetFlow feature summary:")
display(core_feature_summary)


# ========================================================================================
# Notebook Cell C1-2B: Visualize distributions of representative NetFlow features
# ========================================================================================

# Cell C1-2B: Visualize distributions of representative NetFlow features

# Use a sample for faster visualization only
plot_sample = raw_df.sample(
    n=min(200_000, len(raw_df)),
    random_state=RANDOM_STATE
)

# Log1p is used only for visualization here
# It makes highly skewed non-negative features easier to inspect
core_plot_df = np.log1p(plot_sample[core_feature_cols].clip(lower=0))

core_plot_df.hist(
    bins=50,
    figsize=(16, 12),
    edgecolor="black"
)

plt.suptitle("Log-Transformed Distributions of Representative NetFlow Features")
plt.tight_layout()
plt.show()


# ========================================================================================
# Notebook Cell C1-2C: Inspect correlations among representative NetFlow features
# ========================================================================================

# Cell C1-2C: Inspect correlations among representative NetFlow features

core_corr = raw_df[core_feature_cols].corr()

plt.figure(figsize=(11, 9))

corr_mask = np.triu(
    np.ones_like(core_corr, dtype=bool)
)

sns.heatmap(
    core_corr,
    mask=corr_mask,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix of Representative NetFlow Features")
plt.tight_layout()
plt.show()


# ========================================================================================
# Notebook Cell C1-2D: Compare representative feature medians by attack class
# ========================================================================================

# Cell C1-2D: Compare representative feature medians by attack class

attack_compare_cols = [
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "MIN_TTL",
    "MAX_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT"
]

attack_compare_cols = [
    col for col in attack_compare_cols
    if col in raw_df.columns
]

attack_feature_medians = (
    raw_df
    .groupby("Attack")[attack_compare_cols]
    .median()
    .round(2)
)

attack_support = (
    raw_df["Attack"]
    .value_counts()
    .rename("samples")
)

attack_feature_medians = (
    attack_feature_medians
    .join(attack_support)
    .sort_values("samples", ascending=False)
)

# Put sample count first
attack_feature_medians = attack_feature_medians[
    ["samples"] + attack_compare_cols
]

print("Median values of representative features by Attack class:")
display(attack_feature_medians)

