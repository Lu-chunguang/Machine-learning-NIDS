"""
Component 1 - Initial Dataset Audit

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C1-0A, C1-0B, C1-0C

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C1-0A: Project setup and load the UNSW dataset
# ========================================================================================

# Cell C1-0A: Project setup and load the UNSW dataset

from pathlib import Path
import warnings
import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

# Keep notebook output cleaner
warnings.filterwarnings("ignore")

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Dataset path
# Original notebook path was an absolute Windows path.
# In this VS Code project, place the dataset at data/raw/NF-UNSW-NB15-v2.csv
# or set the UNSW_PATH environment variable.
PROJECT_ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
UNSW_PATH = Path(os.getenv("UNSW_PATH", PROJECT_ROOT / "data" / "raw" / "NF-UNSW-NB15-v2.csv"))

# Load the original dataset
# raw_df will stay unchanged so later experiments can reuse the raw data safely
raw_df = pd.read_csv(UNSW_PATH)

print("UNSW dataset loaded.")
print("Dataset shape:", raw_df.shape)

display(raw_df.head())


# ========================================================================================
# Notebook Cell C1-0B: Check dataset structure and label meaning
# ========================================================================================

# Cell C1-0B: Check dataset structure and label meaning

print("Number of rows:", raw_df.shape[0])
print("Number of columns:", raw_df.shape[1])

# Show every column name and its data type
structure_df = pd.DataFrame({
    "column": raw_df.columns,
    "dtype": raw_df.dtypes.astype(str).values
})

print("\nColumn structure:")
display(structure_df)

# Check the values in the two label columns
print("\nUnique values in Label:")
print(sorted(raw_df["Label"].dropna().unique()))

print("\nAttack classes:")
print(raw_df["Attack"].dropna().unique())

# Confirm how binary Label corresponds to multi-class Attack
label_attack_map = (
    raw_df
    .groupby(["Label", "Attack"])
    .size()
    .reset_index(name="samples")
    .sort_values(["Label", "samples"], ascending=[True, False])
)

print("\nMapping between Label and Attack:")
display(label_attack_map)


# ========================================================================================
# Notebook Cell C1-0C: Check binary and multi-class label distributions
# ========================================================================================

# Cell C1-0C: Check binary and multi-class label distributions
# 目的：报告类别分布，并观察 class imbalance

# Binary label distribution: 0 = Benign, 1 = Attack
binary_distribution = (
    raw_df["Label"]
    .map({0: "Benign", 1: "Attack"})
    .value_counts()
    .rename_axis("binary_label")
    .reset_index(name="samples")
)

binary_distribution["percent"] = (
    binary_distribution["samples"] / len(raw_df) * 100
).round(4)

print("Binary label distribution:")
display(binary_distribution)

# Multi-class attack distribution
attack_distribution = (
    raw_df["Attack"]
    .value_counts()
    .rename_axis("attack_class")
    .reset_index(name="samples")
)

attack_distribution["percent"] = (
    attack_distribution["samples"] / len(raw_df) * 100
).round(4)

print("Multi-class Attack distribution:")
display(attack_distribution)

# Plot binary distribution
plt.figure(figsize=(6, 4))
sns.barplot(
    data=binary_distribution,
    x="binary_label",
    y="samples",
    hue="binary_label",
    palette="Set2",
    legend=False
)
plt.title("Binary Label Distribution")
plt.xlabel("Binary Label")
plt.ylabel("Samples")
plt.tight_layout()
plt.show()

# Plot multi-class distribution
plt.figure(figsize=(12, 5))
sns.barplot(
    data=attack_distribution,
    x="attack_class",
    y="samples",
    hue="attack_class",
    palette="viridis",
    legend=False
)
plt.yscale("log")
plt.title("Multi-Class Attack Distribution")
plt.xlabel("Attack Class")
plt.ylabel("Samples (log scale)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()

