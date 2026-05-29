"""
Component 2B - Load CICIDS2018 Dataset

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C2B-1

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C2B-1: Load CICIDS2018 sample and inspect basic structure
# ========================================================================================

import os
from pathlib import Path
# Cell C2B-1: Load CICIDS2018 sample and inspect basic structure
# Purpose:
# - Load a manageable CICIDS2018 sample first
# - Check columns, dtypes, and attack distribution
# - Prepare for source-only cross-dataset evaluation

# Original notebook path was an absolute Windows path.
# In this VS Code project, place the dataset at data/raw/NF-CSE-CIC-IDS2018-v2.csv
# or set the CICIDS_PATH environment variable.
PROJECT_ROOT = Path(__file__).resolve().parents[3] if "__file__" in globals() else Path.cwd()
cic_path = Path(os.getenv("CICIDS_PATH", PROJECT_ROOT / "data" / "raw" / "NF-CSE-CIC-IDS2018-v2.csv"))

cic_raw = pd.read_csv(cic_path, nrows=100_000)

print("CICIDS2018 sample shape:", cic_raw.shape)

print("\nFirst five rows:")
display(cic_raw.head())

print("\nCICIDS2018 columns and dtypes:")
display(pd.DataFrame({
    "column": cic_raw.columns,
    "dtype": cic_raw.dtypes.astype(str).values
}))

print("\nCICIDS2018 Attack distribution:")
display(cic_raw["Attack"].value_counts().to_frame("samples"))

print("\nCICIDS2018 binary label distribution:")
display(
    (cic_raw["Attack"] != "Benign")
    .map({False: "Benign", True: "Attack"})
    .value_counts()
    .to_frame("samples")
)

