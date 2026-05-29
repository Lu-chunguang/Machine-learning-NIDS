"""
Component 2B - Column Compatibility Audit

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C2B-2

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C2B-2: Audit CICIDS2018 column compatibility with UNSW
# ========================================================================================

# Cell C2B-2: Audit CICIDS2018 column compatibility with UNSW
# Purpose:
# - Verify that CICIDS2018 and UNSW share the same raw columns
# - Identify missing or extra columns before preprocessing
# - Confirm that model feature columns can be constructed consistently

unsw_columns = set(raw_df.columns)
cic_columns = set(cic_raw.columns)

missing_in_cic = sorted(list(unsw_columns - cic_columns))
extra_in_cic = sorted(list(cic_columns - unsw_columns))

print("Number of UNSW columns:", len(raw_df.columns))
print("Number of CICIDS2018 columns:", len(cic_raw.columns))

print("\nColumns missing in CICIDS2018:")
print(missing_in_cic if missing_in_cic else "None")

print("\nExtra columns in CICIDS2018:")
print(extra_in_cic if extra_in_cic else "None")

# Compare dtypes for shared columns
shared_columns = [col for col in raw_df.columns if col in cic_raw.columns]

dtype_compare_df = pd.DataFrame({
    "column": shared_columns,
    "UNSW_dtype": [str(raw_df[col].dtype) for col in shared_columns],
    "CIC_dtype": [str(cic_raw[col].dtype) for col in shared_columns]
})

dtype_mismatch_df = dtype_compare_df[
    dtype_compare_df["UNSW_dtype"] != dtype_compare_df["CIC_dtype"]
]

print("\nColumns with dtype differences:")
if dtype_mismatch_df.empty:
    print("None")
else:
    display(dtype_mismatch_df)

print("\nFeature columns expected by Component 1:")
print("Raw X columns:", len(X_raw.columns))
print("Categorical columns:", categorical_feature_cols)
print("Numeric columns:", len(numeric_feature_cols))

