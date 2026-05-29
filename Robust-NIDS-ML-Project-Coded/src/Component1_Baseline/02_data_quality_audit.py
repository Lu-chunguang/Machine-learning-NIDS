"""
Component 1 - Data Quality Audit

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C1-1A, C1-1B

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C1-1A: Check missing values, infinite values, and duplicate rows
# ========================================================================================

# Cell C1-1A: Check missing values, infinite values, and duplicate rows

# 1. Missing-value audit
missing_count = raw_df.isna().sum()
missing_percent = (missing_count / len(raw_df) * 100).round(4)

missing_audit = pd.DataFrame({
    "column": raw_df.columns,
    "missing_count": missing_count.values,
    "missing_percent": missing_percent.values
})

missing_audit = missing_audit[missing_audit["missing_count"] > 0]

print("Missing-value audit:")
if missing_audit.empty:
    print("No missing values found.")
else:
    display(missing_audit.sort_values("missing_percent", ascending=False))


# 2. Infinite-value audit for numeric columns only
numeric_cols_raw = raw_df.select_dtypes(include=[np.number]).columns

inf_count = np.isinf(raw_df[numeric_cols_raw]).sum()

inf_audit = pd.DataFrame({
    "column": numeric_cols_raw,
    "infinite_count": inf_count.values
})

inf_audit = inf_audit[inf_audit["infinite_count"] > 0]

print("\nInfinite-value audit:")
if inf_audit.empty:
    print("No infinite values found in numeric columns.")
else:
    display(inf_audit.sort_values("infinite_count", ascending=False))


# 3. Duplicate-row audit
duplicate_rows = raw_df.duplicated().sum()

print("\nDuplicate-row audit:")
print("Duplicate rows:", duplicate_rows)
print("Duplicate row percent:", round(duplicate_rows / len(raw_df) * 100, 4), "%")


# ========================================================================================
# Notebook Cell C1-1B: Check placeholder values in text columns
# ========================================================================================

# Cell C1-1B: Check placeholder values in text columns

# Text columns in the raw dataset
text_cols_raw = raw_df.select_dtypes(include="object").columns.tolist()

# Common text placeholders that may represent hidden missing values
placeholder_tokens = {
    "", "?", "na", "n/a", "none", "null", "nan", "unknown", "-"
}

placeholder_rows = []

for col in text_cols_raw:
    cleaned_text = (
        raw_df[col]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    
    placeholder_count = cleaned_text.isin(placeholder_tokens).sum()
    
    placeholder_rows.append({
        "column": col,
        "placeholder_count": int(placeholder_count)
    })

placeholder_audit = pd.DataFrame(placeholder_rows)

print("Text columns checked:")
print(text_cols_raw)

print("\nPlaceholder-value audit:")
if (placeholder_audit["placeholder_count"] == 0).all():
    print("No placeholder values found in text columns.")
else:
    display(
        placeholder_audit[placeholder_audit["placeholder_count"] > 0]
        .sort_values("placeholder_count", ascending=False)
    )

