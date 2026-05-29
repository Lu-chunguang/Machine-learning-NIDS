"""
Component 2A - Full Open-Set Mapping Matrix

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C2A-19

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C2A-19: Build full open-set mapping matrices for A1 and A2
# ========================================================================================

# Cell C2A-19: Build full open-set mapping matrices for A1 and A2
# Purpose:
# - Provide a confusion-style matrix for all true classes
# - Show how known and unknown classes map into the known output space

def build_open_set_mapping_matrix(detail_df):
    mapping_df = pd.crosstab(
        detail_df["true_attack"],
        detail_df["predicted_known_class"],
        normalize="index"
    ).mul(100).round(2)
    
    sample_counts = detail_df["true_attack"].value_counts().rename("samples")
    mapping_df = mapping_df.join(sample_counts)
    
    ordered_columns = ["samples"] + [
        col for col in mapping_df.columns
        if col != "samples"
    ]
    
    return mapping_df[ordered_columns]

c2a_a1_full_mapping_df = build_open_set_mapping_matrix(c2a_a1_detail_df)
c2a_a2_full_mapping_df = build_open_set_mapping_matrix(c2a_a2_detail_df)

print("Scenario A1 full open-set mapping matrix:")
display(c2a_a1_full_mapping_df)

print("Scenario A2 full open-set mapping matrix:")
display(c2a_a2_full_mapping_df)

