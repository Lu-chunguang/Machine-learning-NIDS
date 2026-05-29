"""
Component 2B - Per-Attack Failure Analysis

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C2B-7

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C2B-7: Analyze source-only failure by CICIDS2018 attack type
# ========================================================================================

# Cell C2B-7: Analyze source-only failure by CICIDS2018 attack type
# Purpose:
# - Check which CIC attack types are detected or missed
# - Identify which UNSW class each CIC attack is mapped to most often

attack_only_detail_df = cic_prediction_detail_df[
    cic_prediction_detail_df["true_attack"] != "Benign"
].copy()

cic_attack_failure_rows = []

for attack_name, group_df in attack_only_detail_df.groupby("true_attack"):
    pred_binary_counts = group_df["predicted_binary"].value_counts(normalize=True)
    pred_unsw_counts = group_df["predicted_unsw_class"].value_counts(normalize=True)
    
    cic_attack_failure_rows.append({
        "cic_attack": attack_name,
        "samples": len(group_df),
        "detected_as_attack_rate": pred_binary_counts.get(1, 0),
        "missed_as_benign_rate": pred_binary_counts.get(0, 0),
        "most_common_predicted_unsw_class": pred_unsw_counts.idxmax(),
        "most_common_mapping_percent": pred_unsw_counts.max() * 100
    })

cic_attack_failure_df = (
    pd.DataFrame(cic_attack_failure_rows)
    .sort_values(by="samples", ascending=False)
    .round(4)
)

display(cic_attack_failure_df)

