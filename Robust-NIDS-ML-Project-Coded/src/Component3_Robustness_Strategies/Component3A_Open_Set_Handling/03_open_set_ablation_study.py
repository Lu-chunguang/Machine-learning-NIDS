"""
Component 3A - Open-Set Ablation Study

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3A-11

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3A-11: Ablation study for C3A open-set handling on Scenario A2
# ========================================================================================

# Cell C3A-11: Ablation study for C3A open-set handling on Scenario A2
# Purpose:
# - Compare three variants:
#   1. Threshold only
#   2. Isolation Forest only
#   3. Threshold + Isolation Forest
# - This shows whether the two strategies are complementary

c3a_threshold = 0.90

a2_known_mask = ~c3a_a2_detail_df["is_unknown"]
a2_unknown_mask = c3a_a2_detail_df["is_unknown"]

# Variant 1: threshold only
threshold_flags_unknown = c3a_a2_detail_df["confidence"] < c3a_threshold

# Variant 2: Isolation Forest fallback only
# It only flags samples predicted as Benign and judged anomalous by Isolation Forest.
a2_pred_benign_mask = c3a_a2_detail_df["predicted_known_class"] == "Benign"
iforest_flags_unknown = pd.Series(False, index=c3a_a2_detail_df.index)
iforest_flags_unknown.loc[a2_pred_benign_mask] = (
    iforest_test_is_anomaly_series.loc[a2_pred_benign_mask].values
)

# Variant 3: combined strategy
combined_flags_unknown = threshold_flags_unknown | iforest_flags_unknown

ablation_rows = []

strategy_flags = {
    "Threshold only": threshold_flags_unknown,
    "Isolation Forest only": iforest_flags_unknown,
    "Threshold + Isolation Forest": combined_flags_unknown
}

for strategy_name, flags_unknown in strategy_flags.items():
    ablation_rows.append({
        "strategy": strategy_name,
        "unknown_detection_rate": flags_unknown[a2_unknown_mask].mean(),
        "known_false_rejection_rate": flags_unknown[a2_known_mask].mean(),
        "coverage": 1 - flags_unknown.mean(),
        "unknown_detected_samples": int(flags_unknown[a2_unknown_mask].sum()),
        "known_false_rejected_samples": int(flags_unknown[a2_known_mask].sum())
    })

c3a_a2_ablation_df = pd.DataFrame(ablation_rows).round(4)
display(c3a_a2_ablation_df)

