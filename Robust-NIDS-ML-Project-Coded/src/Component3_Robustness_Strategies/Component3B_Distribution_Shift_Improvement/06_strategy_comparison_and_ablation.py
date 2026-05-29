"""
Component 3B - Strategy Comparison and Ablation

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3B-11

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3B-11: Final Component 3B strategy comparison
# ========================================================================================

# Cell C3B-11: Final Component 3B strategy comparison
# Purpose:
# - Compare source-only, cost-sensitive, and few-shot calibrated strategies
# - Summarize the trade-off between recall, F1, and false positive rate
# - Support the Component 3B ablation discussion

c3b_strategy_comparison_df = pd.DataFrame([
    {
        "strategy": "Source-only UNSW XGBoost",
        "training_data": "UNSW only",
        "test_data": "CICIDS2018 sample",
        "accuracy": c2b_final_metrics_df.loc[0, "accuracy"],
        "precision": c2b_final_metrics_df.loc[0, "precision"],
        "recall": c2b_final_metrics_df.loc[0, "recall"],
        "f1": c2b_final_metrics_df.loc[0, "f1"],
        "fpr": c2b_final_metrics_df.loc[0, "fpr"],
        "fnr": c2b_final_metrics_df.loc[0, "fnr"],
        "tp": c2b_final_metrics_df.loc[0, "tp"],
        "fp": c2b_final_metrics_df.loc[0, "fp"],
        "fn": c2b_final_metrics_df.loc[0, "fn"]
    },
    {
        "strategy": "Cost-sensitive binary XGBoost",
        "training_data": "UNSW only with attack weighting",
        "test_data": "CICIDS2018 sample",
        "accuracy": cic_cost_metrics_df.loc[0, "accuracy"],
        "precision": cic_cost_metrics_df.loc[0, "precision"],
        "recall": cic_cost_metrics_df.loc[0, "recall"],
        "f1": cic_cost_metrics_df.loc[0, "f1"],
        "fpr": cic_cost_metrics_df.loc[0, "fpr"],
        "fnr": cic_cost_metrics_df.loc[0, "fnr"],
        "tp": cic_cost_metrics_df.loc[0, "tp"],
        "fp": cic_cost_metrics_df.loc[0, "fp"],
        "fn": cic_cost_metrics_df.loc[0, "fn"]
    },
    {
        "strategy": "Few-shot target calibration",
        "training_data": "UNSW + CIC few-shot",
        "test_data": "CICIDS2018 holdout",
        "accuracy": cic_fewshot_metrics_df.loc[0, "accuracy"],
        "precision": cic_fewshot_metrics_df.loc[0, "precision"],
        "recall": cic_fewshot_metrics_df.loc[0, "recall"],
        "f1": cic_fewshot_metrics_df.loc[0, "f1"],
        "fpr": cic_fewshot_metrics_df.loc[0, "fpr"],
        "fnr": cic_fewshot_metrics_df.loc[0, "fnr"],
        "tp": cic_fewshot_metrics_df.loc[0, "tp"],
        "fp": cic_fewshot_metrics_df.loc[0, "fp"],
        "fn": cic_fewshot_metrics_df.loc[0, "fn"]
    }
])

display(c3b_strategy_comparison_df.round(4))

