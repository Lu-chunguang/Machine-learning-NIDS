"""
Component 3A - Confidence Thresholding with Rejection

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3A-1, C3A-1B, C3A-2, C3A-3, C3A-4, C3A-5, C3A-6

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3A-1: Prepare C2A detail tables for threshold analysis
# ========================================================================================

# Cell C3A-1: Prepare C2A detail tables for threshold analysis
# Purpose:
# - Use the sample-level prediction tables saved in C2A
# - A1 represents lower-confidence unknowns
# - A2 represents high-confidence unknowns, often predicted as Benign

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

c3a_a1_detail_df = c2a_a1_detail_df.copy()
c3a_a2_detail_df = c2a_a2_detail_df.copy()

c3a_detail_summary_df = pd.DataFrame({
    "scenario": ["A1_weak_classes", "A2_common_attack_classes"],
    "samples": [len(c3a_a1_detail_df), len(c3a_a2_detail_df)],
    "unknown_samples": [
        int(c3a_a1_detail_df["is_unknown"].sum()),
        int(c3a_a2_detail_df["is_unknown"].sum())
    ],
    "unknown_mean_confidence": [
        c3a_a1_detail_df.loc[c3a_a1_detail_df["is_unknown"], "confidence"].mean(),
        c3a_a2_detail_df.loc[c3a_a2_detail_df["is_unknown"], "confidence"].mean()
    ],
    "unknown_median_confidence": [
        c3a_a1_detail_df.loc[c3a_a1_detail_df["is_unknown"], "confidence"].median(),
        c3a_a2_detail_df.loc[c3a_a2_detail_df["is_unknown"], "confidence"].median()
    ]
}).round(4)

display(c3a_detail_summary_df)


# ========================================================================================
# Notebook Cell C3A-1B: AUROC of confidence as an unknown-class detector
# ========================================================================================

# Cell C3A-1B: AUROC of confidence as an unknown-class detector
# Purpose:
# - Satisfy the proposal metric for Stress Test A
# - Use 1 - confidence because lower classifier confidence should indicate higher unknown risk

from sklearn.metrics import roc_auc_score

c3a_confidence_auroc_rows = []

for scenario_name, detail_df in [
    ("A1_weak_classes", c3a_a1_detail_df),
    ("A2_common_attack_classes", c3a_a2_detail_df)
]:
    y_unknown = detail_df["is_unknown"].astype(int)
    unknown_score = 1 - detail_df["confidence"]
    
    c3a_confidence_auroc_rows.append({
        "scenario": scenario_name,
        "unknown_detector_score": "1 - confidence",
        "known_samples": int((~detail_df["is_unknown"]).sum()),
        "unknown_samples": int(detail_df["is_unknown"].sum()),
        "auroc": roc_auc_score(y_unknown, unknown_score)
    })

c3a_confidence_auroc_df = pd.DataFrame(c3a_confidence_auroc_rows)
display(c3a_confidence_auroc_df.round(4))


# ========================================================================================
# Notebook Cell C3A-2: Sweep confidence thresholds for Scenario A1
# ========================================================================================

# Cell C3A-2: Sweep confidence thresholds for Scenario A1
# Purpose:
# - Reject samples with confidence below the threshold
# - Measure unknown rejection, known false rejection, coverage, and accepted known-class accuracy

c3a_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
c3a_a1_threshold_rows = []

for threshold in c3a_thresholds:
    rejected = c3a_a1_detail_df["confidence"] < threshold
    accepted = ~rejected
    
    known_mask = ~c3a_a1_detail_df["is_unknown"]
    unknown_mask = c3a_a1_detail_df["is_unknown"]
    
    accepted_known_mask = accepted & known_mask
    accepted_known_accuracy = (
        (
            c3a_a1_detail_df.loc[accepted_known_mask, "true_attack"] ==
            c3a_a1_detail_df.loc[accepted_known_mask, "predicted_known_class"]
        ).mean()
        if accepted_known_mask.sum() else np.nan
    )
    
    c3a_a1_threshold_rows.append({
        "threshold": threshold,
        "overall_coverage": accepted.mean(),
        "accepted_known_accuracy": accepted_known_accuracy,
        "known_rejection_rate": rejected[known_mask].mean(),
        "unknown_rejection_rate": rejected[unknown_mask].mean(),
        "known_accepted_samples": int(accepted_known_mask.sum()),
        "known_rejected_samples": int((rejected & known_mask).sum()),
        "unknown_accepted_samples": int((accepted & unknown_mask).sum()),
        "unknown_rejected_samples": int((rejected & unknown_mask).sum())
    })

c3a_a1_threshold_df = pd.DataFrame(c3a_a1_threshold_rows).round(4)
display(c3a_a1_threshold_df)


# ========================================================================================
# Notebook Cell C3A-3: Plot Scenario A1 threshold trade-off
# ========================================================================================

# Cell C3A-3: Plot Scenario A1 threshold trade-off
# Purpose:
# - Show the proposal-required coverage vs accepted accuracy relationship
# - Also show unknown rejection and known false rejection for deployment interpretation

plt.figure(figsize=(8, 5))
plt.plot(
    c3a_a1_threshold_df["threshold"],
    c3a_a1_threshold_df["overall_coverage"],
    marker="o",
    label="Overall coverage"
)
plt.plot(
    c3a_a1_threshold_df["threshold"],
    c3a_a1_threshold_df["accepted_known_accuracy"],
    marker="o",
    label="Accepted known-class accuracy"
)
plt.plot(
    c3a_a1_threshold_df["threshold"],
    c3a_a1_threshold_df["unknown_rejection_rate"],
    marker="o",
    label="Unknown rejection rate"
)
plt.plot(
    c3a_a1_threshold_df["threshold"],
    c3a_a1_threshold_df["known_rejection_rate"],
    marker="o",
    label="Known false rejection rate"
)
plt.title("Scenario A1 Confidence Threshold Trade-off")
plt.xlabel("Confidence threshold")
plt.ylabel("Rate")
plt.ylim(-0.02, 1.02)
plt.grid(True)
plt.legend()
plt.show()


# ========================================================================================
# Notebook Cell C3A-4: Sweep confidence thresholds for Scenario A2
# ========================================================================================

# Cell C3A-4: Sweep confidence thresholds for Scenario A2
# Purpose:
# - Apply the same rejection rule to A2
# - A2 is harder because many unknown attacks have high confidence

c3a_a2_threshold_rows = []

for threshold in c3a_thresholds:
    rejected = c3a_a2_detail_df["confidence"] < threshold
    accepted = ~rejected
    
    known_mask = ~c3a_a2_detail_df["is_unknown"]
    unknown_mask = c3a_a2_detail_df["is_unknown"]
    
    accepted_known_mask = accepted & known_mask
    accepted_known_accuracy = (
        (
            c3a_a2_detail_df.loc[accepted_known_mask, "true_attack"] ==
            c3a_a2_detail_df.loc[accepted_known_mask, "predicted_known_class"]
        ).mean()
        if accepted_known_mask.sum() else np.nan
    )
    
    c3a_a2_threshold_rows.append({
        "threshold": threshold,
        "overall_coverage": accepted.mean(),
        "accepted_known_accuracy": accepted_known_accuracy,
        "known_rejection_rate": rejected[known_mask].mean(),
        "unknown_rejection_rate": rejected[unknown_mask].mean(),
        "known_accepted_samples": int(accepted_known_mask.sum()),
        "known_rejected_samples": int((rejected & known_mask).sum()),
        "unknown_accepted_samples": int((accepted & unknown_mask).sum()),
        "unknown_rejected_samples": int((rejected & unknown_mask).sum())
    })

c3a_a2_threshold_df = pd.DataFrame(c3a_a2_threshold_rows).round(4)
display(c3a_a2_threshold_df)


# ========================================================================================
# Notebook Cell C3A-5: Plot Scenario A2 threshold trade-off
# ========================================================================================

# Cell C3A-5: Plot Scenario A2 threshold trade-off
# Purpose:
# - Show whether confidence thresholding remains useful for high-confidence unknowns

plt.figure(figsize=(8, 5))
plt.plot(
    c3a_a2_threshold_df["threshold"],
    c3a_a2_threshold_df["overall_coverage"],
    marker="o",
    label="Overall coverage"
)
plt.plot(
    c3a_a2_threshold_df["threshold"],
    c3a_a2_threshold_df["accepted_known_accuracy"],
    marker="o",
    label="Accepted known-class accuracy"
)
plt.plot(
    c3a_a2_threshold_df["threshold"],
    c3a_a2_threshold_df["unknown_rejection_rate"],
    marker="o",
    label="Unknown rejection rate"
)
plt.plot(
    c3a_a2_threshold_df["threshold"],
    c3a_a2_threshold_df["known_rejection_rate"],
    marker="o",
    label="Known false rejection rate"
)
plt.title("Scenario A2 Confidence Threshold Trade-off")
plt.xlabel("Confidence threshold")
plt.ylabel("Rate")
plt.ylim(-0.02, 1.02)
plt.grid(True)
plt.legend()
plt.show()


# ========================================================================================
# Notebook Cell C3A-6: Compare selected threshold on A1 and A2
# ========================================================================================

# Cell C3A-6: Compare selected threshold on A1 and A2
# Purpose:
# - Use threshold = 0.90 as the practical operating point
# - Compare rejection, coverage, and accepted known-class accuracy across both scenarios

c3a_selected_threshold = 0.90

c3a_a1_selected = c3a_a1_threshold_df[
    c3a_a1_threshold_df["threshold"] == c3a_selected_threshold
].copy()
c3a_a1_selected["scenario"] = "A1_weak_classes"

c3a_a2_selected = c3a_a2_threshold_df[
    c3a_a2_threshold_df["threshold"] == c3a_selected_threshold
].copy()
c3a_a2_selected["scenario"] = "A2_common_attack_classes"

c3a_threshold_comparison_df = pd.concat(
    [c3a_a1_selected, c3a_a2_selected],
    ignore_index=True
)

c3a_threshold_comparison_df = c3a_threshold_comparison_df[
    [
        "scenario",
        "threshold",
        "overall_coverage",
        "accepted_known_accuracy",
        "known_rejection_rate",
        "unknown_rejection_rate",
        "known_rejected_samples",
        "unknown_rejected_samples",
        "unknown_accepted_samples"
    ]
]

display(c3a_threshold_comparison_df.round(4))

