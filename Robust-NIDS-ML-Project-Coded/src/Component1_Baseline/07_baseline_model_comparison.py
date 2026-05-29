"""
Component 1 - Baseline Model Comparison

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C1-6F, C1-6G, C1-6H

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C1-6F: Compare all baseline models
# ========================================================================================

# Cell C1-6F: Compare all baseline models

baseline_comparison_df = (
    pd.DataFrame(baseline_results)
    .set_index("model")
    .sort_values(by="macro_f1", ascending=False)
)

display(baseline_comparison_df.round(4))

best_baseline_name = baseline_comparison_df["macro_f1"].idxmax()
best_baseline_model = baseline_model_objects[best_baseline_name]
best_baseline_predictions = baseline_predictions[best_baseline_name]

print("Best validation model by macro F1:", best_baseline_name)
print("This model will be the default candidate for Component 2 stress tests.")


# ========================================================================================
# Notebook Cell C1-6G: Combine per-class validation reports for all baselines
# ========================================================================================

# Cell C1-6G: Combine per-class validation reports for all baselines

baseline_all_class_reports_df = (
    pd.concat(baseline_class_reports, names=["model", "attack_class"])
    .reset_index()
)

display(baseline_all_class_reports_df)

print("Weakest classes for the best validation model:")
display(
    baseline_class_reports[best_baseline_name]
    .sort_values(by="f1-score", ascending=True)
)


# ========================================================================================
# Notebook Cell C1-6H: Plot normalized confusion matrices for all baselines
# ========================================================================================

# Cell C1-6H: Plot normalized confusion matrices for all baselines

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.ravel()

for ax, model_name in zip(axes, baseline_confusion_matrices.keys()):
    sns.heatmap(
        baseline_confusion_matrices[model_name],
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        cbar=True
    )
    ax.set_title(f"{model_name} Normalized Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.show()

