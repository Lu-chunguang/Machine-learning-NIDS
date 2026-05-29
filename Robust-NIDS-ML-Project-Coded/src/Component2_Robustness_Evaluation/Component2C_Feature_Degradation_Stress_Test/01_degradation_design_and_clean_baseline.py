"""
Component 2C - Degradation Design and Clean Baseline

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C2C-1

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C2C-1: Degradation design and clean baseline
# ========================================================================================

# Cell C2C-1: Degradation design and clean baseline
# Purpose:
# - Evaluate clean baseline performance on UNSW test data
# - Define proposal-required feature degradation conditions

from sklearn.metrics import accuracy_score, f1_score
from scipy import sparse
import gc

y_test_clean_pred = selected_baseline_model.predict(X_test_processed).astype(int)

c2c_clean_baseline_df = pd.DataFrame([{
    "condition": "clean_test_features",
    "accuracy": accuracy_score(y_test, y_test_clean_pred),
    "macro_f1": f1_score(y_test, y_test_clean_pred, average="macro"),
    "weighted_f1": f1_score(y_test, y_test_clean_pred, average="weighted")
}])

c2c_noise_sigmas = [0.1, 0.5, 1.0]
c2c_masking_ratios = [0.10, 0.25, 0.50]
c2c_dropout_feature_counts = [2, 4, 6]
c2c_repeats = 5

c2c_degradation_design_df = pd.DataFrame([
    {
        "degradation_type": "gaussian_noise",
        "levels": str(c2c_noise_sigmas),
        "proposal_requirement": "sigma in {0.1, 0.5, 1.0}"
    },
    {
        "degradation_type": "random_masking",
        "levels": str(c2c_masking_ratios),
        "proposal_requirement": "p in {10%, 25%, 50%}"
    },
    {
        "degradation_type": "feature_dropout",
        "levels": str(c2c_dropout_feature_counts),
        "proposal_requirement": "k in {2, 4, 6}"
    }
])

display(c2c_clean_baseline_df.round(4))
display(c2c_degradation_design_df)

