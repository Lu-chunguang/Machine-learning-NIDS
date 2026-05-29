"""
Component 3C - Augmentation Design

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3C-1

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3C-1: Define augmentation design
# ========================================================================================

# Cell C3C-1: Define augmentation design
# Purpose:
# - Document augmentation variants and sample sizes
# - Keep the augmentation setup explicit before training

c3c_augmentation_design_df = pd.DataFrame([
    {
        "variant": "random_zeroing",
        "augmented_samples": 200_000,
        "gaussian_noise": "No",
        "zeroing_ratio": 0.25,
        "notes": "Proposal random zeroing at the middle masking level"
    },
    {
        "variant": "gaussian_noise",
        "augmented_samples": 200_000,
        "gaussian_noise": "sigma sampled uniformly from 0.1 to 1.0",
        "zeroing_ratio": 0.00,
        "notes": "Proposal Gaussian noise"
    },
    {
        "variant": "combined_noise_zeroing",
        "augmented_samples": 200_000,
        "gaussian_noise": "sigma sampled uniformly from 0.1 to 1.0",
        "zeroing_ratio": 0.25,
        "notes": "Gaussian noise plus random zeroing"
    },
    {
        "variant": "multi_ratio_zeroing",
        "augmented_samples": 300_000,
        "gaussian_noise": "No",
        "zeroing_ratio": "0.10, 0.25, 0.50",
        "notes": "Ablation-style ratio variation"
    }
])

display(c3c_augmentation_design_df)

