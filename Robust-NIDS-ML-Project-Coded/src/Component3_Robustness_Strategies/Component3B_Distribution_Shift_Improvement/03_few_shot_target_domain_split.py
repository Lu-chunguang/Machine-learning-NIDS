"""
Component 3B - Few-Shot Target-Domain Split

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3B-6

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3B-6: Split CICIDS2018 into few-shot calibration and holdout sets
# ========================================================================================

# Cell C3B-6: Split CICIDS2018 into few-shot calibration and holdout sets
# Purpose:
# - Select a small labeled CIC subset for target-domain calibration
# - Keep enough CIC samples for holdout evaluation
# - Avoid copying or synthetic oversampling

fewshot_rows = []

benign_fewshot_n = 1000
attack_fewshot_max_n = 50

for attack_name, group_df in cic_raw.groupby("Attack"):
    class_size = len(group_df)

    if attack_name == "Benign":
        n_take = min(benign_fewshot_n, max(class_size - 1, 0))
    else:
        n_take = min(attack_fewshot_max_n, max(class_size - 1, 0))

    if n_take > 0:
        sampled_group = group_df.sample(
            n=n_take,
            random_state=RANDOM_STATE
        )
        fewshot_rows.append(sampled_group)

cic_fewshot_df = pd.concat(fewshot_rows).sort_index()
cic_holdout_df = cic_raw.drop(index=cic_fewshot_df.index).copy()

cic_split_distribution_df = pd.DataFrame({
    "fewshot_samples": cic_fewshot_df["Attack"].value_counts(),
    "holdout_samples": cic_holdout_df["Attack"].value_counts()
}).fillna(0).astype(int)

cic_split_distribution_df["total_samples"] = (
    cic_split_distribution_df["fewshot_samples"] +
    cic_split_distribution_df["holdout_samples"]
)

cic_split_distribution_df = cic_split_distribution_df.sort_values(
    by="total_samples",
    ascending=False
)

display(cic_split_distribution_df)

