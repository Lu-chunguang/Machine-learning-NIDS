"""
Component 3B - Cost Weight Trade-Off

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C3B-5B

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C3B-5B: Sweep cost weights for binary XGBoost
# ========================================================================================

# Cell C3B-5B: Sweep cost weights for binary XGBoost
# Purpose:
# - Evaluate the Strategy 4 FNR/FPR trade-off as cost weight varies
# - Compare source validation behavior with target-domain CIC behavior

c3b_cost_weight_values = [
    1.0,
    3.0,
    5.0,
    8.0,
    12.0,
    16.0,
    round(scale_pos_weight_c3b, 4)
]

c3b_cost_sweep_rows = []

for cost_weight in c3b_cost_weight_values:
    sweep_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=cost_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    start_time = time.time()
    sweep_model.fit(X_train_processed, y_train_binary)
    sweep_train_time = time.time() - start_time
    
    for eval_name, X_eval, y_eval in [
        ("UNSW_validation", X_val_processed, y_val_binary),
        ("CICIDS2018_sample", X_cic_processed, y_cic_binary)
    ]:
        y_eval_pred = sweep_model.predict(X_eval).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_eval,
            y_eval_pred,
            labels=[0, 1]
        ).ravel()
        
        c3b_cost_sweep_rows.append({
            "scale_pos_weight": cost_weight,
            "evaluation_data": eval_name,
            "precision": precision_score(y_eval, y_eval_pred, zero_division=0),
            "recall": recall_score(y_eval, y_eval_pred, zero_division=0),
            "f1": f1_score(y_eval, y_eval_pred, zero_division=0),
            "fpr": fp / (fp + tn) if (fp + tn) else 0,
            "fnr": fn / (fn + tp) if (fn + tp) else 0,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "training_time_seconds": sweep_train_time
        })

c3b_cost_sweep_df = pd.DataFrame(c3b_cost_sweep_rows)
display(c3b_cost_sweep_df.round(4))

c3b_cic_cost_sweep_df = c3b_cost_sweep_df[
    c3b_cost_sweep_df["evaluation_data"] == "CICIDS2018_sample"
].copy()

plt.figure(figsize=(7, 5))
plt.plot(
    c3b_cic_cost_sweep_df["fpr"],
    c3b_cic_cost_sweep_df["fnr"],
    marker="o"
)

for _, row in c3b_cic_cost_sweep_df.iterrows():
    plt.annotate(
        str(row["scale_pos_weight"]),
        (row["fpr"], row["fnr"]),
        textcoords="offset points",
        xytext=(5, 5)
    )

plt.title("Cost Weight Trade-off on CICIDS2018")
plt.xlabel("False positive rate")
plt.ylabel("False negative rate")
plt.grid(True)
plt.show()

