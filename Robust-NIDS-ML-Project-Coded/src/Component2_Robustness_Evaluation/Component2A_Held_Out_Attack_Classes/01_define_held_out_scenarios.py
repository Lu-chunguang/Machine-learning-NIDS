"""
Component 2A - Define Held-Out Scenarios

Generated from: notebooks/5802pro2_jupy.ipynb
Notebook cells: C2A-1

Important:
    The original notebook is sequential. This file may depend on variables created by earlier files.
    Use `python src/main.py --section ...` from the project root to execute files in shared order.
"""

# ========================================================================================
# Notebook Cell C2A-1: Define held-out attack-class scenarios
# ========================================================================================

# Cell C2A-1: Define held-out attack-class scenarios

c2a_scenarios = {
    "A1_weak_classes": ["Analysis", "Backdoor", "DoS"],
    "A2_common_attack_classes": ["Exploits", "Fuzzers", "Generic"]
}

scenario_rows = []

for scenario_name, held_out_classes in c2a_scenarios.items():
    for attack_class in held_out_classes:
        scenario_rows.append({
            "scenario": scenario_name,
            "held_out_class": attack_class,
            "train_samples": int((y_train_text == attack_class).sum()),
            "validation_samples": int((y_val_text == attack_class).sum()),
            "test_samples": int((y_test_text == attack_class).sum())
        })

c2a_scenario_summary_df = pd.DataFrame(scenario_rows)

display(c2a_scenario_summary_df)

print("C2A held-out scenarios are ready.")

