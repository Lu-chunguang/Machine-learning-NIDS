
"""
Sequential runner for the refactored Robust NIDS project.

The original notebook uses shared variables across cells. To preserve that behavior,
this runner executes the split scripts with one shared global namespace.

Examples:
    python src/main.py --section c1
    python src/main.py --section c1+c2a+c3a
    python src/main.py --section all
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

SCRIPTS = {
    'c1': [
        'src/Component1_Baseline/01_initial_dataset_audit.py',
        'src/Component1_Baseline/02_data_quality_audit.py',
        'src/Component1_Baseline/03_exploratory_feature_analysis.py',
        'src/Component1_Baseline/04_baseline_preparation_and_split.py',
        'src/Component1_Baseline/05_baseline_feature_preprocessing.py',
        'src/Component1_Baseline/06_baseline_model_training.py',
        'src/Component1_Baseline/07_baseline_model_comparison.py',
        'src/Component1_Baseline/08_final_test_evaluation.py'
    ],
    'c2a': [
        'src/Component2_Robustness_Evaluation/Component2A_Held_Out_Attack_Classes/01_define_held_out_scenarios.py',
        'src/Component2_Robustness_Evaluation/Component2A_Held_Out_Attack_Classes/02_scenario_A1_weak_attack_unknown.py',
        'src/Component2_Robustness_Evaluation/Component2A_Held_Out_Attack_Classes/03_scenario_A2_common_attack_unknown.py',
        'src/Component2_Robustness_Evaluation/Component2A_Held_Out_Attack_Classes/04_full_open_set_mapping_matrix.py'
    ],
    'c3a': [
        'src/Component3_Robustness_Strategies/Component3A_Open_Set_Handling/01_confidence_thresholding.py',
        'src/Component3_Robustness_Strategies/Component3A_Open_Set_Handling/02_isolation_forest_fallback.py',
        'src/Component3_Robustness_Strategies/Component3A_Open_Set_Handling/03_open_set_ablation_study.py'
    ],
    'c2b': [
        'src/Component2_Robustness_Evaluation/Component2B_Cross_Dataset_Distribution_Shift/01_load_cicids2018_dataset.py',
        'src/Component2_Robustness_Evaluation/Component2B_Cross_Dataset_Distribution_Shift/02_column_compatibility_audit.py',
        'src/Component2_Robustness_Evaluation/Component2B_Cross_Dataset_Distribution_Shift/03_source_domain_preprocessing.py',
        'src/Component2_Robustness_Evaluation/Component2B_Cross_Dataset_Distribution_Shift/04_source_only_prediction.py',
        'src/Component2_Robustness_Evaluation/Component2B_Cross_Dataset_Distribution_Shift/05_binary_performance_evaluation.py',
        'src/Component2_Robustness_Evaluation/Component2B_Cross_Dataset_Distribution_Shift/06_per_attack_failure_analysis.py',
        'src/Component2_Robustness_Evaluation/Component2B_Cross_Dataset_Distribution_Shift/07_feature_distribution_shift_audit.py'
    ],
    'c3b': [
        'src/Component3_Robustness_Strategies/Component3B_Distribution_Shift_Improvement/01_cost_sensitive_binary_learning.py',
        'src/Component3_Robustness_Strategies/Component3B_Distribution_Shift_Improvement/02_cost_weight_tradeoff.py',
        'src/Component3_Robustness_Strategies/Component3B_Distribution_Shift_Improvement/03_few_shot_target_domain_split.py',
        'src/Component3_Robustness_Strategies/Component3B_Distribution_Shift_Improvement/04_few_shot_target_domain_training.py',
        'src/Component3_Robustness_Strategies/Component3B_Distribution_Shift_Improvement/05_few_shot_holdout_evaluation.py',
        'src/Component3_Robustness_Strategies/Component3B_Distribution_Shift_Improvement/06_strategy_comparison_and_ablation.py'
    ],
    'c2c': [
        'src/Component2_Robustness_Evaluation/Component2C_Feature_Degradation_Stress_Test/01_degradation_design_and_clean_baseline.py',
        'src/Component2_Robustness_Evaluation/Component2C_Feature_Degradation_Stress_Test/02_degradation_utilities.py',
        'src/Component2_Robustness_Evaluation/Component2C_Feature_Degradation_Stress_Test/03_baseline_degradation_tests.py',
        'src/Component2_Robustness_Evaluation/Component2C_Feature_Degradation_Stress_Test/04_feature_dropout_importance.py',
        'src/Component2_Robustness_Evaluation/Component2C_Feature_Degradation_Stress_Test/05_baseline_degradation_curves.py'
    ],
    'c3c': [
        'src/Component3_Robustness_Strategies/Component3C_Feature_Robustness_Augmentation/01_augmentation_design.py',
        'src/Component3_Robustness_Strategies/Component3C_Feature_Robustness_Augmentation/02_train_augmented_models.py',
        'src/Component3_Robustness_Strategies/Component3C_Feature_Robustness_Augmentation/03_clean_performance_check.py',
        'src/Component3_Robustness_Strategies/Component3C_Feature_Robustness_Augmentation/04_augmented_degradation_stress_test.py',
        'src/Component3_Robustness_Strategies/Component3C_Feature_Robustness_Augmentation/05_original_vs_augmented_comparison.py',
        'src/Component3_Robustness_Strategies/Component3C_Feature_Robustness_Augmentation/06_augmentation_variant_ablation.py',
        'src/Component3_Robustness_Strategies/Component3C_Feature_Robustness_Augmentation/07_degradation_curves.py'
    ],
}

SECTION_ALIASES = {
    "component1": ["c1"],
    "component2": ["c1", "c2a", "c2b", "c2c"],
    "component3": ["c1", "c2a", "c3a", "c2b", "c3b", "c2c", "c3c"],
    "open_set": ["c1", "c2a", "c3a"],
    "cross_dataset": ["c1", "c2b", "c3b"],
    "feature_robustness": ["c1", "c2c", "c3c"],
    "all": ["c1", "c2a", "c3a", "c2b", "c3b", "c2c", "c3c"],
}


def expand_section(section: str) -> list[str]:
    section = section.lower().strip()
    if "+" in section:
        parts = []
        for item in section.split("+"):
            parts.extend(expand_section(item))
        return parts
    if section in SECTION_ALIASES:
        return SECTION_ALIASES[section]
    if section in SCRIPTS:
        return [section]
    raise ValueError(f"Unknown section: {section}")


def execute_script(script_path: Path, namespace: dict) -> None:
    print(f"\n===== Running {script_path.relative_to(PROJECT_ROOT)} =====")
    namespace["__file__"] = str(script_path)
    namespace["__name__"] = "__main__"
    code = script_path.read_text(encoding="utf-8")
    compiled = compile(code, str(script_path), "exec")
    exec(compiled, namespace)


def run_sections(sections: Iterable[str]) -> None:
    namespace = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "__name__": "__main__",
    }
    executed = []
    for section in sections:
        for rel in SCRIPTS[section]:
            if rel in executed:
                continue
            execute_script(PROJECT_ROOT / rel, namespace)
            executed.append(rel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        default="all",
        help=(
            "Section to run: c1, c2a, c2b, c2c, c3a, c3b, c3c, "
            "open_set, cross_dataset, feature_robustness, component1, component2, component3, all."
        ),
    )
    args = parser.parse_args()
    sections = expand_section(args.section)
    print("Project root:", PROJECT_ROOT)
    print("Execution sections:", sections)
    run_sections(sections)


if __name__ == "__main__":
    main()
