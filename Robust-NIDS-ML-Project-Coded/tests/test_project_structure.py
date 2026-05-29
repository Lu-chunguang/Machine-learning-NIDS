from pathlib import Path

def test_required_directories_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "Component1_Baseline").exists()
    assert (root / "src" / "Component2_Robustness_Evaluation").exists()
    assert (root / "src" / "Component3_Robustness_Strategies").exists()
    assert (root / "notebooks" / "5802pro2_jupy.ipynb").exists()
