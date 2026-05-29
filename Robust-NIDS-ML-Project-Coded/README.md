# Robust NIDS Machine Learning Project

This package refactors the original Jupyter Notebook report into a formal VS Code-style
machine learning project directory.

## Main Directory Structure

```text
Robust-NIDS-ML-Project-Coded/
├── data/
├── notebooks/
├── src/
│   ├── Component1_Baseline/
│   ├── Component2_Robustness_Evaluation/
│   │   ├── Component2A_Held_Out_Attack_Classes/
│   │   ├── Component2B_Cross_Dataset_Distribution_Shift/
│   │   └── Component2C_Feature_Degradation_Stress_Test/
│   ├── Component3_Robustness_Strategies/
│   │   ├── Component3A_Open_Set_Handling/
│   │   ├── Component3B_Distribution_Shift_Improvement/
│   │   └── Component3C_Feature_Robustness_Augmentation/
│   └── main.py
├── models/
├── results/
├── reports/
└── docs/
```

## What was changed from the notebook?

The notebook code cells were separated into Python files according to the original cell labels:

- `C1-*` → `Component1_Baseline`
- `C2A-*`, `C2B-*`, `C2C-*` → `Component2_Robustness_Evaluation`
- `C3A-*`, `C3B-*`, `C3C-*` → `Component3_Robustness_Strategies`

See `docs/CODE_SPLIT_MAP.md` for the complete mapping.

## How to run in VS Code

### 1. Open the project folder

Open `Robust-NIDS-ML-Project-Coded/` in VS Code.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add datasets

Put the required datasets into `data/raw/`:

```text
data/raw/NF-UNSW-NB15-v2.csv
data/raw/NF-CSE-CIC-IDS2018-v2.csv
```

You may also set environment variables instead:

```bash
set UNSW_PATH=C:\path\to\NF-UNSW-NB15-v2.csv
set CICIDS_PATH=C:\path\to\NF-CSE-CIC-IDS2018-v2.csv
```

On Mac/Linux:

```bash
export UNSW_PATH=/path/to/NF-UNSW-NB15-v2.csv
export CICIDS_PATH=/path/to/NF-CSE-CIC-IDS2018-v2.csv
```

### 5. Run the project

Run Component 1 only:

```bash
python src/main.py --section c1
```

Run open-set experiment path:

```bash
python src/main.py --section open_set
```

Run cross-dataset experiment path:

```bash
python src/main.py --section cross_dataset
```

Run feature robustness path:

```bash
python src/main.py --section feature_robustness
```

Run all notebook-equivalent scripts:

```bash
python src/main.py --section all
```

## Important Note

The original notebook is sequential and uses shared variables across cells. Therefore, the safest way
to run the refactored code is through `src/main.py`, which executes the split files in one shared
namespace. Running a later file directly may fail because it depends on variables created by earlier files.
