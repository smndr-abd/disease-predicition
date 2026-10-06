# Disease Prediction with Decision Trees

Symptom-based disease classification using decision trees, with a **synthetic
dataset generator** and an optional **cross-dataset test** against a real dataset.

> Educational project. The synthetic disease-symptom profiles are illustrative
> assumptions, not clinical data. Do not use for real diagnosis.

## Setup (macOS / VS Code)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```
In VS Code: Cmd+Shift+P -> "Python: Select Interpreter" -> pick `.venv`.

## Files
| File | Purpose |
|------|---------|
| `make_dataset.py` | Synthetic generator (disease "recipes") + `align_to_real()` |
| `train.py` | Overfitting demo, CV tuning, evaluation, plots, forest comparison |
| `figures/` | Confusion matrix, tree, feature importances |

## Cross-dataset test
Download the Kaggle "Disease Prediction Using Machine Learning" `Training.csv`, then:
```bash
python train.py --real data/Training.csv
```
Check that symptom column names match `PROFILES` in `make_dataset.py`.

## What it demonstrates
- Gini/entropy splits, overfitting, pruning (`max_depth`, `min_samples_leaf`, `ccp_alpha`)
- Cross-validation, confusion matrix, precision/recall per class
- Domain shift: synthetic -> real

## Done by Samandar Abdujabbar
