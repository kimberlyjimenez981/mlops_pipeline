# MLflow Experiments Documentation

All experiments use the `iris_classification` experiment name in MLflow and the Iris dataset from scikit-learn. Runs were executed via `scripts/run_multiple_experiments.py`.

---

## Experiment Runs

| # | Run Name | Run ID | n_estimators | max_depth | test_size | normalize | accuracy | precision | recall | F1 |
|---|----------|--------|--------------|-----------|-----------|-----------|----------|-----------|--------|----|
| 1 | exp_01_baseline | `67fea33da4ab4d04893f56153fc2d9e7` | 100 | 10 | 0.20 | True | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | exp_02_shallow_tree | `39b78dc32fc9439b8e30c4285d688378` | 100 | 3 | 0.20 | True | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3 | exp_03_more_estimators | `1b75d43f4e3846dbadad24de32268cae` | 200 | 10 | 0.20 | True | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | exp_04_large_test_split | `73a22a5fc1c94393af84d5cbbfec62b9` | 100 | 10 | 0.30 | True | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 5 | exp_05_no_normalize | `1a625cbeeef346589e69e7a57aba298f` | 100 | 10 | 0.20 | False | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 6 | exp_06_few_estimators | `b726953c47bb41b896e28d8fe0312a05` | 50 | 5 | 0.25 | True | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 7 | exp_07_deep_tree | `1d9fc11f6338447f8134d90b2111dfab` | 150 | None (unlimited) | 0.20 | True | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## Cross-Validation Results (5-fold Stratified)

| # | Run Name | CV Accuracy (mean ± std) | CV F1 (mean ± std) | CV Precision (mean ± std) | CV Recall (mean ± std) |
|---|----------|--------------------------|--------------------|---------------------------|------------------------|
| 1 | exp_01_baseline | 0.9467 ± 0.0267 | 0.9464 ± 0.0268 | 0.9512 ± 0.0263 | 0.9467 ± 0.0267 |
| 2 | exp_02_shallow_tree | 0.9667 ± 0.0298 | 0.9665 ± 0.0300 | 0.9695 ± 0.0276 | 0.9667 ± 0.0298 |
| 3 | exp_03_more_estimators | 0.9600 ± 0.0389 | 0.9598 ± 0.0390 | 0.9633 ± 0.0369 | 0.9600 ± 0.0389 |
| 4 | exp_04_large_test_split | 0.9467 ± 0.0267 | 0.9464 ± 0.0268 | 0.9512 ± 0.0263 | 0.9467 ± 0.0267 |
| 5 | exp_05_no_normalize | 0.9467 ± 0.0267 | 0.9464 ± 0.0268 | 0.9512 ± 0.0263 | 0.9467 ± 0.0267 |
| 6 | exp_06_few_estimators | 0.9533 ± 0.0340 | 0.9531 ± 0.0341 | 0.9572 ± 0.0326 | 0.9533 ± 0.0340 |
| 7 | exp_07_deep_tree | 0.9467 ± 0.0267 | 0.9464 ± 0.0268 | 0.9512 ± 0.0263 | 0.9467 ± 0.0267 |

---

## Parameter Variations

Three parameters were varied across runs:

1. **`n_estimators`**: 50, 100, 150, 200 — controls the number of trees in the forest.
2. **`max_depth`**: 3, 5, 10, `None` (unlimited) — controls tree depth and potential overfitting.
3. **`test_size`**: 0.20, 0.25, 0.30 — controls the train/test split ratio.
4. **`normalize`**: `True` / `False` — controls whether features are StandardScaler-normalized before training.

---

## Notes

- **Date executed**: 2026-04-06
- **Dataset**: Iris (150 samples, 4 features, 3 classes) from `sklearn.datasets`
- **MLflow experiment name**: `iris_classification`
- **Reproducibility**: All runs use `random_state=42`; re-run `scripts/run_multiple_experiments.py` to reproduce.
- **Observation**: The Iris dataset is well-separated; all configurations achieve perfect test-set accuracy. Cross-validation scores reveal subtle differences between configurations, with the shallow tree (depth=3) achieving the highest mean CV accuracy (0.9667).
