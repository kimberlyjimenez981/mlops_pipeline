# Sprint 17 Updates

### Evaluation Module
- Introduced comprehensive model evaluation and validation functions in `src/evaluation.py`.

### DVC Setup
- Initialized DVC with the `.dvc/` directory and included pointer files: `data.dvc`, `models.dvc`.

### MLflow Experiments
- Added a new script `scripts/run_multiple_experiments.py` to execute 7 MLflow experiments.
- Documented all experiment runs and parameters in `EXPERIMENTS.md`.

### Other Updates
- Updated `src/train.py` to utilize the new evaluation module.
- Enhanced `README.md` with details regarding the evaluation module and experiment tracking.