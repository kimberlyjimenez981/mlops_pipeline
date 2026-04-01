# MLOps Pipeline

A complete MLOps pipeline with experiment tracking, testing, CI/CD, and drift monitoring.

## Project Structure

```
mlops_pipeline/
├── src/                          # Source code
│   ├── preprocessing.py         # Data preprocessing functions
│   └── train.py                 # Training script
├── configs/                      # Configuration files
│   └── config.yaml              # Training configuration
├── tests/                        # Test suite
│   ├── test_preprocessing.py    # Preprocessing unit tests
│   ├── test_dataset.py          # Data validation tests
│   └── test_model.py            # Model validation tests
├── .github/workflows/            # CI/CD pipelines
│   └── ci-cd.yml                # GitHub Actions workflow
├── data/                         # Data directory (tracked with DVC)
├── models/                       # Trained models
├── reports/                      # Analysis reports
├── compare_experiments.py        # MLflow experiment comparison
├── monitor_drift.py              # Data drift detection
├── MONITORING.md                 # Drift analysis documentation
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Setup

1. **Create Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize DVC**
   ```bash
   dvc init
   ```

## Training

Run training with configuration:

```bash
python src/train.py
```

Configuration is read from `configs/config.yaml` (no hardcoded values).

## Experiment Tracking

View all experiments in MLflow:

```bash
mlflow ui
```

Compare experiments:

```bash
python compare_experiments.py
```

## Testing

Run all tests:

```bash
pytest tests/ -v
```

## Data Drift Monitoring

Check for data drift:

```bash
python monitor_drift.py
```

Generates HTML report in `reports/` and exits with code 1 if drift exceeds threshold.

## CI/CD Pipeline

Automated testing and training on push to main branch:

- Tests run first
- Training runs only if tests pass
- Progress visible in GitHub Actions tab
