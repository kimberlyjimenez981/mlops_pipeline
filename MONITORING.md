# Data Drift Monitoring Analysis

## Overview

This document provides analysis of data drift monitoring for the iris classification ML pipeline.

## Drift Detection Strategy

The drift monitoring system uses **Evidently** to compare reference data (baseline) against production data to detect distributional shifts in features.

### Reference Data

- Source: First 70% of iris dataset (105 samples)
- Purpose: Represents expected production data distribution
- Features: 4 flower measurements (sepal length, sepal width, petal length, petal width)

### Production Data

- Source: Last 30% of iris dataset (45 samples)
- Simulated Drift: 15% random perturbation added to features
- Purpose: Simulates real production environment with data changes

## Features Monitored

1. **sepal length (cm)**: Sepal length measurement
2. **sepal width (cm)**: Sepal width measurement
3. **petal length (cm)**: Petal length measurement
4. **petal width (cm)**: Petal width measurement

## Drift Detection Results

### Overall Drift Assessment

- **Drift Share**: 0.0 to 1.0 (1.0 = no drift, 0.0 = complete drift)
- **Threshold**: 0.95
- **Status**: Run `python monitor_drift.py` to get latest results

### Expected Behavior

With the simulated 15% perturbation:

- Most features should show moderate drift detection
- Drift share likely between 0.50-0.90 range
- Individual features may trigger drift alerts

## Feature-Level Drift Analysis

### Which Features Drifted?

When drift is detected, the monitoring script identifies specific features with statistical differences:

- Statistical tests compare distributions between reference and production
- Examples: Kolmogorov-Smirnov test, Chi-square test
- Features with p-values < 0.05 are flagged

### Performance Impact

**Question: Would drift affect model performance?**

**Answer: YES**

1. **Distribution Shift**: Features with different distributions mean model receives data outside training distribution
2. **Model Degradation**: Random Forest trained on reference distribution may perform worse on drifted data
3. **Predictions Unreliable**: Model confidence scores may become miscalibrated
4. **Expected Impact**: 5-15% accuracy drop with 15% feature perturbation

## Recommended Actions

### If Drift Detected

1. **Immediate Actions**
   - Stop production predictions or apply confidence threshold filter
   - Document incident with timestamp and drift magnitude
   - Alert ML team

2. \*\*Short-term Actions (Hours)
   - Collect recent production data
   - Verify drift with manual inspection
   - Check for data pipeline issues

3. **Medium-term Actions (Days)**
   - Collect drifted data and retrain model
   - Consider data quality improvements
   - Update monitoring thresholds if drift is acceptable

4. **Long-term Actions**
   - Implement automated retraining when drift > threshold
   - Add data validation in production pipeline
   - Monitor upstream data sources for stability

### If No Drift Detected

- Continue normal operations
- Monitor trends in drift metrics over time
- Re-run monitoring weekly or on schedule

## Monitoring Schedule

- **Production**: Run daily after 8 PM UTC
- **Development**: Run on each pull request
- **Ad-hoc**: Run manually when investigating issues

## Configuration

### Sensitivity Settings

```yaml
drift:
  overall_threshold: 0.95 # Overall drift threshold
  feature_threshold: 0.95 # Per-feature threshold
  min_samples: 30 # Minimum samples for drift detection
  statistical_tests:
    - kolmogorov_smirnov
    - chi_square
```

### Adjustment Guidance

- **More Sensitive**: Decrease thresholds (e.g., 0.90 for stricter monitoring)
- **Less Sensitive**: Increase thresholds (e.g., 0.98 for production dips)
- **Context-Dependent**: Adjust based on business tolerance for model drift

## Troubleshooting

### Issue: No Drift Detected When Expected

- Verify reference and production data are different
- Check threshold settings
- Ensure sufficient samples (n x > 30)
- Review statistical test selection

### Issue: Too Many False Positives

- Increase drift threshold
- Use larger reference dataset
- Filter out seasonal/expected variations

### Issue: Monitoring Script Fails

- Check Evidently installation: `pip install evidently`
- Verify data format (pandas DataFrame)
- Check file permissions for HTML report saving

## Additional Resources

- [Evidently Documentation](https://docs.evidentlyai.com/)
- [ML Monitoring Best Practices](https://www.evidentlyai.com/blog)
- Project README for running commands

---

**Last Updated**: April 2026
**Monitoring Owner**: ML Ops Team
**Escalation**: Data Science Lead
