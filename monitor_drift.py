"""Data drift monitoring using Evidently."""

import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from evidently.report import Report
from evidently.metrics import DataDriftTable, ColumnDriftMetric
from src.preprocessing import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Drift detection threshold (0-1, where 1 means no drift)
DRIFT_THRESHOLD = 0.95
FEATURE_DRIFT_THRESHOLD = 0.95


def create_reference_data(test_size=0.3, random_state=42):
    """Create reference data from current dataset.
    
    Args:
        test_size: Proportion to use as reference
        random_state: Random seed
    
    Returns:
        reference_df: Reference dataset
    """
    from sklearn.model_selection import train_test_split
    
    X, y = load_data()
    
    # Use first part as reference
    n_reference = int(len(X) * (1 - test_size))
    reference_df = X.iloc[:n_reference].copy()
    reference_df['target'] = y.iloc[:n_reference].values
    
    logger.info(f"Created reference data with {len(reference_df)} samples")
    return reference_df


def create_production_data(test_size=0.3, random_state=42):
    """Create simulated production data (with some drift).
    
    Args:
        test_size: Proportion to use as production
        random_state: Random seed
    
    Returns:
        production_df: Production dataset
    """
    np.random.seed(random_state)
    X, y = load_data()
    
    # Use second part as production (simulating drift)
    n_reference = int(len(X) * (1 - test_size))
    production_df = X.iloc[n_reference:].copy()
    
    # Add slight drift to features (increase values slightly)
    production_df = production_df.astype(float)
    drift_magnitude = 0.15  # 15% increase
    for col in production_df.columns:
        production_df[col] = production_df[col] * (1 + drift_magnitude * np.random.uniform(-0.5, 0.5, len(production_df)))
    
    production_df['target'] = y.iloc[n_reference:].values
    
    logger.info(f"Created production data with {len(production_df)} samples (with drift)")
    return production_df


def detect_drift(reference_df, production_df, threshold=DRIFT_THRESHOLD):
    """Detect data drift using Evidently.
    
    Args:
        reference_df: Reference dataset
        production_df: Production dataset
        threshold: Drift threshold (0-1)
    
    Returns:
        drift_detected: Boolean indicating if drift was detected
        drifted_features: List of features with detected drift
        drift_share: Overall drift share
    """
    logger.info("Detecting data drift...")
    
    # Create Evidently report
    report = Report(metrics=[
        DataDriftTable(),
    ])
    
    report.run(reference_data=reference_df, current_data=production_df)
    
    # Extract drift information
    drift_data = report.as_dict()
    metrics = drift_data.get('metrics', [])
    
    drifted_features = []
    drift_share = 0
    
    # Parse drift metrics
    for metric in metrics:
        if 'result' in metric:
            result = metric['result']
            if isinstance(result, dict) and 'drift_by_columns' in result:
                drift_by_cols = result['drift_by_columns']
                drift_share = result.get('dataset_drift', 0)
                
                # Check individual features
                for feature, drift_info in drift_by_cols.items():
                    if isinstance(drift_info, dict):
                        is_drifted = drift_info.get('drift_detected', False)
                        if is_drifted:
                            drifted_features.append({
                                'feature': feature,
                                'drift_detected': is_drifted,
                                'statistic': drift_info.get('statistic_name', 'N/A'),
                                'statistic_value': drift_info.get('statistic_value', 0),
                                'threshold': drift_info.get('threshold', 0)
                            })
    
    drift_detected = drift_share < threshold
    
    logger.info(f"Drift share: {drift_share:.4f}")
    logger.info(f"Features with drift detected: {len(drifted_features)}")
    
    return drift_detected, drifted_features, drift_share


def save_drift_report(reference_df, production_df, output_path="reports/drift_report.html"):
    """Save drift report as HTML.
    
    Args:
        reference_df: Reference dataset
        production_df: Production dataset
        output_path: Path to save HTML report
    """
    import os
    
    logger.info(f"Generating drift report...")
    
    # Create report
    report = Report(metrics=[
        DataDriftTable(),
    ])
    
    report.run(reference_data=reference_df, current_data=production_df)
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.save_html(output_path)
    
    logger.info(f"Drift report saved to {output_path}")


def main():
    """Main drift monitoring pipeline."""
    try:
        # Create reference and production data
        reference_df = create_reference_data()
        production_df = create_production_data()
        
        # Detect drift
        drift_detected, drifted_features, drift_share = detect_drift(reference_df, production_df)
        
        # Log results
        logger.info("\n" + "="*60)
        logger.info("DRIFT DETECTION REPORT")
        logger.info("="*60)
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Reference samples: {len(reference_df)}")
        logger.info(f"Production samples: {len(production_df)}")
        logger.info(f"Drift share: {drift_share:.4f}")
        logger.info(f"Drift detected: {drift_detected}")
        
        if drifted_features:
            logger.info("\nDrifted Features:")
            for feature_info in drifted_features:
                feature = feature_info['feature']
                stat_name = feature_info['statistic']
                stat_val = feature_info['statistic_value']
                logger.info(f"  - {feature}: {stat_name} = {stat_val:.4f}")
        
        logger.info("="*60 + "\n")
        
        # Save HTML report
        save_drift_report(reference_df, production_df)
        
        # Exit with code 1 if drift exceeds threshold
        if drift_detected and drift_share < DRIFT_THRESHOLD:
            logger.warning("⚠️  Data drift detected! Action required.")
            sys.exit(1)
        else:
            logger.info("✓ No significant data drift detected.")
            sys.exit(0)
    
    except Exception as e:
        logger.error(f"Error in drift monitoring: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
