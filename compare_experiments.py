"""Compare MLflow experiments and find the best run."""

import mlflow
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def search_best_runs(experiment_name="iris_classification", metric_name="accuracy", limit=10):
    """Search for best runs in an experiment.
    
    Args:
        experiment_name: MLflow experiment name
        metric_name: Metric to sort by
        limit: Maximum number of runs to return
    
    Returns:
        runs_df: DataFrame with run information
    """
    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return None
        
        # Search runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=limit
        )
        
        if runs.empty:
            logger.warning(f"No runs found for experiment '{experiment_name}'")
            return None
        
        # Extract relevant columns
        cols_to_keep = [col for col in runs.columns if 
                       col.startswith('params.') or 
                       col.startswith('metrics.') or 
                       col == 'start_time' or
                       col == 'run_id']
        
        runs_summary = runs[cols_to_keep].copy()
        
        logger.info(f"Found {len(runs)} runs for experiment '{experiment_name}'")
        return runs_summary
    
    except Exception as e:
        logger.error(f"Error searching experiments: {str(e)}")
        return None


def compare_best_worst(experiment_name="iris_classification", metric_name="accuracy"):
    """Compare best and worst runs.
    
    Args:
        experiment_name: MLflow experiment name
        metric_name: Metric to compare
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.error(f"Experiment '{experiment_name}' not found")
            return
        
        # Get best run
        best_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1
        )
        
        # Get worst run
        worst_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} ASC"],
            max_results=1
        )
        
        if best_runs.empty or worst_runs.empty:
            logger.warning("Not enough runs to compare")
            return
        
        best_run = best_runs.iloc[0]
        worst_run = worst_runs.iloc[0]
        
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT COMPARISON")
        logger.info("="*60)
        
        logger.info(f"\nBEST RUN (ID: {best_run['run_id']})")
        logger.info(f"  {metric_name}: {best_run.get(f'metrics.{metric_name}', 'N/A')}")
        
        logger.info(f"\nWORST RUN (ID: {worst_run['run_id']})")
        logger.info(f"  {metric_name}: {worst_run.get(f'metrics.{metric_name}', 'N/A')}")
        
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error comparing runs: {str(e)}")


def main():
    """Main comparison function."""
    experiment_name = "iris_classification"
    
    logger.info(f"\nSearching for best runs in '{experiment_name}' experiment...\n")
    
    # Search for best runs
    runs_df = search_best_runs(experiment_name, metric_name="accuracy")
    
    if runs_df is not None and not runs_df.empty:
        logger.info("\nTop 5 Runs by Accuracy:")
        logger.info(runs_df.head(5).to_string())
        
        # Compare best and worst
        logger.info("\n")
        compare_best_worst(experiment_name, metric_name="accuracy")
    else:
        logger.warning("No runs found to compare")


if __name__ == "__main__":
    main()
