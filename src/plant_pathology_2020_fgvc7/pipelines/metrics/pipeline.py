"""
This is a boilerplate pipeline 'metrics'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import calc_metrics

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=calc_metrics,
            inputs=["train_csv", "classifier_val_output"],
            outputs="metrics",
            name="calc_metrics",
        ),
        node(
            func=calc_metrics,
            inputs=["train_csv", "classifier_val_output_fold_0"],
            outputs="metrics_dev",
            name="calc_metrics_dev",
            tags="dev"
        ),
    ])
