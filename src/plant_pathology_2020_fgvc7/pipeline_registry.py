"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from plant_pathology_2020_fgvc7.pipelines import data_preprocess, train_classifier, submission, metrics


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_process_pipeline = data_preprocess.create_pipeline()
    train_classifier_pipeline = train_classifier.create_pipeline()
    submission_pipeline = submission.create_pipeline()
    metrics_pipeline = metrics.create_pipeline()
    return {
        "__default__": train_classifier_pipeline + data_process_pipeline + submission_pipeline + metrics_pipeline,
    }
