"""
This is a boilerplate pipeline 'submission'
generated using Kedro 0.17.7
"""
from functools import partial


from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from plant_pathology_2020_fgvc7.extras.pipelines.nodes import predict
from .nodes import make_submission_from_outputs, ensemble_predictions


def _create_prediction_pipeline(val_fold: int) -> Pipeline:
    tags = "dev" if val_fold == 0 else None
    return pipeline(
        [
            node(
                func=partial(predict, target="test"),
                inputs=[
                    f"datamodule_fold_{val_fold}",
                    f"classifier_fold_{val_fold}",
                    "params:gpus",
                ],
                outputs=f"classifier_test_output_fold_{val_fold}",
                name=f"predict_test_fold_{val_fold}",
            ),
        ],
        tags=tags,
    )


def create_pipeline(**kwargs) -> Pipeline:
    n_folds = kwargs.get("n_folds", 5)
    prediction_pipelines = [_create_prediction_pipeline(val_fold=i) for i in range(5)]
    ensemble_pipeline = pipeline(
        pipe=[
            node(
                func=ensemble_predictions,
                inputs=[
                    f"classifier_test_output_fold_{fold}" for fold in range(n_folds)
                ],
                outputs="ensemble_test_output",
                name="ensemble_predictions",
            ),
            node(
                func=make_submission_from_outputs,
                inputs=["ensemble_test_output"],
                outputs="submission",
                name="make_submission",
            ),
        ],
    )

    return pipeline(
        pipe=[
            *prediction_pipelines,
            ensemble_pipeline,
            node(
                func=make_submission_from_outputs,
                inputs=["classifier_test_output_fold_0"],
                outputs="submission_fold_0",
                name="make_submission_fold_0",
                tags="dev"
            ),
        ],
        inputs=[
            *[f"classifier_fold_{i}" for i in range(n_folds)],
            *[f"datamodule_fold_{i}" for i in range(n_folds)],
        ],
        outputs=["submission"],
        namespace="submission_pipeline",
    )
