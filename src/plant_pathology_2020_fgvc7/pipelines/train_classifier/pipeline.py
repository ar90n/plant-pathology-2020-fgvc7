"""
This is a boilerplate pipeline 'single_model'
generated using Kedro 0.17.7
"""

from functools import partial

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from plant_pathology_2020_fgvc7.extras.pipelines.nodes import predict
from .nodes import train_classifier, load_datamodule, merge_predictions


def _create_model_training_pipeline(val_fold: int) -> Pipeline:
    tags = "dev" if val_fold == 0 else None
    return pipeline(
        [
            node(
                func=train_classifier,
                inputs=[
                    f"datamodule_fold_{val_fold}",
                    "params:max_epochs",
                    "params:gpus",
                ],
                outputs=f"classifier_fold_{val_fold}",
                name=f"train_classifier_fold_{val_fold}",
            ),
            node(
                func=partial(predict, target="val"),
                inputs=[
                    f"datamodule_fold_{val_fold}",
                    f"classifier_fold_{val_fold}",
                    "params:gpus",
                ],
                outputs=f"classifier_val_output_fold_{val_fold}",
                name=f"predict_val_fold_{val_fold}",
            ),
        ],
        tags=tags,
    )


def create_pipeline(**kwargs) -> Pipeline:
    n_folds = kwargs.get("n_folds", 5)
    internal_pipelines = [
        _create_model_training_pipeline(val_fold=i) for i in range(n_folds)
    ]
    internal_pipelines.append(
        node(
            func=merge_predictions,
            inputs=[f"classifier_val_output_fold_{fold}" for fold in range(n_folds)],
            outputs="classifier_val_output",
            name="merge_predictions",
        )
    )

    return pipeline(
        pipe=internal_pipelines,
        inputs=[f"datamodule_fold_{i}" for i in range(n_folds)],
        outputs=[
            *[f"classifier_fold_{i}" for i in range(n_folds)],
            "classifier_val_output",
            "classifier_val_output_fold_0",
        ],
        namespace="classifier_training_pipeline",
    )
