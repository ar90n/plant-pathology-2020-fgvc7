"""
This is a boilerplate pipeline 'data_preprocess'
generated using Kedro 0.17.7
"""

from functools import partial

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import add_fold_column, load_datamodule


def _load_datamodule_node(val_fold: int) -> Pipeline:
    tags = "dev" if val_fold == 0 else None
    return node(
        func=partial(load_datamodule, val_fold=val_fold),
        inputs=[
            "train_with_fold",
            "test_csv",
            "params:batch_size",
            "params:data.raw",
        ],
        outputs=f"datamodule_fold_{val_fold}",
        name=f"load_datamodule_fold_{val_fold}",
        tags=tags,
    )


def create_pipeline(**kwargs) -> Pipeline:
    n_folds = kwargs.get("n_folds", 5)
    return pipeline(
        pipe=[
            node(
                func=partial(add_fold_column, n_folds=n_folds),
                inputs=["train_csv"],
                outputs="train_with_fold",
                name="add_fold_column",
                tags=["dev"],
            ),
            *[_load_datamodule_node(val_fold=i) for i in range(n_folds)],
        ],
        inputs=["train_csv", "test_csv"],
        outputs=[f"datamodule_fold_{i}" for i in range(n_folds)],
        namespace="data_preprocess",
    )
