"""
This is a boilerplate pipeline 'data_preprocess'
generated using Kedro 0.17.7
"""
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from flash.image import ImageClassificationData


def add_fold_column(df, n_folds):
    """
    Adds a fold column to a dataframe
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    target = df[["healthy", "multiple_diseases", "rust", "scab"]].idxmax(1)
    df["fold"] = 0
    for i, (_, index) in enumerate(skf.split(df, target)):
        df.iloc[index, df.columns.get_loc("fold")] = i
    return df

def load_datamodule(train_all_df, test_df, batch_size, data_root_dir, val_fold):
    train_df = train_all_df[train_all_df["fold"] != val_fold]
    val_df = train_all_df[train_all_df["fold"] == val_fold]

    images_root = str(Path(data_root_dir) / "plant-pathology-2020-fgvc7" / "images")
    filename_resolver = lambda root, _id: str(Path(root) / f"{_id}.jpg")

    datamodule = ImageClassificationData.from_data_frame(
        "image_id",
        ["healthy", "multiple_diseases", "rust", "scab"],
        train_data_frame=train_df,
        train_images_root=images_root,
        train_resolver=filename_resolver,
        val_data_frame=val_df,
        val_images_root=images_root,
        val_resolver=filename_resolver,
        test_data_frame=val_df,
        test_images_root=images_root,
        test_resolver=filename_resolver,
        predict_data_frame=test_df,
        predict_images_root=images_root,
        predict_resolver=filename_resolver,
        batch_size=batch_size,
    )

    return datamodule