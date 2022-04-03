"""
This is a boilerplate pipeline 'single_model'
generated using Kedro 0.17.7
"""
from pathlib import Path

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import flash
from flash.image import ImageClassifier
from flash.image import ImageClassificationData


def merge_predictions(*predictions):
    image_ids = sum([p["image_ids"] for p in predictions], [])
    logits = torch.vstack([p["logits"] for p in predictions])
    return {
        "image_ids": image_ids,
        "logits": logits,
    }


def train_classifier(datamodule, max_epochs, gpus):
    labels = ["healthy", "multiple_diseases", "rust", "scab"]
    model = ImageClassifier(backbone="resnet18", labels=labels, pretrained=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_accuracy",
        mode="max",
    )
    trainer = flash.Trainer(
        max_epochs=max_epochs, gpus=gpus, callbacks=[checkpoint_callback]
    )
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    best_model = ImageClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    return best_model


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
