from pathlib import Path

import torch
import pandas as pd
import flash

def _get_dataloader(datamodule, target):
    if target == "train":
        return datamodule.train_dataloader()
    elif target == "val":
        return datamodule.val_dataloader()
    elif target == "test":
        return datamodule.predict_dataloader()

    raise ValueError(f"target must be one of 'train', 'val', 'test': {target}")


def predict(datamodule, model, gpus, target):
    trainer = flash.Trainer(gpus=gpus)

    outputs = trainer.predict(model, dataloaders=_get_dataloader(datamodule, target))
    outputs = sum(outputs, [])

    image_ids = [Path(o["metadata"]["filepath"]).stem for o in outputs]
    preds = torch.stack([o["preds"] for o in outputs])
    return {
        "image_ids": image_ids,
        "logits": preds,
    }
