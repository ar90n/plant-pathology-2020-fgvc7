"""
This is a boilerplate pipeline 'metrics'
generated using Kedro 0.17.7
"""

import torch
import torchmetrics


def calc_metrics(df, outputs):
    df = df.set_index("image_id")
    exp = torch.tensor(df.loc[outputs["image_ids"]].to_numpy())
    act = outputs["logits"].softmax(axis=1)
    metrics = torchmetrics.Accuracy()
    return {"accuracy": float(metrics(act, exp))}
