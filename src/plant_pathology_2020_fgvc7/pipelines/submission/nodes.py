"""
This is a boilerplate pipeline 'submission'
generated using Kedro 0.17.7
"""

import pandas as pd
import torch
import torch.nn.functional as F


def ensemble_predictions(*outputs):
    logits = torch.mean(torch.stack([o["logits"] for o in outputs]), axis=0)
    return {
        "image_ids": outputs[0]["image_ids"],
        "logits": logits,
    }

def make_submission_from_outputs(outputs):
    prob = F.softmax(outputs["logits"]).numpy()
    return pd.DataFrame({
        "image_id": outputs["image_ids"],
        "healthy": prob[:, 0],
        "multiple_diseases": prob[:, 1],
        "rust": prob[:, 2],
        "scab": prob[:, 3],
    })