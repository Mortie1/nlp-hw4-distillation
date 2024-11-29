import numpy as np
from transformers import EvalPrediction
from seqeval.metrics import f1_score

from src.definitions import id2label, label2id

def f1(eval_pred: EvalPrediction):
    predictions = eval_pred.predictions
    predictions = np.argmax(predictions, axis=2)

    labels = eval_pred.label_ids

    text_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    text_labels = [[id2label[l] for l in label if l != -100] for label in labels]

    return {"f1": f1_score(y_pred=text_predictions, y_true=text_labels)}