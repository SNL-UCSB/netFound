import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from modules.utils import get_logger

logger = get_logger(__name__)


def regression_metrics(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return {"loss": np.mean(np.absolute((logits - p.label_ids)))}


def classif_metrics(p: EvalPrediction, label_encoder: LabelEncoder):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    num_classes = len(label_encoder.classes_)
    labels = np.arange(num_classes)
    label_ids = p.label_ids.astype(int)

    # logits can be raw (N, num_classes) or pre-argmax'd (N,). Derive preds
    # and keep raw scores (for top-k) separate.
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        # Already argmax'd (or single-class edge case)
        preds = logits.reshape(-1).astype(int)
        raw_scores = None
    else:
        preds = logits.argmax(axis=1)
        raw_scores = logits  # needed for top_k_accuracy_score

    weighted_f1 = f1_score(
        y_true=label_ids, y_pred=preds, average="weighted", zero_division=0, labels=labels,
    )
    weighted_prec = precision_score(
        y_true=label_ids, y_pred=preds, average="weighted", zero_division=0, labels=labels,
    )
    weighted_recall = recall_score(
        y_true=label_ids, y_pred=preds, average="weighted", zero_division=0, labels=labels,
    )
    accuracy = accuracy_score(y_true=label_ids, y_pred=preds)
    logger.warning(classification_report(y_true=label_ids, y_pred=preds, digits=5, labels=labels, target_names=label_encoder.classes_))
    if raw_scores is not None:
        if num_classes > 3:
            logger.warning(f"top3:{top_k_accuracy_score(label_ids, raw_scores, k=3, labels=np.arange(num_classes))}")
        if num_classes > 5:
            logger.warning(f"top5:{top_k_accuracy_score(label_ids, raw_scores, k=5, labels=np.arange(num_classes))}")
        if num_classes > 10:
            logger.warning(f"top10:{top_k_accuracy_score(label_ids, raw_scores, k=10, labels=np.arange(num_classes))}")
    return {
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "weighted_prec": weighted_prec,
        "weighted_recall": weighted_recall,
    }


def preprocess_logits_for_metrics(logits, _):
    if isinstance(logits, tuple):
        return tuple(i.argmax(dim=-1) for i in logits)
    return logits.argmax(dim=-1)


def pretraining_metrics(eval_preds):
    all_preds, all_labels = eval_preds

    labels = all_labels[0] if isinstance(all_labels, tuple) else all_labels
    preds = all_preds[0] if isinstance(all_preds, tuple) else all_preds
    swappedBurstGTs = all_labels[1] if isinstance(all_labels, tuple) else None
    swappedBurstPreds = all_preds[1] if isinstance(all_preds, tuple) else None

    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return_metrics = {
        "macro_mlm_f1": f1_score(labels, preds, average="macro"),
        "macro_mlm_prec": precision_score(labels, preds, average="macro"),
        "macro_mlm_recall": recall_score(labels, preds, average="macro"),
        "weighted_mlm_f1": f1_score(labels, preds, average="weighted"),
        "weighted_mlm_prec": precision_score(labels, preds, average="weighted"),
        "weighted_mlm_recall": recall_score(labels, preds, average="weighted"),
        "mlm_acc": accuracy_score(labels, preds),
    }
    if swappedBurstGTs is not None and swappedBurstPreds is not None:
        return_metrics.update(
            {
                "swapped_macro_pred_f1": f1_score(swappedBurstGTs, swappedBurstPreds, average="macro"),
                "swapped_macro_pred_prec": precision_score(
                    swappedBurstGTs, swappedBurstPreds, average="macro"
                ),
                "swapped_macro_pred_recall": recall_score(
                    swappedBurstGTs, swappedBurstPreds, average="macro"
                ),
                "swapped_weighted_pred_f1": f1_score(
                    swappedBurstGTs, swappedBurstPreds, average="weighted"
                ),
                "swapped_weighted_pred_prec": precision_score(
                    swappedBurstGTs, swappedBurstPreds, average="weighted"
                ),
                "swapped_weighted_pred_recall": recall_score(
                    swappedBurstGTs, swappedBurstPreds, average="weighted"
                ),
                "swapped_pred_acc": accuracy_score(swappedBurstGTs, swappedBurstPreds),
            }
        )
    return return_metrics
