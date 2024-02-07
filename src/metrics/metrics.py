from typing import List, Optional
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def get_accuracy(y_test, y_pred):
    """
    Get accuracy.

    Args:
        y_test: True labels
        y_pred: Predicted labels

    Returns:
        float: Accuracy
    """
    return round(accuracy_score(y_test, y_pred), 4)


def get_auroc(y_test, y_pred_proba):
    """
    Get area under receiver operating characteristic curve.

    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        float: Area under receiver operating characteristic curve
    """
    # Get the probability of the positive class
    try:
        y_score = y_pred_proba[:, 1]
    except IndexError:
        # If all predictions are False(=0), no predictions are True(=1)
        y_score = np.zeros_like(y_pred_proba)

    # Calculate area under receiver operating characteristic curve
    try:
        auroc = round(roc_auc_score(y_test, y_score), 4)
    except ValueError:
        auroc = "Not defined"

    return auroc


def get_auprc(y_test, y_pred_proba):
    """
    Get area under precision-recall curve.

    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        float: Area under precision-recall curve
    """
    # Get the probability of the positive class
    try:
        y_score = y_pred_proba[:, 1]
    except IndexError:
        # If all predictions are False(=0), no predictions are True(=1)
        y_score = np.zeros_like(y_pred_proba)

    # Calculate average precision
    try:
        auprc = round(average_precision_score(y_test, y_score), 4)
    except ValueError:
        auprc = "Not defined"

    return auprc


def get_confusion_matrix(y_test, y_pred):
    """
    Get confusion matrix.

    Args:
        y_test: True labels
        y_pred: Predicted labels

    Returns:
        np.ndarray: Confusion matrix
    """
    return confusion_matrix(y_test, y_pred)


def get_average_confusion_matrix(
    cm_list: List[np.ndarray], entries_to_consider: Optional[int] = None
) -> np.ndarray:
    """
    Average confusion matrices in list. If entries_to_consider is not None, only the last entries_to_consider
    matrices are considered.

    Args:
        cm_list: List of confusion matrices
        entries_to_consider: Number of last entries to consider

    Returns:
        np.ndarray: Averaged confusion matrix
    """
    # If specified: Only consider last entries_to_consider elements
    if entries_to_consider is not None:
        cm_list = cm_list[-entries_to_consider:]

    return sum(cm_list)


def get_average_metric(
    metric_list: List[float], entries_to_consider: Optional[int] = None
) -> float:
    """
    Average metric in list. If entries_to_consider is not None, only the last entries_to_consider
    metrics are considered.

    Args:
        metric_list: List of metrics
        entries_to_consider: Number of last entries to consider

    Returns:
        float: Averaged metric or "Not defined" if an exception occurs
    """
    # If specified: Only consider last entries_to_consider elements
    if entries_to_consider is not None:
        metric_list = metric_list[-entries_to_consider:]

    # Remove "Not defined" elements
    metric_list = [metric for metric in metric_list if metric != "Not defined"]

    try:
        # Calculate average
        return round(sum(metric_list) / len(metric_list), 4)
    except Exception:
        return "Avg calculation not possible."
