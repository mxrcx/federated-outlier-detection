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
    try:
        try:
            auroc = round(roc_auc_score(y_test, y_pred_proba[:, 1]), 4)
        except IndexError:
            auroc = round(roc_auc_score(y_test, np.zeros_like(y_pred_proba)), 4) # If all predictions are False(=0), no predictions are True(=1)
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
    try:
        try:
            auprc = round(average_precision_score(y_test, y_pred_proba[:, 1]), 4)
        except IndexError:
            auprc = round(average_precision_score(y_test, np.zeros_like(y_pred_proba)), 4) # If all predictions are False(=0), no predictions are True(=1) 
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
    if entries_to_consider is None:
        return sum(cm_list)  # Average across complete list
    else:
        return sum(cm_list[-entries_to_consider:])


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
        float: Averaged metric
    """
    try:
        if entries_to_consider is None:
            return round(
                sum(metric_list) / len(metric_list), 4
            )  # Average across complete list
        else:
            return round(sum(metric_list[-entries_to_consider:]) / entries_to_consider, 4)
    except Exception:
        return "Not defined"
        
