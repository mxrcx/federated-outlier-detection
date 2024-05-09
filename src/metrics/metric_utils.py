import numpy as np

def get_y_score(y_pred_proba):
    """
    Get the probability of the positive class (y_score).

    Args:
        y_pred_proba: Predicted probabilities

    Returns:
        np.ndarray: Probability of the positive class
    """
    try:
        y_score = y_pred_proba[:, 1]
    except IndexError:
        # If all predictions are False(=0), no predictions are True(=1)
        y_score = np.zeros_like(y_pred_proba)
    return y_score