import numpy as np
from sklearn.linear_model import SGDOneClassSVM
from sklearn.utils.extmath import safe_sparse_dot

from flwr.common import NDArrays


def get_model_parameters(model: SGDOneClassSVM) -> NDArrays:
    """Returns the parameters of a sklearn SGDOneClassSVM model."""
    params = [
        model.coef_,
        model.offset_,
    ]
    return params


def set_model_params(model: SGDOneClassSVM, params: NDArrays) -> SGDOneClassSVM:
    """Sets the parameters of a sklean SGDOneClassSVM model."""
    model.coef_ = params[0]
    model.offset_ = params[1]
    return model


def set_initial_params(model: SGDOneClassSVM) -> SGDOneClassSVM:
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.SGDOneClassSVM documentation for more information.
    """
    n_features = 117  # Number of features in dataset
    model.coef_ = np.zeros((1, n_features))
    model.offset_ = np.zeros((1,))

    return model
