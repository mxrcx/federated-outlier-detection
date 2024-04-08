import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.utils.extmath import safe_sparse_dot

from flwr.common import NDArrays


def get_model_parameters(model: OneClassSVM) -> NDArrays:
    """Returns the parameters of a sklearn OneClassSVM model."""
    params = [
        safe_sparse_dot(model.dual_coef_, model.support_vectors_),
        model.offset_,
        model.support_vectors_,
    ]
    return params


def set_model_params(model: OneClassSVM, params: NDArrays) -> OneClassSVM:
    """Sets the parameters of a sklean OneClassSVM model."""
    model.dual_coef_ = params[0]
    model.offset_ = params[1]
    model.support_vectors_ = params[2]
    return model


def set_initial_params(model: OneClassSVM) -> OneClassSVM:
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.OneClassSVM documentation for more information.
    """
    n_features = 117  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])
    model.dual_coef_ = np.zeros((1, n_features))
    model.support_vectors_ = np.zeros((n_features, n_features))
    model.offset_ = 0.0

    return model
