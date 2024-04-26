import numpy as np
from sklearn.mixture import GaussianMixture

from flwr.common import NDArrays


def get_model_parameters(model: GaussianMixture) -> NDArrays:
    """Returns the parameters of a sklearn GaussianMixture model."""
    params = [
        model.weights_,
        model.means_,
        model.covariances_,
        model.precisions_,
        model.precisions_cholesky_,
    ]
    return params


def set_model_params(model: GaussianMixture, params: NDArrays) -> GaussianMixture:
    """Sets the parameters of a sklean GaussianMixture model."""
    model.weights_ = params[0]
    model.means_ = params[1]
    model.covariances_ = params[2]
    model.precisions_ = params[3]
    model.precisions_cholesky_ = params[4]
    return model


def set_initial_params(model: GaussianMixture) -> GaussianMixture:
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.GaussianMixture documentation for more information.
    """
    n_features = 192  # Number of features in dataset
    n_components = 4  # Number of components in GMM

    model.weights_ = np.zeros((n_components,))
    model.means_ = np.zeros((n_components, n_features))

    # if spherical instead of full, then covariances_ is 1D array
    # model.covariances_ = np.zeros((n_components,))
    # model.precisions_ = np.zeros((n_components,))
    # model.precisions_cholesky_ = np.zeros((n_components,))

    # if full instead of spherical, then covariances_ is 3D array
    model.covariances_ = np.zeros((n_components, n_features, n_features))
    model.precisions_ = np.zeros((n_components, n_features, n_features))
    model.precisions_cholesky_ = np.zeros((n_components, n_features, n_features))

    return model
