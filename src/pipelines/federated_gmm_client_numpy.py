import sys

sys.path.append("..")

from logging import INFO
import logging
import pandas as pd

import flwr as fl
from flwr.common.logger import log
from sklearn.mixture import GaussianMixture
from sklearn.metrics import average_precision_score

from data.processing import impute, scale
import utils_gmm as utils


class GMMClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        train: pd.DataFrame,
        test: pd.DataFrame,
        training_columns_to_drop: list,
        random_state: int,
    ):
        self.model = None  # keep the Model objects that remain consistent across rounds
        self.config = None

        # Perform imputation
        train = impute(train)
        test = impute(test)
        
        # Add relative time column
        '''
        train = train.sort_values(by=['stay_id', 'time'])
        train['time_relative'] = train.groupby('stay_id').cumcount()
        test = test.sort_values(by=['stay_id', 'time'])
        test['time_relative'] = test.groupby('stay_id').cumcount()
        '''
        training_columns_to_drop.append("time")

        # Define the features and target
        X_train = train.drop(columns=training_columns_to_drop)
        self.y_train = train["label"]
        X_valid = test.drop(columns=training_columns_to_drop)
        self.y_valid = test["label"]

        # Perform scaling
        self.X_train, self.X_valid = scale(X_train, X_valid)

        # Get number of samples
        self.num_train = len(X_train)
        self.num_valid = len(X_valid)

        log(
            INFO,
            f"Client {cid} - X_train shape: {self.X_train.shape} - X_valid shape: {self.X_valid.shape}",
        )

        # Hyperparamters for training
        self.num_local_round = 1
        self.params = {
            "n_components": 3,
            "random_state": random_state,
        }
        self.__build_model__()
        self.model = utils.set_initial_params(self.model)

    def __build_model__(self):
        self.model = GaussianMixture()
        self.model.n_components = self.params["n_components"]
        self.model.random_state = self.params["random_state"]

    def get_parameters(self, config):  # type: ignore
        return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore
        self.model = utils.set_model_params(self.model, parameters)
        self.model.fit(self.X_train)
        return utils.get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        self.model = utils.set_model_params(self.model, parameters)
        if not hasattr(self.model, "precisions_cholesky_"):
            self.model.fit(self.X_train)
        y_pred = self.model.predict_proba(self.X_valid)
        auprc = average_precision_score(self.y_valid, y_pred)

        log(INFO, f"AUPRC: {auprc}")

        return 0.0, len(self.X_valid), {"AUPRC": auprc}
