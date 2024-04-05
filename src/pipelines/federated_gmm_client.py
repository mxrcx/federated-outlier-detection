"""
Module Name: federated_client
Description: This module defines a client class.

Classes & Methods:
- OCSVMClient: Defines a Flower client that will be used for FL simulation
    - fit: trains a model on the client's data using the provided parameters as a starting point
    - evalute: evaluates a model, whose parameters are set to the ones provided, on the client's data
"""

import sys

sys.path.append("..")

from logging import INFO
import pandas as pd
import json
import pickle

import flwr as fl
from flwr.common.logger import log
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
    GetParametersIns,
    GetParametersRes,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import average_precision_score

from data.processing import impute, scale


class GMMClient(fl.client.Client):
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
            "n_components": 1,
            "random_state": random_state,
        }

    def __build_model__(self):
        self.model = GaussianMixture()
        self.model.n_components = self.params["n_components"]
        self.model.random_state = self.params["random_state"]

    def __set_model_params__(self, params: Parameters):
        surrogate = None
        for item in params.tensors:
            surrogate = bytearray(item)

        # load the surrogate model into the booster
        self.model.set_params(surrogate)

    def get_parameters(self, ins: GetParametersIns = None) -> GetParametersRes:
        if self.model is not None:
            local_model = self.model.get_params()
            local_model_bytes = bytes(local_model)

            return GetParametersRes(
                status=Status(
                    code=Code.OK,
                    message="OK",
                ),
                parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            )
        else:
            return GetParametersRes(
                status=Status(
                    code=Code.OK,
                    message="No model to return",
                ),
                parameters=Parameters(tensor_type="", tensors=[]),
            )

    def fit(self, ins: FitIns) -> FitRes:
        """Trains a model on the client's data using the provided parameters as a starting point.

        Args:
            ins (FitIns): the training instructions containing (global) model parameters received from the server and a dictionary of configuration values


        Returns:
            FitRes: the training result containing updated parameters and other details such as the number of local training examples used for training
        """
        if self.model is None:
            self.__build_model__()
            self.model.fit(self.X_train)
        else:
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into model
            self.model.set_params(global_model)
            self.model.fit(self.X_train)

        local_model = self.model.get_params()
        print(local_model)

        #json_string = json.dumps(local_model)
        #local_model_bytes = json_string.encode("utf-8")
        #pickled_model = pickle.dumps(self.model)
        #local_model_bytes = bytes(local_model)
        local_model_bytes = pickle.dumps(self.model)
        log(
            INFO,
            f"lol {local_model_bytes}",
        )

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        if self.model is None:
            self.__build_model__()
        for para in ins.parameters.tensors:
            para_b = bytearray(para)
        self.model.set_params(para_b)

        y_pred = self.model.predict(self.X_valid)
        auprc = average_precision_score(self.y_valid, y_pred)

        log(INFO, f"AUPRC: {auprc}")

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_valid,
            metrics={"AUPRC": auprc},
        )
