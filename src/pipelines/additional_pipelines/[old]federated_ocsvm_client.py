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
import logging
import pandas as pd
import json
from typing import Tuple, Union, List
import numpy as np

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
from sklearn.svm import OneClassSVM
from sklearn.metrics import average_precision_score

from data.processing import impute, scale

# Configure the logger
file_handler = logging.FileHandler('app.log', delay=False)
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OCSVMClient(fl.client.Client):
    def __init__(
        self,
        cid: str,
        train: pd.DataFrame,
        test: pd.DataFrame,
        training_columns_to_drop: list,
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
        logger.info(f"Client {cid} - X_train shape: {self.X_train.shape} - X_valid shape: {self.X_valid.shape}")

        # Hyperparamters for training
        self.num_local_round = 1
        self.params = {
            "kernel": "linear",
            "nu": 0.01,
        }

    def __build_model__(self):
        self.model = OneClassSVM()
        self.model.kernel = self.params["kernel"]
        self.model.nu = self.params["nu"]

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
            
    def get_model_parameters(model: OneClassSVM) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray]]:
        """Returns the parameters of a scikit-learn OneClassSVM model."""
        params = [model.coef_, model.offset_]
        return params


    def set_model_params(model: OneClassSVM, params: Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray]]) -> OneClassSVM:
        """Sets the parameters of a scikit-learn OneClassSVM model."""
        model.coef_ = params[0]
        model.offset_ = params[1]
        return model

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
        json_string = json.dumps(local_model)
        local_model_bytes = json_string.encode('utf-8')
        #local_model_bytes = str(local_model).encode('utf-8')

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