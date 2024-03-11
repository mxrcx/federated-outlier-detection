"""
Module Name: federated_client
Description: This module defines a client class.

Classes & Methods:
- XGBClient: Defines a Flower client that will be used for FL simulation
    - fit: trains a model on the client's data using the provided parameters as a starting point
    - evalute: evaluates a model, whose parameters are set to the ones provided, on the client's data
"""

import sys

sys.path.append("..")

from logging import INFO
import pandas as pd

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
import xgboost as xgb

from data.loading import load_configuration
from data.processing import init_imputer
from training.preparation import split_data_on_stay_ids


class XGBClient(fl.client.Client):
    def __init__(
        self,
        cid: str,
        data: pd.DataFrame,
    ):
        self.bst = None  # keep the Booster objects that remain consistent across rounds
        self.config = None

        _path, _filename, config_settings = load_configuration()

        log(INFO, f"Client {cid} - Inital data shape: {data.shape}")

        # Split the data into training and test set based on stay_id
        train, test = split_data_on_stay_ids(data=data, test_size=0.2, random_state=0)

        # Define the features and target
        X_train = train.drop(columns=config_settings["training_columns_to_drop"])
        y_train = train["label"]
        X_valid = test.drop(columns=config_settings["training_columns_to_drop"])
        y_valid = test["label"]

        # Perform imputation
        imputer = init_imputer(X_train)
        X_train_imputed = imputer.fit_transform(X_train)
        X_valid_imputed = imputer.transform(X_valid)

        # Get number of samples
        self.num_train = len(X_train_imputed)
        self.num_valid = len(X_valid_imputed)

        log(
            INFO,
            f"Client {cid} - X_train shape: {X_train_imputed.shape} - X_valid shape: {X_valid_imputed.shape}",
        )

        # Reformat data to DMatrix for xgboost
        log(INFO, f"Reformatting data for client {cid}...")
        self.train_dmatrix = xgb.DMatrix(X_train_imputed, label=y_train)
        self.valid_dmatrix = xgb.DMatrix(X_valid_imputed, label=y_valid)

        # Hyperparamters for XGBoost training
        self.num_local_round = 1
        self.params = {
            "eval_metric": "aucpr",  # "auc" or "aucpr"
            "eta": 0.1,  # Learning rate
            "max_depth": 8,
            "num_parallel_tree": 1,
            "subsample": 1,
            "colsample_bytree": 1,
            "reg_lambda": 1,
            "objective": "binary:logistic",
            "tree_method": "hist",
        }

    def __build_model__(self):
        self.bst = xgb.Booster(params=self.params)

    def __set_model_params__(self, params: Parameters):
        surrogate = None
        for item in params.tensors:
            surrogate = bytearray(item)

        # load the surrogate model into the booster
        self.bst.load_model(surrogate)

    def __local_boost__(self, num_local_rounds):
        # update trees based on local training data.
        for _ in range(num_local_rounds):
            self.bst.update(self.train_dmatrix, self.bst.num_boosted_rounds())

        # extract the last N=num_local_round trees for sever aggregation
        bst = self.bst[
            self.bst.num_boosted_rounds()
            - num_local_rounds : self.bst.num_boosted_rounds()
        ]

        return bst

    def get_parameters(self, ins: GetParametersIns = None) -> GetParametersRes:
        if self.bst is not None:
            local_model = self.bst.save_raw("json")
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
        if self.bst is None:
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.train_dmatrix, "train")],
            )
            self.config = bst.save_config()
            self.bst = bst
        else:
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            self.bst.load_model(global_model)
            self.bst.load_config(self.config)

            bst = self.__local_boost__(self.num_local_round)

        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

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
        bst = xgb.Booster(params=self.params)
        for para in ins.parameters.tensors:
            para_b = bytearray(para)
        bst.load_model(para_b)

        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auprc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

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
