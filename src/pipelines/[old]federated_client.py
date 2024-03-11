from logging import INFO
import xgboost as xgb
import yaml

import flwr as fl
from flwr.common.logger import log
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from sklearn.model_selection import train_test_split

from data.loading import load_parquet, load_configuration
from data.processing import (
    encode_categorical_columns,
    drop_cols_with_all_missing,
    init_imputer,
)


# Define Flower Client
class XgbClient(fl.client.Client):
    """
    XGBoost Federated Learning Client.

    Args:
        client_id (int): ID of the client.
    """

    def __init__(self, client_id):
        self.bst = None  # keep the Booster objects that remain consistent across rounds
        self.config = None
        self.client_id = client_id

        path, filename, config_settings = load_configuration()

        # Load data
        log(INFO, f"Loading data for client {client_id}...")
        data = load_parquet(path["features"], filename["features"])
        # Rank the hopsital ids and give them continuous ids starting from 1
        data["hospitalid"] = data["hospitalid"].rank(method="dense").astype(int)
        # Select only instances with hospital id that matches client id
        data = data[data["hospitalid"] == int(client_id)]

        log(INFO, f"Initial shape of client {client_id} data - {data.shape}")

        # Additional preprocessing step: Convert 'time' column from timedelta to total seconds
        if "time" in data.columns:
            data["time"] = data["time"].dt.total_seconds()

        # Additional preprocessing step: Encode categorical data
        data = encode_categorical_columns(
            data, config_settings["training_columns_to_drop"]
        )

        # Additional preprocessing step: Drop all rows with missin 'label' values
        data = data.dropna(subset=["label"])

        # Split the data into training and test set
        train, test = train_test_split(data, random_state=0, test_size=0.2)

        self.X_train = train.drop(columns=config_settings["training_columns_to_drop"])
        self.y_train = train["label"]
        self.X_valid = test.drop(columns=config_settings["training_columns_to_drop"])
        self.y_valid = test["label"]

        # Perform preprocessing & imputation
        self.X_train, self.X_valid = drop_cols_with_all_missing(
            self.X_train, self.X_valid
        )
        imputer = init_imputer(self.X_train)
        self.X_train = imputer.fit_transform(self.X_train)
        self.X_valid = imputer.transform(self.X_valid)

        # Get number of samples
        self.num_train = len(self.X_train)
        self.num_valid = len(self.X_valid)

        # Reformat data to DMatrix for xgboost
        log(INFO, f"Reformatting data for client {client_id}...")
        self.train_dmatrix = xgb.DMatrix(self.X_train, label=self.y_train)
        self.valid_dmatrix = xgb.DMatrix(self.X_valid, label=self.y_valid)

        # Hyperparamters for XGBoost training
        self.num_local_round = 1
        self.params = {
            "objective": "binary:logistic",
            "eta": 0.1,  # learning rate
            "max_depth": 8,
            "eval_metric": "auc",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
        }

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """
        Get model parameters from the server.

        Args:
            ins (GetParametersIns): Input parameters for getting global model parameters.

        Returns:
            GetParametersRes: Response containing the global model parameters.
        """
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self):
        """
        Perform local boosting to update the trees based on local training data.

        Returns:
            xgb.Booster: Locally updated booster object.
        """
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            self.bst.update(self.train_dmatrix, self.bst.num_boosted_rounds())

        # Extract the last N=num_local_round trees for sever aggregation
        bst = self.bst[
            self.bst.num_boosted_rounds()
            - self.num_local_round : self.bst.num_boosted_rounds()
        ]

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        """
        Perform local training and return updated model parameters.
        -> This method andles the logic for both the initial training round and
        subsequent rounds where the global model is updated and boosted
        based on local training data.

        Args:
            ins (FitIns): Input parameters for fitting the model locally.

        Returns:
            FitRes: Response containing the locally trained model parameters.
        """
        if not self.bst:
            # First round local training
            log(INFO, "Start training at round 1")
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
            self.config = bst.save_config()
            self.bst = bst
        else:
            # Receive global model from server
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            self.bst.load_model(global_model)
            self.bst.load_config(self.config)

            # Update model weights on local training data
            bst = self._local_boost()

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
        """
        Evaluate the model on the validation set and return evaluation metrics.

        Args:
            ins (EvaluateIns): Input parameters for model evaluation.

        Returns:
            EvaluateRes: Response containing evaluation metrics.
        """
        if self.bst is None:
            self.__build_model__(ins.config)
            self.__set_model_params__(ins.parameters)
        eval_results = self.bst.eval_set(
            evals=[(self.train_dmatrix, "train")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.n_examples,
            metrics={"AUC": auc},
        )
