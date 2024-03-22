import sys

sys.path.append("..")

from logging import INFO

from typing import List, Tuple
import flwr as fl
from flwr.common.logger import log
from flwr.common import (
    Metrics,
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
import numpy as np

from data.loading import load_parquet, load_configuration
from data.processing import (
    encode_categorical_columns,
    drop_cols_with_all_missing,
    init_imputer,
)
from training.preparation import split_data_on_stay_ids


# FL experimental settings
NUM_CLIENTS = 10  # 131
NUM_ROUNDS = 5


class GMMClient(fl.client.Client):
    def __init__(self, cid, data, num_clusters=3):
        path, filename, config_settings = load_configuration()

        log(INFO, f"Initial shape of client {cid} data - {data.shape}")

        # Additional preprocessing step: Convert 'time' column from timedelta to total seconds
        if "time" in data.columns:
            data.loc[:, "time"] = data["time"].dt.total_seconds()

        # Additional preprocessing step: Encode categorical data
        data = encode_categorical_columns(
            data, config_settings["training_columns_to_drop"]
        )

        # Additional preprocessing step: Drop all rows with missin 'label' values
        data = data.dropna(subset=["label"])

        # Split the data into training and test set based on stay_id
        train, test = split_data_on_stay_ids(data=data, test_size=0.2, random_state=0)

        # Define the features and target
        X_train = train.drop(columns=config_settings["training_columns_to_drop"])
        y_train = train["label"]
        X_valid = test.drop(columns=config_settings["training_columns_to_drop"])
        y_valid = test["label"]

        log(INFO, f"Value counts {y_train.value_counts()} - {y_valid.value_counts()}")

        # Perform preprocessing & imputation
        X_train_processed, X_valid_processed = drop_cols_with_all_missing(
            X_train, X_valid
        )
        imputer = init_imputer(X_train_processed)
        X_train_imputed = imputer.fit_transform(X_train_processed)
        X_valid_imputed = imputer.transform(X_valid_processed)

        # Save results
        self.x_train = X_train_imputed
        self.x_valid = X_valid_imputed
        self.y_train = y_train
        self.y_valid = y_valid
        self.num_clusters = num_clusters
        self.model = GaussianMixture(n_components=num_clusters)

    def fit(self, ins: FitIns) -> FitRes:
        self.model.weights_init = ins.parameters[0]
        self.model.means_init = ins.parameters[1]
        self.model.precisions_init = ins.parameters[2]
        self.model.fit(self.x_train)
    
        local_model = self.model.save_raw("json")
        local_model_bytes = bytes(local_model)
    
        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        self.model.weights_ = ins.parameters[0]
        self.model.means_ = ins.parameters[1]
        self.model.precisions_ = ins.parameters[2]

        y_pred = self.model.predict_proba(self.x_valid)[
            :, 1
        ]  # Assuming binary classification
        auprc = average_precision_score(self.y_valid, y_pred)
    
        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            metrics={"AUPRC": auprc},
        )


def weighted_avg(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Weighted averaging of GMM parameters
    weights_list = [m[1]["weights"] for _, m in metrics]
    means_list = [m[1]["means"] for _, m in metrics]
    precisions_list = [m[1]["precisions"] for _, m in metrics]

    weights_avg = np.average(weights_list, axis=0)
    means_avg = np.average(means_list, axis=0)
    precisions_avg = np.average(precisions_list, axis=0)

    return {"weights": weights_avg, "means": means_avg, "precisions": precisions_avg}


def get_client_fn():
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        client_id = int(cid) + 1

        log(INFO, f"Loading data for client {client_id}...")
        path, filename, config_settings = load_configuration()

        # Load full data
        full_data = load_parquet(path["features"], filename["features"])
        # Rank the hopsital ids and give them continuous ids starting from 1
        full_data["hospitalid"] = (
            full_data["hospitalid"].rank(method="dense").astype(int)
        )
        # Extract partition for client with ranked hospital_id = client_id
        client_data = full_data[full_data["hospitalid"] == client_id]

        # Create and return client
        return GMMClient(client_id, client_data).to_client()

    return client_fn


def aggregate_auprc(metrics):
    auprc_scores = [auprc for auprc, _, _ in metrics]
    auprc_avg = np.mean(auprc_scores)
    return auprc_avg


# Start Flower simulation
fl.simulation.start_simulation(
    client_fn=get_client_fn(),
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),  # Specify number of FL rounds
    strategy=fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=10,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn=aggregate_auprc,
    ),
    ray_init_args={"_temp_dir": "/tmp/"},
)
