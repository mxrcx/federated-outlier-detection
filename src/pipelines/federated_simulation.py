import sys

sys.path.append("..")

from logging import INFO
import numpy as np

import flwr as fl
from flwr.common.logger import log
from flwr.server.strategy import FedXgbBagging
from federated_client import XGBClient

from data.loading import load_parquet, load_configuration
from data.processing import reformat_time_column, encode_categorical_columns

# FL experimental settings
NUM_CLIENTS = 65  # 131
NUM_ROUNDS = 10

auprc_array = []


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUPRC) for evaluation."""
    valid_metrics = [
        (num, metrics)
        for num, metrics in eval_metrics
        if not np.isnan(metrics["AUPRC"])
    ]

    if not valid_metrics:
        return {"AUPRC": np.nan}

    total_num = sum(num for num, _ in valid_metrics)
    auprc_sum = sum(metrics["AUPRC"] * num for num, metrics in valid_metrics)
    auprc_aggregated = auprc_sum / total_num

    metrics_aggregated = {"AUPRC": auprc_aggregated}
    return metrics_aggregated


# Define strategy
strategy = FedXgbBagging(
    fraction_fit=0.35,  # Sample 35% of available clients for training
    fraction_evaluate=0.175,  # Sample 17.5% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=max(
        10, int(NUM_CLIENTS * 0.35)
    ),  # Wait until at least 35% of clients are available
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
)


def get_client_fn(full_data, continuous_hospitalids):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        client_id = int(cid) + 1
        log(INFO, f"Loading data for client {client_id}...")

        # Extract partition for client with ranked hospital_id = client_id
        print(continuous_hospitalids)
        hospitalid_indices = np.where(continuous_hospitalids == client_id)[0]
        client_data = full_data.iloc[hospitalid_indices]

        # Create and return client
        return XGBClient(client_id, client_data).to_client()

    return client_fn


def prepare_data():
    path, filename, config_settings = load_configuration()

    # Load full data
    full_data = load_parquet(path["features"], filename["features"])

    full_data = reformat_time_column(full_data)
    full_data = encode_categorical_columns(
        full_data, config_settings["training_columns_to_drop"]
    )
    # Drop all rows with missing 'label' values
    full_data = full_data.dropna(subset=["label"])

    # Drop columns with missingness above threshold
    cols_to_drop = set(
        full_data.columns[
            full_data.groupby("hospitalid")
            .apply(lambda x: x.isnull().mean() > config_settings["missingness_cutoff"])
            .any()
        ]
    )
    full_data.drop(cols_to_drop, axis=1, inplace=True)

    # Rank the hopsital ids and give them continuous ids starting from 1
    continuous_hospitalids = full_data["hospitalid"].rank(method="dense").astype(int)

    return full_data, continuous_hospitalids


def run_federated_simulation():
    full_data, continuous_hospitalids = prepare_data()

    # Launch the simulation
    hist = fl.simulation.start_simulation(
        client_fn=get_client_fn(
            full_data, continuous_hospitalids
        ),  # A function to run a _virtual_ client when required
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),  # Specify number of FL rounds
        strategy=strategy,
        ray_init_args={
            "_temp_dir": "/tmp/"
        },  # temp file due to path length error "OSError: AF_UNIX path length cannot exceed 107 bytes"
    )


if __name__ == "__main__":
    run_federated_simulation()
