import os
import logging
import sys
from types import SimpleNamespace

sys.path.append("..")

from logging import INFO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import flwr as fl
from flwr.common.logger import log
from flwr.server.strategy import FedXgbBagging
from federated_xgboost_client import XGBClient

from data.loading import load_parquet, load_configuration
from data.saving import save_csv
from data.make_hospital_splits import make_hospital_splits
from data.processing import impute, scale_X_test
from metrics.metrics import Metrics
from metrics.metric_utils import get_y_score
from training.preparation import get_model
import xgboost as xgb

# FL experimental settings
NUM_CLIENTS = 131  # 131
NUM_ROUNDS = 10

# Persistent storage
persistent_storage = {}


def get_server_evaluate(random_state):
    def server_evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: SimpleNamespace
    ):
        """A function to be executed at the end of each communication round. Traditionally, evaluation of the server side model is performed here,
        but in our case, it will only be used to save the model parameters into a global variable.

        Args:
            server_round (int): the communication round
            parameters (fl.common.NDArrays): parameters of the aggregated model #! This needs to be updated
            config (Dict[str, fl.common.Scalar]): a configuration dictionary
        """
        global persistent_storage
        persistent_storage[f"last_model_params_rstate{random_state}"] = parameters

    return server_evaluate


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
def get_strategy(random_state):
    strategy = FedXgbBagging(
        fraction_fit=0.45,  # Sample 45% of available clients for training
        fraction_evaluate=0.175,  # Sample 22.5% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=max(
            10, int(NUM_CLIENTS * 0.45)
        ),  # Wait until at least 35% of clients are available
        evaluate_function=get_server_evaluate(random_state),
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    )

    return strategy


def get_client_fn(path_to_splits, hospitalids, random_state, training_columns_to_drop):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        hospitalid = hospitalids[int(cid)]
        client_id = int(cid) + 1

        log(INFO, f"Loading data for client {client_id}...")

        train = load_parquet(
            os.path.join(path_to_splits, "individual_hospital_splits"),
            f"hospital{hospitalid}_rstate{random_state}_train.parquet",
            optimize_memory=True,
        )
        test = load_parquet(
            os.path.join(path_to_splits, "individual_hospital_splits"),
            f"hospital{hospitalid}_rstate{random_state}_test.parquet",
            optimize_memory=True,
        )

        # Create and return client
        return XGBClient(client_id, train, test, training_columns_to_drop).to_client()

    return client_fn


def evaluate_model_on_all_clients(
    path_to_splits,
    path_to_results,
    hospitalids,
    random_split_reps,
    training_columns_to_drop,
    metrics,
):
    for random_state in range(random_split_reps):
        # Create an evaluation model and set its "weights" to the last saved during fl simulation
        logging.info("Create eval model with last saved params...")
        eval_model = get_model("xgboostclassifier", random_state, n_jobs=-1)
        global_model = None
        for item in persistent_storage[
            f"last_model_params_rstate{random_state}"
        ].tensors:
            global_model = bytearray(item)
        eval_model.load_model(global_model)

        # Create feature importance plot
        if random_state == 0:
            # ax = xgb.plot_importance(eval_model, max_num_features=15)
            # ax.figure.tight_layout()
            # ax.figure.savefig(os.path.join(path_to_results, "federated_xgboost_feature_importance.png"))
            importances = eval_model.get_booster().get_score(importance_type="weight")
            feature_importance_df = pd.DataFrame(
                {
                    "Feature": list(importances.keys()),
                    "Importance": list(importances.values()),
                }
            )
            feature_importance_df = feature_importance_df.sort_values(
                by="Importance", ascending=False
            )
            feature_importance_df["Rank"] = range(1, len(feature_importance_df) + 1)

            # Only show top 15 features
            feature_importance_df_top15 = feature_importance_df[
                feature_importance_df["Rank"] <= 15
            ]
            feature_importance_df_top15.sort_values(by="Rank", inplace=True, ascending=False)

            plt.figure(figsize=(10, 6))
            plt.barh(
                feature_importance_df_top15["Feature"],
                feature_importance_df_top15["Rank"],
                color="skyblue",
            )
            plt.xlabel("Rank")
            plt.ylabel("Feature")
            plt.title("Federated XGBoost - Feature Importance by Rank")
            plt.tight_layout()
            plt.show()
            plt.savefig(
                os.path.join(
                    path_to_results, "federated_xgboost_feature_importance.png"
                )
            )

        logging.info("Evaluate model on all clients...")
        for hospitalid in hospitalids:
            test = load_parquet(
                os.path.join(path_to_splits, "individual_hospital_splits"),
                f"hospital{hospitalid}_rstate{random_state}_test.parquet",
                optimize_memory=True,
            )

            # Perform imputation
            test = impute(test)

            training_columns_to_drop.append("time")

            # Define the features and target
            X_test = test.drop(columns=training_columns_to_drop)
            y_test = test["label"]

            # Perform scaling
            X_test = scale_X_test(X_test)

            # Evaluate
            y_pred = eval_model.predict(X_test)
            y_score = get_y_score(eval_model.predict_proba(X_test))

            metrics.add_hospitalid(hospitalid)
            metrics.add_random_state(random_state)
            metrics.add_accuracy_value(y_test, y_pred)
            metrics.add_auroc_value(y_test, y_score)
            metrics.add_auprc_value(y_test, y_score)
            metrics.add_individual_confusion_matrix_values(
                y_test, y_pred, test["stay_id"]
            )


def run_federated_xgboost_simulation():
    logging.info("Loading configuration...")
    path, filename, config_settings = load_configuration()

    if not os.path.exists(os.path.join(path["splits"], "individual_hospital_splits")):
        logging.info("Make hospital splits...")
        make_hospital_splits()

    # Load hospitalids
    hospitalids = np.load(os.path.join(path["splits"], "hospitalids.npy"))

    # Launch the simulation with differnt random_state
    metrics = Metrics()
    for random_state in range(config_settings["random_split_reps"]):
        hist = fl.simulation.start_simulation(
            client_fn=get_client_fn(
                path["splits"],
                hospitalids,
                random_state,
                config_settings["training_columns_to_drop"],
            ),  # A function to run a _virtual_ client when required
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(
                num_rounds=NUM_ROUNDS
            ),  # Specify number of FL rounds
            strategy=get_strategy(random_state),  # Specify the strategy
            ray_init_args={
                "_temp_dir": "/tmp/"
            },  # temp file due to path length error "OSError: AF_UNIX path length cannot exceed 107 bytes"
        )

    evaluate_model_on_all_clients(
        path["splits"],
        path["results"],
        hospitalids,
        config_settings["random_split_reps"],
        config_settings["training_columns_to_drop"],
        metrics,
    )

    logging.info("Calculating metric averages and saving results...")
    metrics_df = metrics.get_metrics_dataframe(
        additional_metrics=["Hospitalid", "Random State"]
    )
    save_csv(metrics_df, path["results"], "federated_xgboostclassifier_metrics.csv")

    metrics.calculate_averages_per_hospitalid_across_random_states()
    metrics.calculate_total_averages_across_hospitalids()
    metrics_avg_df = metrics.get_metrics_dataframe(
        additional_metrics=["Hospitalid"], avg_metrics=True
    )
    save_csv(
        metrics_avg_df, path["results"], "federated_xgboostclassifier_metrics_avg.csv"
    )

    summary_results = metrics.get_summary_dataframe()
    save_csv(
        summary_results,
        os.path.join(path["results"], "summary"),
        f"federated_xgboostclassifier_summary.csv",
    )


if __name__ == "__main__":
    # Initialize the logging capability
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    run_federated_xgboost_simulation()
