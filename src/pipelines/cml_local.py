import os
import logging
import sys

sys.path.append("..")

import ray

from sklearn.ensemble import RandomForestClassifier

from data.loading import load_parquet, load_configuration
from data.saving import save_csv
from data.processing import (
    encode_categorical_columns,
    drop_cols_with_perc_missing,
    impute,
    scale,
    reformat_time_column,
)
from data.feature_extraction import prepare_cohort_and_extract_features
from metrics.metrics import Metrics
from training.preparation import split_data_on_stay_ids


@ray.remote(num_cpus=2)
def single_local_run(
    data, test_size, random_state, columns_to_drop, missingness_cutoff
):
    """
    Perform a single run of the local pipeline.

    Args:
        data (pd.DataFrame): The data to be used.
        random_state (int): The random state to be used.
        columns_to_drop (list): The configuration settings.
    """
    logging.debug("Splitting data into a train and test subset...")
    train, test = split_data_on_stay_ids(data, test_size, random_state)

    logging.debug("Imputing missing values...")
    train = impute(train)
    test = impute(test)

    # Define the features and target
    X_train = train.drop(columns=columns_to_drop)
    y_train = train["label"]
    X_test = test.drop(columns=columns_to_drop)
    y_test = test["label"]

    logging.debug("Removing columns with all missing values...")
    X_train, X_test = drop_cols_with_perc_missing(X_train, X_test, missingness_cutoff)

    # Perform scaling
    X_train, X_test = scale(X_train, X_test)

    # Create & fit the model
    # the n_jobs parameter should be at most what the num_cpus value is in the ray.remote annotation
    model = RandomForestClassifier(
        n_estimators=100, max_depth=7, random_state=random_state, n_jobs=2
    )

    logging.debug("Training a model...")
    model.fit(X_train, y_train)

    # Predict the test set
    logging.debug("Predicting the test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    return (y_pred, y_pred_proba, y_test, test["stay_id"])


def cml_local_pipeline():
    logging.info("Loading configuration...")
    path, filename, config_settings = load_configuration()

    if not os.path.exists(os.path.join(path["features"], filename["features"])):
        logging.info("Preparing cohort and extracting features...")
        prepare_cohort_and_extract_features()

    # Load data with features
    logging.info("Loading data with extracted features...")
    data = load_parquet(path["features"], filename["features"], optimize_memory=True)

    # Additional preprocessing steps
    logging.info("Reformating dataframe...")
    data = reformat_time_column(data)
    data = encode_categorical_columns(data, config_settings["training_columns_to_drop"])

    metrics = Metrics()

    ray_futures = []
    logging.info(f"Creating the ray objects needed for parallelized execution...")
    # Repeat for each hospital_id in the dataset:
    for hospitalid in data["hospitalid"].unique():
        data_hospital = data[data["hospitalid"] == hospitalid]
        # Repeat with a different random state:
        for random_state in range(config_settings["random_split_reps"]):
            ray_futures.append(
                single_local_run.remote(
                    data_hospital,
                    config_settings["test_size"],
                    random_state,
                    config_settings["training_columns_to_drop"],
                    config_settings["missingness_cutoff"],
                )
            )

    logging.info("Executing local runs...")
    results = ray.get(ray_futures)

    logging.info("Extracting results per hospital and random_state...")
    for hospital_idx, hospitalid in enumerate(data["hospitalid"].unique()):
        # Extract the results per hospital and random state
        for random_state in range(config_settings["random_split_reps"]):
            (y_pred, y_pred_proba, y_test, stay_ids) = results[
                hospital_idx * config_settings["random_split_reps"] + random_state
            ]

            metrics.add_random_state(random_state)
            metrics.add_accuracy_value(y_test, y_pred)
            metrics.add_auroc_value(y_test, y_pred_proba)
            metrics.add_auprc_value(y_test, y_pred_proba)
            metrics.add_confusion_matrix(y_test, y_pred)
            metrics.add_individual_confusion_matrix_values(y_test, y_pred, stay_ids)
            metrics.add_tn_fp_sum()
            metrics.add_fpr()
            metrics.add_hospitalid(hospitalid)

        # Calculate averages accross random states for current hospital_id
        metrics.add_hospitalid_avg(hospitalid)
        metrics.add_metrics_mean(
            [
                "Accuracy",
                "AUROC",
                "AUPRC",
                "True Negatives",
                "Stay IDs with True Negatives",
                "False Positives",
                "Stay IDs with False Positives",
                "False Negatives",
                "Stay IDs with False Negatives",
                "True Positives",
                "Stay IDs with True Positives",
                "TN-FP-Sum",
                "FPR",
            ],
            config_settings["random_split_reps"],
        )
        metrics.add_metrics_std(
            [
                "Accuracy",
                "AUROC",
                "AUPRC",
                "True Negatives",
                "Stay IDs with True Negatives",
                "False Positives",
                "Stay IDs with False Positives",
                "False Negatives",
                "Stay IDs with False Negatives",
                "True Positives",
                "Stay IDs with True Positives",
                "TN-FP-Sum",
                "FPR",
            ],
            config_settings["random_split_reps"],
        )
        metrics.add_confusion_matrix_average()

    logging.info("Finalizing and saving results...")
    # Save normal metrics
    metrics_df = metrics.get_metrics_value_dataframe(
        [
            "Hospitalid",
            "Random State",
            "Accuracy",
            "AUROC",
            "AUPRC",
            "Confusion Matrix",
            "True Negatives",
            "Stay IDs with True Negatives",
            "False Positives",
            "Stay IDs with False Positives",
            "False Negatives",
            "Stay IDs with False Negatives",
            "True Positives",
            "Stay IDs with True Positives",
            "TN-FP-Sum",
            "FPR",
        ]
    )
    save_csv(metrics_df, path["results"], "cml_local_metrics.csv")

    # Calculate total averages
    metrics.add_hospitalid_avg("Total Average")
    metrics.add_metrics_mean(
        [
            "Accuracy",
            "AUROC",
            "AUPRC",
            "True Negatives",
            "Stay IDs with True Negatives",
            "False Positives",
            "Stay IDs with False Positives",
            "False Negatives",
            "Stay IDs with False Negatives",
            "True Positives",
            "Stay IDs with True Positives",
            "TN-FP-Sum",
            "FPR",
        ],
        on_mean_data=True,
    )
    metrics.add_metrics_std(
        [
            "Accuracy",
            "AUROC",
            "AUPRC",
            "True Negatives",
            "Stay IDs with True Negatives",
            "False Positives",
            "Stay IDs with False Positives",
            "False Negatives",
            "Stay IDs with False Negatives",
            "True Positives",
            "Stay IDs with True Positives",
            "TN-FP-Sum",
            "FPR",
        ],
        on_mean_data=True,
    )
    metrics.add_confusion_matrix_average(on_mean_data=True)

    # Save averages
    metrics_avg_df = metrics.get_metrics_avg_dataframe(
        [
            "Hospitalid",
            "Accuracy",
            "AUROC",
            "AUPRC",
            "Confusion Matrix",
            "True Negatives",
            "Stay IDs with True Negatives",
            "False Positives",
            "Stay IDs with False Positives",
            "False Negatives",
            "Stay IDs with False Negatives",
            "True Positives",
            "Stay IDs with True Positives",
            "TN-FP-Sum",
            "FPR",
        ]
    )
    save_csv(metrics_avg_df, path["results"], "cml_local_metrics_avg.csv")


if __name__ == "__main__":
    # Initialize the logging capability
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    os.environ["PYTHONPATH"] = ".." + os.pathsep + os.environ.get("PYTHONPATH", "")

    # Initialize ray
    if "SLURM_CPUS_PER_TASK" in os.environ:
        ray.init(num_cpus=int(os.environ["SLURM_CPUS_PER_TASK"]), _temp_dir="/tmp/")
    else:
        ray.init(num_cpus=2, _temp_dir="/tmp/")

    cml_local_pipeline()
