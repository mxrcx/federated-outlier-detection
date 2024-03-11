import os
import sys
import logging

sys.path.append("..")

from sklearn.ensemble import RandomForestClassifier

from data.loading import load_configuration, load_parquet
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


def single_random_split_run(
    data, test_size, random_state, columns_to_drop, metrics, missingness_cutoff
):
    """
    Perform a single run of the random split pipeline.

    Args:
        data (pd.DataFrame): The data to be used.
        random_state (int): The random state to be used.
        columns_to_drop (list): The configuration settings.
        metrics (Metrics): The metrics object.
    """
    # Split the data into training and test set based on stay_id
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

    logging.debug("Removing columns with missing values...")
    X_train, X_test = drop_cols_with_perc_missing(X_train, X_test, missingness_cutoff)

    # Perform scaling
    X_train, X_test = scale(X_train, X_test)

    # Create & fit the model
    model = RandomForestClassifier(
        n_estimators=100, max_depth=7, random_state=random_state, n_jobs=-1
    )

    logging.debug("Training a model...")
    model.fit(X_train, y_train)

    # Predict the test set
    logging.debug("Predicting the test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Add metrics
    logging.debug("Adding metrics...")
    metrics.add_random_state(random_state)
    metrics.add_accuracy_value(y_test, y_pred)
    metrics.add_auroc_value(y_test, y_pred_proba)
    metrics.add_auprc_value(y_test, y_pred_proba)
    metrics.add_confusion_matrix(y_test, y_pred)
    metrics.add_individual_confusion_matrix_values(y_test, y_pred, test["stay_id"])
    metrics.add_tn_fp_sum()
    metrics.add_fpr()


def cml_random_split_pipeline():
    """
    The pipeline for the CML random split experiment.
    """

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

    # Create metrics object
    metrics = Metrics()

    # Repeat with a different random state:
    for random_state in range(config_settings["random_split_reps"]):
        logging.info(f"Starting a single run with random_state={random_state}")
        single_random_split_run(
            data,
            config_settings["test_size"],
            random_state,
            config_settings["training_columns_to_drop"],
            metrics,
            config_settings["missingness_cutoff"],
        )

    logging.info("Calculating and saving metrics...")

    # Save normal metric values
    metrics_df = metrics.get_metrics_value_dataframe(
        [
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

    save_csv(metrics_df, path["results"], "cml_random_split_metrics.csv")

    # Calculate averages
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
        ]
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
        ]
    )
    metrics.add_confusion_matrix_average()

    # Save averages
    metrics_avg_df = metrics.get_metrics_avg_dataframe(
        [
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
    save_csv(metrics_avg_df, path["results"], "cml_random_split_metrics_avg.csv")


if __name__ == "__main__":

    # Initialize the logging capability
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    cml_random_split_pipeline()
