import os
import logging
import sys

sys.path.append("..")

from sklearn.ensemble import RandomForestClassifier

from data.loading import load_parquet, load_configuration
from data.saving import save_csv
from data.processing import (
    encode_categorical_columns,
    drop_cols_with_perc_missing,
    init_imputer,
    reformat_time_column,
)
from data.feature_extraction import prepare_cohort_and_extract_features
from metrics.metrics import Metrics


def single_loho_split_run(
    data, hospitalid, random_state, columns_to_drop, metrics, missingness_cutoff
):
    """
    Perform a single run of the leave-one-hospital-out split pipeline.

    Args:
        data (pd.DataFrame): The data to be used.
        hospitalid (int): The hospitalid to be left out.
        columns_to_drop (list): The configuration settings.
        metrics (Metrics): The metrics object.
    """
    # Split the data into training and test set based on hospitalid
    logging.debug("Splitting data into a train and test subset...")
    train = data[data["hospitalid"] != hospitalid]
    test = data[data["hospitalid"] == hospitalid]

    # Define the features and target
    X_train = train.drop(columns=columns_to_drop)
    y_train = train["label"]
    X_test = test.drop(columns=columns_to_drop)
    y_test = test["label"]

    # Perform preprocessing & imputation
    logging.debug("Removing columns with missing values...")
    X_train, X_test = drop_cols_with_perc_missing(X_train, X_test, missingness_cutoff)

    logging.debug("Imputing missing values...")
    imputer = init_imputer(X_train)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Create & fit the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        random_state=random_state,
        n_jobs=min(16, int(os.environ["SLURM_CPUS_PER_TASK"])),
    )

    logging.debug("Training a model...")
    model.fit(X_train, y_train)

    # Predict the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Add metrics to the metrics object
    metrics.add_random_state(random_state)
    metrics.add_accuracy_value(y_test, y_pred)
    metrics.add_auroc_value(y_test, y_pred_proba)
    metrics.add_auprc_value(y_test, y_pred_proba)
    metrics.add_confusion_matrix(y_test, y_pred)
    metrics.add_individual_confusion_matrix_values(y_test, y_pred, test["stay_id"])
    metrics.add_tn_fp_sum()
    metrics.add_fpr()
    metrics.add_hospitalid(hospitalid)


def cml_loho_split_pipeline():
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

    # Repeat for each hospital_id in the dataset:
    logging.info("Executing runs...")
    for hospitalid in data["hospitalid"].unique():
        # Repeat with a different random state:
        for random_state in range(config_settings["random_split_reps"]):
            single_loho_split_run(
                data,
                hospitalid,
                random_state,
                config_settings["training_columns_to_drop"],
                metrics,
                config_settings["missingness_cutoff"],
            )

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
    save_csv(metrics_df, path["results"], "cml_loho_split_metrics.csv")

    # Calculate averages
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
    save_csv(metrics_avg_df, path["results"], "cml_loho_split_metrics_avg.csv")


if __name__ == "__main__":

    # Initialize the logging capability
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    cml_loho_split_pipeline()
