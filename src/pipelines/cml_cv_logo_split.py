import os
import logging
import sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

sys.path.append("..")
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


def single_cv_logo_split_run(
    data,
    groups,
    stratify_labels,
    random_state,
    columns_to_drop,
    metrics,
    missingness_cutoff,
    cv_folds,
):
    """
    Perform a single run of the stratified group k-fold split pipeline.

    Args:
        data (pd.DataFrame): The data to be used.
        groups (pd.Series): The group labels for the data.
        stratify_labels (pd.Series): The labels to stratify the groups by.
        random_state (int): The random state for reproducibility.
        columns_to_drop (list): The configuration settings.
        metrics (Metrics): The metrics object.
        missingness_cutoff (float): The missingness cutoff for columns.
        cv_folds (int): The number of cross-validation folds.
    """
    # Create the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        random_state=random_state,
        n_jobs=min(16, int(os.environ["SLURM_CPUS_PER_TASK"])),
    )

    # Perform stratified group k-fold cross-validation
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    for fold, (train_idx, test_idx) in enumerate(
        cv.split(data, y=stratify_labels, groups=groups)
    ):
        logging.debug(f"Training fold {fold + 1}...")
       
        # Define the features and target
        X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
        y = data["label"]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        logging.debug("Removing columns with missing values...")
        X_train, X_test = drop_cols_with_perc_missing(X_train, X_test, missingness_cutoff)
        
        logging.debug("Imputing missing values...")
        X_train = impute(X_train)
        X_valid = impute(X_valid)
        
        # Drop features
        X_train = X_train.drop(columns=columns_to_drop)
        X_valid = X_valid.drop(columns=columns_to_drop)

        # Scale
        X_train, X_valid = scale(X_train, X_valid)
        
        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Add metrics to the metrics object
        metrics.add_random_state(random_state)
        metrics.add_fold((fold + 1))
        metrics.add_accuracy_value(y_test, y_pred)
        metrics.add_auroc_value(y_test, y_pred_proba)
        metrics.add_auprc_value(y_test, y_pred_proba)
        metrics.add_confusion_matrix(y_test, y_pred)
        metrics.add_individual_confusion_matrix_values(
            y_test, y_pred, data.loc[test_idx, "stay_id"]
        )
        metrics.add_tn_fp_sum()
        metrics.add_fpr()


def cml_cv_logo_split_pipeline():
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

    # Determine group and stratification labels
    groups = data["hospitalid"]
    stratify_labels = (data.groupby("hospitalid")["label"].transform("sum") > 0).astype(
        int
    )
    print(stratify_labels.value_counts())

    # Repeat with different random states
    for random_state in range(config_settings["random_split_reps"]):
        single_cv_logo_split_run(
            data,
            groups,
            stratify_labels,
            random_state,
            config_settings["training_columns_to_drop"],
            metrics,
            config_settings["missingness_cutoff"],
            config_settings["cv_folds"],
        )

        # Calculate averages accross folds for current random state
        metrics.add_random_state_avg(random_state)
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
            "Random State",
            "Fold",
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
    save_csv(metrics_df, path["results"], "cml_cv_logo_split_metrics.csv")
    
    # Calculate averages
    metrics.add_random_state_avg("Total Average")
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
    save_csv(metrics_avg_df, path["results"], "cml_cv_logo_split_metrics_avg.csv")


if __name__ == "__main__":

    # Initialize the logging capability
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    cml_cv_logo_split_pipeline()
