import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from data.loading import load_parquet, load_configuration
from data.saving import save_csv
from data.processing import (
    encode_categorical_columns,
    drop_cols_with_all_missing,
    init_imputer,
    reformat_time_column,
)
from data.feature_extraction import prepare_cohort_and_extract_features
from metrics.metrics import Metrics


def single_loho_split_run(data, hospitalid, columns_to_drop, metrics):
    """
    Perform a single run of the leave-one-hospital-out split pipeline.

    Args:
        data (pd.DataFrame): The data to be used.
        hospitalid (int): The hospitalid to be left out.
        columns_to_drop (list): The configuration settings.
        metrics (Metrics): The metrics object.
    """
    print(f"Hospital ID {hospitalid}...", end="", flush=True)

    # Split the data into training and test set based ion hospitalid
    train = data[data["hospitalid"] != hospitalid]
    test = data[data["hospitalid"] == hospitalid]

    # Define the features and target
    X_train = train.drop(columns=columns_to_drop)
    y_train = train["label"]
    X_test = test.drop(columns=columns_to_drop)
    y_test = test["label"]

    # Perform preprocessing & imputation
    X_train, X_test = drop_cols_with_all_missing(X_train, X_test)
    imputer = init_imputer(X_train)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Create & fit the model
    model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Add metrics
    metrics.add_accuracy_value(y_test, y_pred)
    metrics.add_auroc_value(y_test, y_pred_proba)
    metrics.add_auprc_value(y_test, y_pred_proba)
    metrics.add_confusion_matrix(y_test, y_pred)
    metrics.add_tn_fp_sum()
    metrics.add_fpr()

    print(f"DONE.", flush=True)


def cml_loho_split_pipeline():
    path, filename, config_settings = load_configuration()

    if not os.path.exists(os.path.join(path["features"], filename["features"])):
        prepare_cohort_and_extract_features()

    # Load data with features
    data = load_parquet(path["features"], filename["features"])

    # Additional preprocessing steps
    data = reformat_time_column(data)
    data = encode_categorical_columns(data, config_settings["training_columns_to_drop"])

    # Create metrics object
    metrics = Metrics()

    # Repeate for each hospital_id in the dataset:
    for hospitalid in data["hospitalid"].unique():
        single_loho_split_run(
            data, hospitalid, config_settings["training_columns_to_drop"], metrics
        )
        metrics.add_hospitalid(hospitalid)

    # Save normal metrics
    metrics_df = metrics.get_metrics_value_dataframe(
        [
            "Hospitalid",
            "Accuracy",
            "AUROC",
            "AUPRC",
            "Confusion Matrix",
            "TN-FP-Sum",
            "FPR",
        ]
    )
    save_csv(metrics_df, path["results"], "cml_loho_split_metrics.csv")

    # Calculate averages
    metrics.add_metrics_mean(["Accuracy", "AUROC", "AUPRC", "TN-FP-Sum", "FPR"])
    metrics.add_metrics_std(["Accuracy", "AUROC", "AUPRC", "TN-FP-Sum", "FPR"])
    metrics.add_confusion_matrix_average()

    # Save averages
    metrics_avg_df = metrics.get_metrics_avg_dataframe(
        ["Accuracy", "AUROC", "AUPRC", "Confusion Matrix", "TN-FP-Sum", "FPR"]
    )
    save_csv(metrics_avg_df, path["results"], "cml_loho_split_metrics_avg.csv")


if __name__ == "__main__":
    cml_loho_split_pipeline()
