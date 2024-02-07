import os
import sys
import yaml
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
from metrics.metrics import (
    get_accuracy,
    get_auroc,
    get_auprc,
    get_confusion_matrix,
    get_average_confusion_matrix,
    get_average_metric,
)


def single_loho_split_run(data, hospitalid, columns_to_drop):
    """
    Perform a single run of the leave-one-hospital-out split pipeline.

    Args:
        data (pd.DataFrame): The data to be used.
        hospitalid (int): The hospitalid to be left out.
        columns_to_drop (list): The configuration settings.

    Returns:
        acc (float): The accuracy.
        auroc (float): The AUROC.
        auprc (float): The AUPRC.
        cm (np.ndarray): The confusion matrix.
    """
    print(f"Hospital ID {hospitalid}...", end="", flush=True)

    # Split the data into training and test set
    train = data[data["hospitalid"] != hospitalid]
    test = data[data["hospitalid"] == hospitalid]

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

    # Calculate metrics
    acc = get_accuracy(y_test, y_pred)
    auroc = get_auroc(y_test, y_pred_proba)
    auprc = get_auprc(y_test, y_pred_proba)
    cm = get_confusion_matrix(y_test, y_pred)

    print(f"DONE.", flush=True)

    return acc, auroc, auprc, cm


def save_metrics_as_csv(hospitalids, accs, aurocs, auprcs, cms, path, filename):
    """
    Save the metrics as a csv file.

    Args:
        hospitalids (list): The list of hospitalids.
        accs (list): The list of accuracies.
        aurocs (list): The list of AUROCs.
        auprcs (list): The list of AUPRCs.
        cms (list): The list of confusion matrices.
        path (str): The path to the directory where the csv file should be saved.
        filename (str): The name of the csv file.

    Returns:
        None
    """
    metrics_df = pd.DataFrame(
        {
            "Hospital ID (left out/test set)": hospitalids,
            "Accuracy": accs,
            "AUROC": aurocs,
            "AUPRC": auprcs,
            "Confusion_Matrix": cms,
        }
    )
    save_csv(metrics_df, path, filename)


def cml_loho_split_pipeline():
    path, filename, config_settings = load_configuration()

    if not os.path.exists(os.path.join(path["features"], filename["features"])):
        prepare_cohort_and_extract_features()

    # Load data with features
    data = load_parquet(path["features"], filename["features"])

    # Additional preprocessing steps
    data = reformat_time_column(data)
    data = encode_categorical_columns(data, config_settings["training_columns_to_drop"])

    # Create lists for saving metrics
    hospitalids = []
    accs = []
    aurocs = []
    auprcs = []
    cms = []

    # Repeate for each hospital_id in the dataset:
    for hospitalid in data["hospitalid"].unique():
        acc, auroc, auprc, cm = single_loho_split_run(
            data, hospitalid, config_settings["training_columns_to_drop"]
        )
        hospitalids.append(hospitalid)
        accs.append(acc)
        aurocs.append(auroc)
        auprcs.append(auprc)
        cms.append(cm)

    # Calculate averages
    hospitalids.append("Average")
    accs.append(get_average_metric(accs))
    aurocs.append(get_average_metric(aurocs))
    auprcs.append(get_average_metric(auprcs))
    cms.append(get_average_confusion_matrix(cms))

    save_metrics_as_csv(
        hospitalids,
        accs,
        aurocs,
        auprcs,
        cms,
        path["results"],
        "cml_loho_split_metrics.csv",
    )


if __name__ == "__main__":
    cml_loho_split_pipeline()
