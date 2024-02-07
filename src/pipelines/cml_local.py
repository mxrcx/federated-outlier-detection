import os
import pandas as pd

from sklearn.model_selection import train_test_split
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


def single_local_run(data_hospital, random_state, columns_to_drop):
    """
    Perform a single run of the local pipeline.

    Args:
        data_hospital (pd.DataFrame): The data to be used.
        random_state (int): The random state to be used.
        columns_to_drop (list): The configuration settings.

    Returns:
        acc (float): The accuracy.
        auroc (float): The AUROC.
        auprc (float): The AUPRC.
        cm (np.ndarray): The confusion matrix.
    """
    print(f"Random State {random_state}...", end="", flush=True)

    # Split the data into training and test set based on stay_id
    stay_ids = data_hospital["stay_id"].unique()
    train_stay_ids, test_stay_ids = train_test_split(
        stay_ids, test_size=0.2, random_state=random_state
    )
    train = data_hospital[data_hospital["stay_id"].isin(train_stay_ids)]
    test = data_hospital[data_hospital["stay_id"].isin(test_stay_ids)]

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


def save_metrics_as_csv(
    hospitalids, random_states, accs, aurocs, auprcs, cms, path, filename
):
    """
    Save the metrics as a csv file.

    Args:
        hospitalids (list): The list of hospitalids.
        random_states (list): The list of random states.
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
            "Hospital ID": hospitalids,
            "Random State": random_states,
            "Accuracy": accs,
            "AUROC": aurocs,
            "AUPRC": auprcs,
            "Confusion_Matrix": cms,
        }
    )
    save_csv(metrics_df, path, filename)


def cml_local_pipeline():
    path, filename, config_settings = load_configuration()

    if not os.path.exists(os.path.join(path["features"], filename["features"])):
        prepare_cohort_and_extract_features()

    # Load data with features
    data = load_parquet(path["features"], filename["features"])

    # Additional preprocessing steps
    data = reformat_time_column(data)
    data = encode_categorical_columns(data, config_settings["training_columns_to_drop"])

    # Create lists for saving metrics, hospitalids and random_states
    hospitalids = []
    random_states = []
    accs = []
    aurocs = []
    auprcs = []
    cms = []

    # Repeate for each hospital_id in the dataset:
    for hospitalid in data["hospitalid"].unique():
        print(f"Hospital ID {hospitalid}:", flush=True)
        data_hospital = data[data["hospitalid"] == hospitalid]

        # Repeate with a different random state:
        for random_state in range(config_settings["random_split_reps"]):
            acc, auroc, auprc, cm = single_local_run(
                data_hospital,
                random_state,
                config_settings["training_columns_to_drop"],
            )
            hospitalids.append(hospitalid)
            random_states.append(random_state)
            accs.append(acc)
            aurocs.append(auroc)
            auprcs.append(auprc)
            cms.append(cm)

        # Calculate averages accross random states for current hospital_id
        hospitalids.append(hospitalid)
        random_states.append("Average")
        accs.append(get_average_metric(accs, config_settings["random_split_reps"]))
        aurocs.append(get_average_metric(aurocs, config_settings["random_split_reps"]))
        auprcs.append(get_average_metric(auprcs, config_settings["random_split_reps"]))
        cms.append(
            get_average_confusion_matrix(cms, config_settings["random_split_reps"])
        )

    save_metrics_as_csv(
        hospitalids,
        random_states,
        accs,
        aurocs,
        auprcs,
        cms,
        path["results"],
        "cml_local_metrics.csv",
    )


if __name__ == "__main__":
    cml_local_pipeline()
