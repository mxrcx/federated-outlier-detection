import os
import sys
import yaml
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from data.loading import load_parquet
from data.processing import (
    encode_categorical_columns,
    drop_cols_with_all_missing,
    init_imputer,
)
from sklearn.preprocessing import OneHotEncoder
from analysis.metrics import (
    get_accuracy,
    get_auroc,
    get_auprc,
    get_confusion_matrix,
    get_average_confusion_matrix,
    get_average_metric,
)

# Load the YAML configuration file
with open("configuration.yml") as f:
    config = yaml.safe_load(f)

# Initial configuration
path_to_features = config["path"]["features"]
filename_features = config["filename"]["features"]
random_split_reps = config["split"]["random_split_reps"]
path_to_results = config["path"]["results"]
columns_to_drop = config["training"]["columns_to_drop"]

# Check if features have been extracted
if not os.path.exists(os.path.join(path_to_features, filename_features)):
    print(
        "No parquet file containing the features found. Please run the initial pipeline to extract features."
    )
    sys.exit(0)

# Load the features data
data = load_parquet(path_to_features, filename_features)

# Additional preprocessing step: Convert 'time' column from timedelta to total seconds
if "time" in data.columns:
    data["time"] = data["time"].dt.total_seconds()

# Additional preprocessing step: Encode categorical data
data = encode_categorical_columns(data, columns_to_drop)

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
    for random_state in range(random_split_reps):
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
        hospitalids.append(hospitalid)
        random_states.append(random_state)
        accs.append(get_accuracy(y_test, y_pred))
        aurocs.append(get_auroc(y_test, y_pred_proba))
        auprcs.append(get_auprc(y_test, y_pred_proba))
        cms.append(get_confusion_matrix(y_test, y_pred))

        print(f"DONE.", flush=True)

    # Calculate averages accross random states for current hospital_id
    hospitalids.append(hospitalid)
    random_states.append("Average")
    accs.append(get_average_metric(accs, random_split_reps))
    aurocs.append(get_average_metric(aurocs, random_split_reps))
    auprcs.append(get_average_metric(auprcs, random_split_reps))
    cms.append(get_average_confusion_matrix(cms, random_split_reps))

# Save metrics as csv file
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
metrics_df.to_csv(os.path.join(path_to_results, "cml_local_metrics.csv"), index=False)
