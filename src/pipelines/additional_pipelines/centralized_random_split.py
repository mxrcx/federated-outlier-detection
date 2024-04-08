import os
import sys
import logging

sys.path.append("..")

from sklearn.ensemble import RandomForestClassifier

from data.loading import load_configuration, load_parquet
from data.saving import save_csv
from data.processing import impute, scale
from data.preprocessing import preprocessing
from metrics.metrics import Metrics
from training.preparation import split_data_on_stay_ids


def single_random_split_run(
    data,
    test_size,
    random_state,
    columns_to_drop,
    metrics,
):
    """
    Perform a single run of the random split pipeline.

    Args:
        data (pd.DataFrame): The data to be used.
        test_size (float): The size of the test set.
        random_state (int): The random state to be used.
        columns_to_drop (list): The configuration settings.
        metrics (Metrics): The metrics object.
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

    # Perform scaling
    X_train, X_test = scale(X_train, X_test)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=7, random_state=random_state, n_jobs=-1
    )

    logging.debug("Training a model...")
    model.fit(X_train, y_train)

    logging.debug("Predicting the test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    logging.debug("Adding metrics...")
    metrics.add_random_state(random_state)
    metrics.add_accuracy_value(y_test, y_pred)
    metrics.add_auroc_value(y_test, y_pred_proba)
    metrics.add_auprc_value(y_test, y_pred_proba)
    metrics.add_confusion_matrix(y_test, y_pred)
    metrics.add_individual_confusion_matrix_values(y_test, y_pred, test["stay_id"])
    metrics.add_tn_fp_sum()
    metrics.add_fpr()


def centralized_learning_random_split_pipeline():
    logging.debug("Loading configuration...")
    path, filename, config_settings = load_configuration()

    if not os.path.exists(os.path.join(path["processed"], filename["processed"])):
        logging.info("Preprocess data...")
        preprocessing()

    logging.debug("Loading preprocessed data...")
    data = load_parquet(path["processed"], filename["processed"], optimize_memory=True)

    metrics = Metrics()
    for random_state in range(config_settings["random_split_reps"]):
        logging.info(f"Starting a single run with random_state={random_state}")
        single_random_split_run(
            data,
            config_settings["test_size"],
            random_state,
            config_settings["training_columns_to_drop"],
            metrics,
        )

    logging.info("Calculating averages and saving results...")
    metrics_df = metrics.get_metrics_dataframe(additional_metrics=["Random State"])
    save_csv(
        metrics_df, path["results"], "centralized_learning_random_split_metrics.csv"
    )

    metrics.calculate_averages_across_random_states()
    metrics_avg_df = metrics.get_metrics_dataframe(avg_metrics=True)
    save_csv(
        metrics_avg_df,
        path["results"],
        "centralized_learning_random_split_metrics_avg.csv",
    )


if __name__ == "__main__":
    # Initialize the logging capability
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    centralized_learning_random_split_pipeline()
