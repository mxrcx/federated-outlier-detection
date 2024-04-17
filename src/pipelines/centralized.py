import os
import logging
import sys

sys.path.append("..")
from data.loading import load_parquet, load_configuration
from data.saving import save_csv
from data.processing import impute, scale
from data.make_hospital_splits import make_hospital_splits
from metrics.metrics import Metrics
from training.preparation import get_model, reformatting_model_name


def single_cv_run(
    model_name,
    path_to_splits,
    random_state,
    cv_folds,
    columns_to_drop,
    metrics,
):
    """
    Perform a single run of the stratified group k-fold split pipeline.

    Args:
        path_to_splits (str): The path to the splits.
        random_state (int): The random state to use for the model.
        cv_folds (int): The number of cross-validation folds.
        columns_to_drop (list): The columns to drop from the training and test data.
        metrics (Metrics): The metrics object.
    """
    # Create the model
    model = get_model(
        model_name, random_state, n_jobs=min(16, int(os.environ["SLURM_CPUS_PER_TASK"]))
    )

    # Perform stratified group k-fold cross-validation
    for fold in range(cv_folds):
        train = load_parquet(
            os.path.join(path_to_splits, "group_hospital_splits"),
            f"fold{fold}_rstate{random_state}_train.parquet",
            optimize_memory=True,
        )
        test = load_parquet(
            os.path.join(path_to_splits, "group_hospital_splits"),
            f"fold{fold}_rstate{random_state}_test.parquet",
            optimize_memory=True,
        )

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

        # Invert the outcome train label
        if model_name in ["isolationforest", "oneclasssvm"]:
            y_train = [-1 if x == 1 else 1 for x in y_train]

        # Fit model
        model.fit(X_train, y_train)

        # Predict the test set
        y_pred = model.predict(X_test)
        try:
            y_score = model.predict_proba(X_test)
        except AttributeError:
            y_score = model.decision_function(X_test)

        # Add metrics to the metrics object for each hospital
        for hospitalid in test["hospitalid"].unique():
            mask = test["hospitalid"] == hospitalid
            y_test_hospital = y_test[mask]
            y_pred_hospital = y_pred[mask]
            y_score_hospital = y_score[mask]

            # Invert the outcome test label
            if model_name in ["isolationforest", "oneclasssvm"]:
                y_test_hospital = [-1 if x == 1 else 1 for x in y_test_hospital]

            metrics.add_hospitalid(hospitalid)
            metrics.add_random_state(random_state)
            metrics.add_accuracy_value(y_test_hospital, y_pred_hospital)
            metrics.add_auroc_value(y_test_hospital, y_score_hospital)
            metrics.add_auprc_value(y_test_hospital, y_score_hospital)
            metrics.add_individual_confusion_matrix_values(
                y_test_hospital, y_pred_hospital, test["stay_id"][mask]
            )


def leave_one_group_out_pipeline():
    logging.info("Loading configuration...")
    path, _filename, config_settings = load_configuration()
    model_name = reformatting_model_name(config_settings["model"])

    if not os.path.exists(os.path.join(path["splits"], "group_hospital_splits")):
        logging.info("Make hospital splits...")
        make_hospital_splits()

    metrics = Metrics()

    for random_state in range(config_settings["random_split_reps"]):
        single_cv_run(
            model_name,
            path["splits"],
            random_state,
            config_settings["cv_folds"],
            config_settings["training_columns_to_drop"],
            metrics,
        )

    logging.info("Calculating averages and saving results...")
    metrics_df = metrics.get_metrics_dataframe(
        additional_metrics=["Hospitalid", "Random State"]
    )
    save_csv(metrics_df, path["results"], f"centralized_{model_name}_metrics.csv")

    metrics.calculate_averages_per_hospitalid_across_random_states()
    metrics.calculate_total_averages_across_hospitalids()
    metrics_avg_df = metrics.get_metrics_dataframe(
        additional_metrics=["Hospitalid"], avg_metrics=True
    )
    save_csv(
        metrics_avg_df, path["results"], f"centralized_{model_name}_metrics_avg.csv"
    )


if __name__ == "__main__":
    # Initialize the logging capability
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    leave_one_group_out_pipeline()
