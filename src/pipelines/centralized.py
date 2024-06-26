import os
import logging
import sys
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd

sys.path.append("..")
from data.loading import load_parquet, load_configuration
from data.saving import save_csv
from data.processing import impute, scale
from data.make_hospital_splits import make_hospital_splits
from metrics.metrics import Metrics
from metrics.metric_utils import get_y_score
from training.preparation import get_model, reformatting_model_name

feature_importances = []


def single_cv_run(
    model_name,
    path_to_splits,
    path_to_results,
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

        columns_to_drop.append("time")

        # For OCSVM: Remove the observations with sepsis label from training set
        if model_name == "oneclasssvm":
            train = train[train["label"] == 0]

        # Define the features and target
        X_train = train.drop(columns=columns_to_drop)
        y_train = train["label"]
        X_test = test.drop(columns=columns_to_drop)
        y_test = test["label"]

        # Perform scaling
        X_train, X_test = scale(X_train, X_test)

        # Fit model
        model.fit(X_train, y_train)

        # Predict the test set
        y_pred = model.predict(X_test)
        try:
            y_score = get_y_score(model.predict_proba(X_test))
        except AttributeError:
            y_score = model.decision_function(X_test)

        # Add metrics to the metrics object for each hospital
        for hospitalid in test["hospitalid"].unique():
            mask = test["hospitalid"] == hospitalid
            y_test_hospital = y_test[mask]
            y_pred_hospital = y_pred[mask]
            y_score_hospital = y_score[mask]

            # Invert the outcome label
            if model_name in ["isolationforest", "oneclasssvm"]:
                y_pred_hospital = [1 if x == -1 else 0 for x in y_pred_hospital]
                y_score_hospital = y_score_hospital * -1

            metrics.add_hospitalid(hospitalid)
            metrics.add_random_state(random_state)
            metrics.add_accuracy_value(y_test_hospital, y_pred_hospital)
            metrics.add_auroc_value(y_test_hospital, y_score_hospital)
            metrics.add_auprc_value(y_test_hospital, y_score_hospital)
            metrics.add_individual_confusion_matrix_values(
                y_test_hospital, y_pred_hospital, test["stay_id"][mask]
            )

        # Add feature importances of current xgb model to list
        if model_name == "xgboostclassifier" and random_state == 0:
            intermediate_importances = model.get_booster().get_score(
                importance_type="weight"
            )
            intermediate_feature_importance_df = pd.DataFrame(
                {
                    "Feature": list(intermediate_importances.keys()),
                    "Importance": list(intermediate_importances.values()),
                }
            )
            intermediate_feature_importance_df = (
                intermediate_feature_importance_df.sort_values(
                    by="Importance", ascending=False
                )
            )
            feature_importances.append(
                intermediate_feature_importance_df["Feature"].to_list()
            )

    # Create feature importance plot
    if model_name == "xgboostclassifier" and random_state == 0:
        feature_ranks = {}
        for inner_list in feature_importances:
            for i, feature in enumerate(inner_list):
                if feature not in feature_ranks:
                    feature_ranks[feature] = []
                feature_ranks[feature].append(i + 1)  # Adjusting index to start from 1

        # Calculate average rank for each feature
        average_ranks = {
            feature: round(sum(ranks) / len(ranks)) for feature, ranks in feature_ranks.items()
        }
        feature_importance_df = pd.DataFrame(
            list(average_ranks.items()), columns=["Feature", "Rank"]
        )

        # Only show top 15 features
        feature_importance_df_top15 = feature_importance_df[
            feature_importance_df["Rank"] <= 15
        ]
        feature_importance_df_top15.sort_values(by="Rank", inplace=True, ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(
            feature_importance_df_top15["Feature"],
            feature_importance_df_top15["Rank"],
            color="skyblue",
        )
        plt.xlabel("Average Rank")
        plt.ylabel("Feature")
        plt.title("Centralized XGBoost - Feature Importance by Rank")
        plt.tight_layout()
        plt.show()
        plt.savefig(
            os.path.join(path_to_results, "centralized_xgboost_feature_importance.png")
        )

        # ax = xgb.plot_importance(model, max_num_features=15)
        # ax.figure.tight_layout()
        # ax.figure.savefig(os.path.join(path_to_results, "centralized_xgboost_feature_importance.png"))


def leave_one_group_out_pipeline():
    logging.info("Loading configuration...")
    path, _filename, config_settings = load_configuration()

    # Get model_name
    if len(sys.argv) > 1:
        model_name = reformatting_model_name(sys.argv[1])
    else:
        model_name = reformatting_model_name(config_settings["model"])

    if not os.path.exists(os.path.join(path["splits"], "group_hospital_splits")):
        logging.info("Make hospital splits...")
        make_hospital_splits()

    metrics = Metrics()

    for random_state in range(config_settings["random_split_reps"]):
        single_cv_run(
            model_name,
            path["splits"],
            path["results"],
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

    summary_results = metrics.get_summary_dataframe()
    save_csv(
        summary_results,
        os.path.join(path["results"], "summary"),
        f"centralized_{model_name}_summary.csv",
    )


if __name__ == "__main__":
    # Initialize the logging capability
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    leave_one_group_out_pipeline()
