import os
import logging
import sys

sys.path.append("..")

import ray
import numpy as np
import xgboost as xgb
import matplotlib as plt

from data.loading import load_parquet, load_configuration
from data.saving import save_csv
from data.processing import impute, scale
from data.make_hospital_splits import make_hospital_splits
from metrics.metrics import Metrics
from metrics.metric_utils import get_y_score
from training.preparation import get_model, reformatting_model_name

# List to store feature importances from each xgb model
feature_importances_list = []

@ray.remote(num_cpus=2)
def single_local_run(train, test, model_name, path_to_results, random_state, columns_to_drop):
    """
    Perform a single run of the local pipeline.

    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The test data.
        random_state (int): The random state to use for the model.
        columns_to_drop (list): The columns to drop from the training and test data.
    """
    logging.debug("Imputing missing values...")
    train = impute(train)
    test = impute(test)
    
    # Add relative time column
    '''
    train = train.sort_values(by=['stay_id', 'time'])
    train['time_relative'] = train.groupby('stay_id').cumcount()
    test = test.sort_values(by=['stay_id', 'time'])
    test['time_relative'] = test.groupby('stay_id').cumcount()
    '''
    columns_to_drop.append("time")

    # Define the features and target
    X_train = train.drop(columns=columns_to_drop)
    y_train = train["label"]
    X_test = test.drop(columns=columns_to_drop)
    y_test = test["label"]

    # Perform scaling
    X_train, X_test = scale(X_train, X_test)

    # Create & fit the model
    # the n_jobs parameter should be at most what the num_cpus value is in the ray.remote annotation
    model = get_model(model_name, random_state, n_jobs=2)

    logging.debug("Training a model...")
    model.fit(X_train, y_train)

    # Predict the test set
    logging.debug("Predicting the test set...")
    y_pred = model.predict(X_test)

    try:
        y_score = get_y_score(model.predict_proba(X_test))
    except AttributeError:
        y_score = model.decision_function(X_test)
        
    # Add feature importances of current xgb model to list
    if model_name == "xgboostclassifier":
        feature_importances_list.append(model.feature_importances_)
            
    # Invert the outcome label
    if model_name in ["isolationforest", "oneclasssvm"]:
        y_pred = [1 if x == -1 else 0 for x in y_pred]
        y_score = y_score * -1

    return (y_pred, y_score, y_test, test["stay_id"])


def local_learning_pipeline():
    logging.info("Loading configuration...")
    path, _filename, config_settings = load_configuration()
    
    # Get model_name
    if len(sys.argv) > 1:
        model_name = reformatting_model_name(sys.argv[1])
    else:
        model_name = reformatting_model_name(config_settings["model"])

    if not os.path.exists(os.path.join(path["splits"], "individual_hospital_splits")):
        logging.info("Make hospital splits...")
        make_hospital_splits()

    # Load hospitalids
    hospitalids = np.load(os.path.join(path["splits"], "hospitalids.npy"))

    metrics = Metrics()
    ray_futures = []
    logging.info(f"Creating the ray objects needed for parallelized execution...")
    for hospitalid in hospitalids:
        for random_state in range(config_settings["random_split_reps"]):
            train = load_parquet(
                os.path.join(path["splits"], "individual_hospital_splits"),
                f"hospital{hospitalid}_rstate{random_state}_train.parquet",
                optimize_memory=True,
            )
            test = load_parquet(
                os.path.join(path["splits"], "individual_hospital_splits"),
                f"hospital{hospitalid}_rstate{random_state}_test.parquet",
                optimize_memory=True,
            )

            ray_futures.append(
                single_local_run.remote(
                    train,
                    test,
                    model_name,
                    path["results"],
                    random_state,
                    config_settings["training_columns_to_drop"],
                )
            )

    logging.info("Executing local runs...")
    results = ray.get(ray_futures)

    logging.info("Extracting results per hospital and random_state...")
    for hospital_idx, hospitalid in enumerate(hospitalids):
        # Extract the results per hospital and random state
        for random_state in range(config_settings["random_split_reps"]):
            (y_pred, y_score, y_test, stay_ids) = results[
                hospital_idx * config_settings["random_split_reps"] + random_state
            ]

            metrics.add_hospitalid(hospitalid)
            metrics.add_random_state(random_state)
            metrics.add_accuracy_value(y_test, y_pred)
            metrics.add_auroc_value(y_test, y_score)
            metrics.add_auprc_value(y_test, y_score)
            metrics.add_individual_confusion_matrix_values(y_test, y_pred, stay_ids)

    logging.info("Calculating averages and saving results...")
    metrics_df = metrics.get_metrics_dataframe(
        additional_metrics=["Hospitalid", "Random State"]
    )
    save_csv(metrics_df, path["results"], f"local_{model_name}_metrics.csv")

    metrics.calculate_averages_per_hospitalid_across_random_states()
    metrics.calculate_total_averages_across_hospitalids()
    metrics_avg_df = metrics.get_metrics_dataframe(
        additional_metrics=["Hospitalid"], avg_metrics=True
    )
    save_csv(metrics_avg_df, path["results"], f"local_{model_name}_metrics_avg.csv")
    
    if model_name == "xgboostclassifier":
        print("Shape of each inner list:")
        for i, inner_list in enumerate(feature_importances_list):
            print(f"Inner list {i + 1}: {len(inner_list)}")
        
        # Create average feature importances plot        
        average_feature_importances = []

        # Iterate over each column group
        for col_group in range(np.array(feature_importances_list).shape[1]):
            # Extract the column group and calculate its mean, ignoring NaN values
            mean_value = np.nanmean(feature_importances_list[:, col_group])
            average_feature_importances.append(mean_value)
        print(average_feature_importances)
        
        # Sort feature importances and select top 15
        feature_names = train.drop(columns=config_settings["training_columns_to_drop"].append("time")).columns
        sorted_indices = np.argsort(average_feature_importances)[::-1][:15]
        print(sorted_indices)
        top_feature_importances = average_feature_importances[sorted_indices]
        top_feature_names = [feature_names[i] for i in sorted_indices]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(top_feature_importances)), top_feature_importances, tick_label=top_feature_names)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Average Feature Importances')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(path["results"], "average_local_xgboost_feature_importances.png"))
        plt.show()

if __name__ == "__main__":
    # Initialize the logging capability
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    os.environ["PYTHONPATH"] = ".." + os.pathsep + os.environ.get("PYTHONPATH", "")

    # Initialize ray
    if "SLURM_CPUS_PER_TASK" in os.environ:
        ray.init(num_cpus=int(os.environ["SLURM_CPUS_PER_TASK"]), _temp_dir="/tmp/")
    else:
        ray.init(num_cpus=2, _temp_dir="/tmp/")

    local_learning_pipeline()
