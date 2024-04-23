import os
import sys
import logging

sys.path.append("..")

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from data.loading import load_configuration, load_parquet
from data.processing import impute, scale, scale_X_test
from data.preprocessing import preprocessing
from training.preparation import split_data_on_stay_ids, reformatting_model_name
import pandas as pd


def single_random_split_run(
    model_name,
    data,
    test_size,
    random_state,
    columns_to_drop,
):
    """
    Perform a single run of the random split pipeline.

    Args:
        data (pd.DataFrame): The data to be used.
        test_size (float): The size of the test set.
        random_state (int): The random state to be used.
        columns_to_drop (list): The configuration settings.
    """
    # logging.debug("Splitting data into a train and test subset...")
    # train, test = split_data_on_stay_ids(data, test_size, random_state)

    # Shrink down train and test to 30% of their current samples randomly
    # train = data.sample(frac=0.01, random_state=random_state)
    # test = test.sample(frac=0.3, random_state=random_state)

    logging.debug("Imputing missing values...")
    train = impute(data)
    # test = impute(test)

    # Define the features and target
    X_train = train.drop(columns=columns_to_drop)
    y_train = train["label"]
    # X_test = test.drop(columns=columns_to_drop)
    # y_test = test["label"]

    # Perform scaling
    # X_train, X_test = scale(X_train, X_test)
    X_train = scale_X_test(X_train)

    # Invert the label for outcome
    y_train = 1 - y_train

    if model_name == "isolationforest":
        model = IsolationForest(random_state=random_state, n_jobs=-1)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_samples": [0.25, 0.5, 1.0],
            "contamination": [0.01, 0.05, 0.1],
            "max_features": [0.5, 0.75, 1.0],
            "bootstrap": [True],
        }
        param_grid_random = {
            "n_estimators": [10, 50, 100, 150, 200, 300, 500],
            "max_samples": [0.25, 0.5, 0.75, 1.0],
            "contamination": [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5],
            "max_features": [0.5, 0.75, 1.0],
            "bootstrap": [True],
        }
    elif model_name == "gaussianmixture":
        model = GaussianMixture(random_state=random_state)
        param_grid = {
            "n_components": [1, 2, 4, 8, 16],
            "covariance_type": ["full", "tied", "diag", "spherical"],
            "init_params": ["kmeans", "random"],
            "max_iter": [50, 100, 200],
        }
        param_grid_random = {
            "n_components": [1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 16],
            "covariance_type": ["full", "tied", "diag", "spherical"],
            "init_params": ["kmeans", "random"],
            "max_iter": [50, 75, 100, 125, 150, 200, 250],
        }
    elif model_name == "oneclasssvm":
        model = SGDOneClassSVM()
        param_grid = {
            "nu": [0.01, 0.05, 0.1],
            "learning_rate": ["optimal", "adaptive", "constant", "invscaling"],
            "eta0": [0.01, 0.05, 0.1],
            "max_iter": [500, 1000, 1500],
        }
        param_grid_random = {
            "nu": [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5],
            "learning_rate": ["optimal", "adaptive", "constant", "invscaling"],
            "eta0": [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
            "max_iter": [500, 750, 1000, 1250, 1500, 2000],
        }
    else:
        model = IsolationForest(random_state=random_state, n_jobs=-1)
        param_grid = {}

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid_random,
        n_iter=20,
        scoring="average_precision",
        cv=5,
        random_state=random_state,
        n_jobs=-1,
    )

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring=["average_precision", "roc_auc", "accuracy"],
        refit="average_precision",
        n_jobs=-1,
    )

    logging.debug(f"Training a {model_name} model...")
    # grid_search.fit(X_train, y_train)
    random_search.fit(X_train, y_train)

    # best_params = grid_search.best_params_
    # best_score = grid_search.best_score_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    logging.debug(f"Best parameters for {model_name} found:")
    logging.debug(best_params)
    logging.debug(f"Best score for {model_name} found:")
    logging.debug(best_score)
    # logging.debug(grid_search.cv_results_)
    # logging.debug(random_search.cv_results_)


def grid_search_random_split():
    logging.debug("Loading configuration...")
    path, filename, config_settings = load_configuration()
    
    # Get model_name
    if len(sys.argv) > 1:
        model_name = reformatting_model_name(sys.argv[1])
    else:
        model_name = reformatting_model_name(config_settings["model"])

    if not os.path.exists(os.path.join(path["processed"], filename["processed"])):
        logging.info("Preprocess data...")
        preprocessing()

    # logging.debug("Loading preprocessed data...")
    # data = load_parquet(path["processed"], filename["processed"], optimize_memory=True)

    for random_state in range(config_settings["random_split_reps"]):
        logging.info(f"Starting a single run with random_state={random_state}")
        # hospital 58 or 67
        data = load_parquet(
            os.path.join(path["splits"], "group_hospital_splits"),
            f"fold0_rstate{random_state}_train.parquet",
            optimize_memory=True,
        )

        single_random_split_run(
            model_name,
            data,
            config_settings["test_size"],
            random_state,
            config_settings["training_columns_to_drop"],
        )


if __name__ == "__main__":
    # Initialize the logging capability
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    grid_search_random_split()
