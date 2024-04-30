import os
import sys
import logging

sys.path.append("..")

from sklearn.ensemble import IsolationForest, RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import SGDOneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, average_precision_score

from data.loading import load_configuration, load_parquet
from data.processing import impute, scale, scale_X_test
from data.preprocessing import preprocessing
from training.preparation import split_data_on_stay_ids, reformatting_model_name
import pandas as pd
import numpy as np


def single_run(
    model_name,
    data,
    test_size,
    random_state,
    columns_to_drop,
):
    logging.debug("Imputing missing values...")
    train = impute(data)

    # Define the features and target
    columns_to_drop.append("time")
    X_train = train.drop(columns=columns_to_drop)
    y_train = train["label"]
    #if model_name == "isolationforest" or model_name == "oneclasssvm":
        #y_train = [-1 if x == 1 else 1 for x in y_train]

    X_train = scale_X_test(X_train)

    if model_name == "isolationforest":
        model = IsolationForest(random_state=random_state, n_jobs=-1)
        param_grid = {
            "n_estimators": [50, 100, 150, 200],
            "max_samples": [0.5, 0.75, 1.0],
            "contamination": [0.01, 0.05, 0.1, 0.25],
            "max_features": [0.5, 0.75, 1.0],
            "bootstrap": [True],
        }
    elif model_name == "randomforestclassifier":
        model = RandomForestClassifier(random_state=random_state)
        param_grid = {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [5, 7, 10, 15],
        }
    elif model_name == "xgboostclassifier":
        model = xgb.XGBClassifier(random_state=random_state)
        param_grid = {
            "eval_metric": ["aucpr"],
            "learning_rate": [0.1, 0.3, 0.5],
            "max_depth": [4, 6, 8, 10],
            "num_parallel_tree": [1, 2, 3],
            "subsample": [0.5, 1],
            "colsample_bytree": [1],
            "reg_lambda": [0.5, 1],
            "objective": ["binary:logistic"],
            "tree_method": ["hist"],
        }
    elif model_name == "gaussianmixture":
        model = GaussianMixture(random_state=random_state)
        param_grid = {
            "n_components": [1, 2, 3, 4, 8],
            "covariance_type": ["full", "spherical"],
            "init_params": ["kmeans", "random"],
            "max_iter": [50, 100, 150],
            "reg_covar": [0.000001, 0.0001, 0.01, 0.1],
        }
    elif model_name == "oneclasssvm":
        model = SGDOneClassSVM()
        param_grid = {
            "nu": [0.01, 0.05, 0.1, 0.25],
            "learning_rate": ["optimal", "adaptive", "constant", "invscaling"],
            "eta0": [0.1, 0.3, 0.5],
            "max_iter": [500, 1000, 1500],
        }
    else:
        model = IsolationForest(random_state=random_state, n_jobs=-1)
        param_grid = {}

    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        #scoring="average_precision",
        scoring="accuracy",
        n_jobs=-1,
    )

    logging.debug(f"Training a {model_name} model...")
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logging.debug(f"Best parameters for {model_name} found:")
    logging.debug(best_params)
    logging.debug(f"Best score for {model_name} found:")
    logging.debug(best_score)
    
    return grid_search.cv_results_


def grid_search_local():
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
        
    # Load hospitalids
    hospitalids = np.load(os.path.join(path["splits"], "hospitalids.npy"))
    random_hospitalids = np.random.choice(hospitalids, size=10, replace=False)
    
    # Randomly select hospitals
    random_state = 0
    np.random.seed(random_state) 
    random_hospitalids = np.random.choice(hospitalids, size=10, replace=False)
    
    params_list = []
    mean_test_score_list = [[] for _ in range(10)]
    
    for hospitalid_idx, hospitalid in enumerate(random_hospitalids):
        data = load_parquet(
            os.path.join(path["splits"], "individual_hospital_splits"),
            f"hospital{hospitalid}_rstate{random_state}_train.parquet",
            optimize_memory=True,
        )

        cv_results = single_run(
            model_name,
            data,
            config_settings["test_size"],
            random_state,
            config_settings["training_columns_to_drop"],
        )
        
        for mean_test_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
            if hospitalid_idx == 0:
                params_list.append(params)
            mean_test_score_list[hospitalid_idx].append(mean_test_score)
            
    transposed_data = list(map(list, zip(*mean_test_score_list)))
    average_scores = [sum(column) / len(column) for column in transposed_data]
    logging.debug("Average scores:")
    logging.debug(average_scores)
    
    max_score = max(average_scores)
    max_index = average_scores.index(max_score)
    best_params = params_list[max_index]
    
    logging.debug("Max Score:")
    logging.debug(max_score)
    logging.debug("Best Params:")
    logging.debug(best_params)


if __name__ == "__main__":
    # Initialize the logging capability
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    grid_search_local()
