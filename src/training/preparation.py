import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def split_data_on_stay_ids(data: pd.DataFrame, test_size: float, random_state: int):
    """
    Split data based on stay_ids.

    Args:
        data (pd.DataFrame): The data to be split.
        test_size (float): The percentage of the data to be allocated as a test set.
        random_state (int): The random state to be used.

    Returns:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The test data.
    """

    # Initialize an empty list to store labels
    stay_ids_with_labels = data[["stay_id", "label"]].groupby(by="stay_id").sum()

    stay_ids = stay_ids_with_labels.index
    labels = stay_ids_with_labels.iloc[:, 0].apply(lambda x: 1 if x >= 1 else 0)

    # Convert labels to numpy array
    labels = labels.values

    if labels.sum() > 1:
        train_stay_ids, test_stay_ids = train_test_split(
            stay_ids,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=labels,
        )
    else:
        train_stay_ids, test_stay_ids = train_test_split(
            stay_ids, test_size=test_size, random_state=random_state, shuffle=True
        )

    # Create training and test datasets
    train = data[data["stay_id"].isin(train_stay_ids)]
    test = data[data["stay_id"].isin(test_stay_ids)]

    return train, test


def reformatting_model_name(model_name):
    """Reformat the model name to a standard format.

    Args:
        model_name (str): The model name.

    Returns:
        str: The reformatted model name.
    """
    model_name = model_name.lower().replace(" ", "")
    if model_name in ["randomforestclassifier", "randomforest", "rf"]:
        return "randomforestclassifier"
    elif model_name in ["xgboostclassifier", "xgboost"]:
        return "xgboostclassifier"
    elif model_name in ["isolationforest", "if"]:
        return "isolationforest"
    elif model_name in ["gaussianmixture", "gaussianmixturemodel", "gm", "gmm"]:
        return "gaussianmixture"
    elif model_name in ["oneclasssvm", "ocsvm"]:
        return "oneclasssvm"
    else:
        return model_name


def get_model(model_name, random_state, n_jobs):
    """
    Get a model object based on the model name.

    Args:
        model_name (str): The name of the model.
        random_state (int): The random state to be used.
        n_jobs (int): The number of jobs to run in parallel.

    Returns:
        model: The model object.
    """
    model_name = reformatting_model_name(model_name)

    if model_name == "randomforestclassifier":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=150, # 150
            max_depth=15,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif model_name == "xgboostclassifier":
        import xgboost as xgb
        
        return xgb.XGBClassifier(
            eval_metric="aucpr",
            learning_rate=0.3,
            max_depth=10,
            num_parallel_tree=3,
            subsample=0.5,
            colsample_bytree=1,
            reg_lambda=1,
            objective="binary:logistic",
            tree_method="hist",
        )
    elif model_name == "isolationforest":
        from sklearn.ensemble import IsolationForest

        return IsolationForest(
            bootstrap=True,
            contamination=0.01,
            max_features=0.75,
            max_samples=0.5,
            n_estimators=100,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif model_name == "gaussianmixture":
        from sklearn.mixture import GaussianMixture

        return GaussianMixture(
            covariance_type="full",
            init_params="random",
            max_iter=75,
            n_components=2,
            reg_covar=0.1,
            random_state=random_state,
        )
    elif model_name == "oneclasssvm":
        from sklearn.linear_model import SGDOneClassSVM

        return SGDOneClassSVM(eta0=0.3, learning_rate="constant", max_iter=1500, nu=0.25)
    else:
        raise ValueError(
            "Invalid model name. Specifiy a different model in the configuration file, choose from: 'randomforestclassifier', 'xgboostclassifier', 'isolationforest', 'gaussianmixture', 'oneclasssvm'"
        )
