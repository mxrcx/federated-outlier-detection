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
    labels = stay_ids_with_labels.iloc[:, 0].apply(lambda x: 1 if x>=1 else 0)

    # Convert labels to numpy array
    labels = labels.values

    if labels.sum() > 1:
        train_stay_ids, test_stay_ids = train_test_split(stay_ids, test_size=test_size, random_state=random_state, shuffle=True, stratify=labels)
    else:
        train_stay_ids, test_stay_ids = train_test_split(stay_ids, test_size=test_size, random_state=random_state, shuffle=True)

    # Create training and test datasets
    train = data[data["stay_id"].isin(train_stay_ids)]
    test = data[data["stay_id"].isin(test_stay_ids)]

    return train, test