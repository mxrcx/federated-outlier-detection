import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def split_data_on_stay_ids(data, test_size, random_state):
    """
    Split data based on stay_ids.

    Args:
        data (pd.DataFrame): The data to be split.
        random_state (int): The random state to be used.

    Returns:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The test data.
    """
    # Get unique stay_ids
    stay_ids = data["stay_id"].unique()

    # Initialize an empty list to store labels
    labels = []

    # Iterate through unique stay_ids
    for stay_id in stay_ids:
        # Check if there is at least one "True" label associated with the stay_id
        if any(data[data["stay_id"] == stay_id]["label"]):
            labels.append(True)
        else:
            labels.append(False)

    # Convert labels to numpy array
    labels = np.array(labels)

    # Initialize StratifiedShuffleSplit
    stratified_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )

    try:
        # Split data based on labels
        for train_index, test_index in stratified_splitter.split(stay_ids, labels):
            train_stay_ids = stay_ids[train_index]
            test_stay_ids = stay_ids[test_index]
    except ValueError:
        # If there is only one label:
        train_stay_ids, test_stay_ids = train_test_split(
            stay_ids, test_size=0.2, random_state=random_state
        )

    # Create training and test datasets
    train = data[data["stay_id"].isin(train_stay_ids)]
    test = data[data["stay_id"].isin(test_stay_ids)]

    return train, test
