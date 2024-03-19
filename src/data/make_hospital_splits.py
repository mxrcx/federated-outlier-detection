import os
import logging
import sys
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold


sys.path.append("..")

from data.loading import load_parquet, load_configuration
from data.preprocessing import preprocessing
from training.preparation import split_data_on_stay_ids


def make_splits(data, random_split_reps, test_size):
    train_dfs = []
    test_dfs = []
    hospitalids = data["hospitalid"].unique()
    sepsis_cases = []

    logging.debug(
        "Make individual splits by repeatedly splitting data into a train and test subset for each hospital..."
    )
    for hospitalid in hospitalids:
        data_hospital = data[data["hospitalid"] == hospitalid]

        # Save if there are any sepsis cases in the hospital
        if any(data_hospital["label"]):
            sepsis_cases.append(True)
        else:
            sepsis_cases.append(False)

        for random_state in range(random_split_reps):
            train, test = split_data_on_stay_ids(data_hospital, test_size, random_state)
            train_dfs.append(train)
            test_dfs.append(test)

    return train_dfs, test_dfs, hospitalids, sepsis_cases


def save_splits_to_parquet(train_dfs, test_dfs, hospitalids, output_dir):
    logging.debug("Save individual splits to parquet...")
    hospitalid = -1
    random_state = 0
    for i, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):
        # Reset hospitalid and random_state
        if i % 5 == 0:
            if hospitalid == -1:
                hospitalid = hospitalids[0]
            else:
                hospitalid = hospitalids[np.where(hospitalids == hospitalid)[0][0] + 1]
            random_state = 0
        else:
            random_state += 1

        table_train = pa.Table.from_pandas(train_df)
        train_filename = os.path.join(
            output_dir, f"hospital{hospitalid}_rstate{random_state}_train.parquet"
        )
        pq.write_table(table_train, train_filename)

        table_test = pa.Table.from_pandas(test_df)
        test_filename = os.path.join(
            output_dir, f"hospital{hospitalid}_rstate{random_state}_test.parquet"
        )
        pq.write_table(table_test, test_filename)


def make_stratified_group_splits(
    train_dfs,
    test_dfs,
    hospitalids,
    sepsis_cases,
    random_split_reps,
    n_splits,
):
    logging.debug("Make stratified group splits...")
    group_splits = []
    for random_state in range(random_split_reps):
        positions = range(random_state, len(train_dfs), random_split_reps)
        train_dfs_for_random_state = [train_dfs[pos] for pos in positions]
        test_dfs_for_random_state = [test_dfs[pos] for pos in positions]

        # Create a StratifiedKFold object with n_split folds
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

        # Split hospitalids into 5 folds, stratifying based on sepsis_cases
        folds = []
        for train_index, test_index in skf.split(hospitalids, sepsis_cases):
            train_fold_list = [train_dfs_for_random_state[i] for i in train_index]
            train_fold = pd.concat(train_fold_list, ignore_index=True)

            test_fold_list = [test_dfs_for_random_state[i] for i in test_index]
            test_fold = pd.concat(test_fold_list, ignore_index=True)

            folds.append((train_fold, test_fold))
        group_splits.append(folds)

    return group_splits


def save_stratified_group_splits_to_parquet(
    group_splits, random_split_reps, cv_folds, output_dir
):
    logging.debug("Save stratified group splits to parquet...")
    for random_state in range(random_split_reps):
        for fold in range(cv_folds):
            train_filename = os.path.join(
                output_dir, f"fold{fold}_rstate{random_state}_train.parquet"
            )
            train_df = group_splits[random_state][fold][0]
            train_df.to_parquet(train_filename)
            test_filename = os.path.join(
                output_dir, f"fold{fold}_rstate{random_state}_test.parquet"
            )
            test_df = group_splits[random_state][fold][1]
            test_df.to_parquet(test_filename)


def make_hospital_splits():
    logging.debug("Loading configuration...")
    path, filename, config_settings = load_configuration()

    if not os.path.exists(os.path.join(path["processed"], filename["processed"])):
        logging.info("Preprocess data...")
        preprocessing()

    logging.debug("Loading preprocessed data...")
    data = load_parquet(path["processed"], filename["processed"], optimize_memory=True)

    train_dfs, test_dfs, hospitalids, sepsis_cases = make_splits(
        data, config_settings["random_split_reps"], config_settings["test_size"]
    )

    # Create output directory if necessary
    output_dir_individual = os.path.join(path["splits"], "individual_hospital_splits")
    if not os.path.exists(output_dir_individual):
        os.makedirs(output_dir_individual)

    save_splits_to_parquet(train_dfs, test_dfs, hospitalids, output_dir_individual)

    group_splits = make_stratified_group_splits(
        train_dfs,
        test_dfs,
        hospitalids,
        sepsis_cases,
        config_settings["random_split_reps"],
        config_settings["cv_folds"],
    )

    # Create output directory if necessary
    output_dir_group = os.path.join(path["splits"], "group_hospital_splits")
    if not os.path.exists(output_dir_group):
        os.makedirs(output_dir_group)

    save_stratified_group_splits_to_parquet(
        group_splits,
        config_settings["random_split_reps"],
        config_settings["cv_folds"],
        output_dir_group,
    )

    # Save the hospitalids to a .npy file
    np.save(os.path.join(path["splits"], "hospitalids.npy"), hospitalids)


if __name__ == "__main__":
    # Initialize the logging capability
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    make_hospital_splits()
