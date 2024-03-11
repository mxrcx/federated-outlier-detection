import os
import yaml
import logging
from tqdm import tqdm

from pyarrow import parquet as pq
import pandas as pd


def load_configuration():
    """
    Load the configuration file and return the path and filename dictionaries.

    Returns:
        dict: The path dictionary.
        dict: The filename dictionary.
    """
    with open("../../config/configuration.yml") as f:
        config = yaml.safe_load(f)

    path = {
        key: config["path"][key]
        for key in [
            "cohorts",
            "interim",
            "features",
            "raw_data",
            "original_cohorts",
            "results",
        ]
    }
    filename = {
        key: config["filename"][key]
        for key in [
            "raw_patient_data",
            "static_data",
            "dynamic_data",
            "outcome_data",
            "combined",
            "features",
        ]
    }
    config_settings = {
        key: config["config_settings"][key]
        for key in [
            "random_split_reps",
            "cv_folds",
            "test_size",
            "missingness_cutoff",
            "feature_extraction_columns_to_exclude",
            "training_columns_to_drop",
        ]
    }

    return path, filename, config_settings


def load_csv(path, filename):
    """
    Load a dataframe from a csv file.

    Args:
        path (str): The path to the directory containing the csv file.
        filename (str): The name of the csv file.

    Returns:
        pandas.DataFrame: The loaded dataframe from csv.
    """
    return pd.read_csv(os.path.join(path, filename))


def load_cohort_parquet_files(
    path, filename_static, filename_dynamic, filename_outcome
):
    """
    Load static, dynamic and outcome dataframes from parquet files.

    Args:
        path (str): The path to the directory containing the parquet files.
        filename_static (str): The name of the static parquet file.
        filename_dynamic (str): The name of the dynamic parquet file.
        filename_outcome (str): The name of the outcome parquet file.

    Returns:
        pandas.DataFrame: The dataframe with static cohort data.
        pandas.DataFrame: The dataframe with dynamic cohort data.
        pandas.DataFrame: The dataframe with outcome cohort data.
    """
    eICU_cohort_static_data = pq.read_table(
        os.path.join(path, filename_static)
    ).to_pandas()
    eICU_cohort_dynamic_data = pq.read_table(
        os.path.join(path, filename_dynamic)
    ).to_pandas()
    eICU_cohort_outcome_data = pq.read_table(
        os.path.join(path, filename_outcome)
    ).to_pandas()
    return eICU_cohort_static_data, eICU_cohort_dynamic_data, eICU_cohort_outcome_data


def _optimize_dataframe_footprint(df: pd.DataFrame):
    """
    Safely downcast a pandas DataFrame to reduce memory usage.

    Parameters:
        df (pd.DataFrame): The DataFrame to downcast.

    Returns:
        df (pd.DataFrame): The safely downcasted DataFrame.
    """

    logging.debug("Optimizing dataframe memory footprint...")

    # Copy the dataframe to avoid changing the original one
    df_downcast = df.copy()

    # Downcast numeric columns to smallest type
    for col in tqdm(df_downcast.select_dtypes(include=['int', 'float']).columns):
        df_downcast[col] = pd.to_numeric(df_downcast[col], downcast='integer', errors='ignore')
        df_downcast[col] = pd.to_numeric(df_downcast[col], downcast='float', errors='ignore')

    return df_downcast

def load_parquet(path, filename, optimize_memory = False):
    """
    Load data from a parquet file.

    Args:
        path (str): The path to the directory containing the parquet file.
        filename (str): The name of the parquet file.
        optimize_memory (bool): Whether to optimize the memory of the pd.DataFrame by changing the column types

    Returns:
        pandas.DataFrame: The loaded dataframe.
    """
    table = pq.read_table(os.path.join(path, filename)).to_pandas()

    if optimize_memory:
        table = _optimize_dataframe_footprint(table)

    return table