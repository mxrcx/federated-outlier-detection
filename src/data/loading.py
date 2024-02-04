import os
from pyarrow import parquet as pq
import pandas as pd

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
    

def load_cohort_parquet_files(path):
    """
    Load static, dynamic and outcome dataframes from parquet files.
    
    Args:
        path (str): The path to the directory containing the parquet files.
        
    Returns:
        pandas.DataFrame: The dataframe with static cohort data.
        pandas.DataFrame: The dataframe with dynamic cohort data.
        pandas.DataFrame: The dataframe with outcome cohort data.
    """
    eICU_cohort_static_data = pq.read_table(os.path.join(path, "sta.parquet")).to_pandas()
    eICU_cohort_dynamic_data = pq.read_table(os.path.join(path, "dyn.parquet")).to_pandas()
    eICU_cohort_outcome_data = pq.read_table(os.path.join(path, "outc.parquet")).to_pandas()
    return eICU_cohort_static_data, eICU_cohort_dynamic_data, eICU_cohort_outcome_data

def load_parquet(path, filename):
    """
    Load data from a parquet file.
    
    Args:
        path (str): The path to the directory containing the parquet file.
        filename (str): The name of the parquet file.
        
    Returns:
        pandas.DataFrame: The loaded dataframe.
    """
    return pq.read_table(os.path.join(path, filename)).to_pandas()
