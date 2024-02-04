import os

def save_parquet(df, output_dir, filename):
    """
    Save dataframe to a parquet file.

    Args:
        df (pandas.DataFrame): The dataframe to be saved.
        output_dir (str): The path to the directory to save the parquet file.
        filename (str): The name of the parquet file.
        
    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_parquet(os.path.join(output_dir, filename))
    
def copy_parquet(input_dir, output_dir, filename):
    """
    Copy parquet file to a new directory.

    Args:
        input_dir (str): The path to the directory containing the parquet file.
        output_dir (str): The path to the directory to save the parquet file.
        filename (str): The name of the parquet file.
        
    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.system(f"cp {os.path.join(input_dir, filename)} {os.path.join(output_dir, filename)}")
