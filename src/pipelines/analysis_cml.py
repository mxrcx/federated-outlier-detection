import pandas as pd
import numpy as np
import re
from data.loading import load_csv
from data.saving import save_csv


def load_csv_files():
    """Load LOHO and local analysis dataframes

    Returns:
        pd.DataFrame: LOHO avg dataframe
        pd.DataFrame: Local avg dataframe
    """
    data_loho_avg = load_csv("../../results", "cml_loho_split_metrics_avg.csv")
    data_local_avg = load_csv("../../results", "cml_local_metrics_avg.csv")

    return data_loho_avg, data_local_avg


def combine_loho_local(data_loho_avg, data_local_avg):
    """Combine LOHO and local analysis dataframes

    Args:
        data_loho_avg (pd.DataFrame): LOHO avg dataframe
        data_local_avg (pd.DataFrame): Local avg dataframe

    Returns:
        pd.DataFrame: Combined dataframe
    """
    combined_df = data_loho_avg.join(data_local_avg, lsuffix="_loho", rsuffix="_local")
    combined_df = combined_df.drop(columns=["Hospitalid Mean_local"])
    combined_df = combined_df.rename(
        columns={
            "Hospitalid Mean_loho": "Hospitalid Mean",
        }
    )

    return combined_df


def replace_string_with_nan(df, exclude_columns=[]):
    """Replace string values with NaN

    Args:
        df (pd.DataFrame): Dataframe to process
        exclude_columns (list): Columns to exclude from processing

    Returns:
        pd.DataFrame: Processed dataframe
    """
    columns_to_process = df.columns.difference(exclude_columns)
    df[columns_to_process] = df[columns_to_process].map(
        lambda x: pd.to_numeric(x, errors="coerce") if isinstance(x, str) else x
    )

    return df


def calculate_differences(combined_df):
    """Calculate differences between LOHO and local metrics on non NaN values

    Args:
        combined_df (pd.DataFrame): Combined dataframe

    Returns:
        pd.DataFrame: Processed dataframe
    """
    difference_colnames = [
        ("Accuracy", "Accuracy"),
        ("AUROC", "AUROC"),
        ("AUPRC", "AUPRC"),
        ("True Negatives", "TN"),
        ("Stay IDs with True Negatives", "TN Stay IDs"),
        ("False Positives", "FP"),
        ("Stay IDs with False Positives", "FP Stay IDs"),
        ("False Negatives", "FN"),
        ("Stay IDs with False Negatives", "FN Stay IDs"),
        ("True Positives", "TP"),
        ("Stay IDs with True Positives", "TP Stay IDs"),
    ]
    for org_colname, diff_colname in difference_colnames:
        numeric_cases = combined_df[
            (~combined_df[f"{org_colname} Mean_loho"].isna())
            & (~combined_df[f"{org_colname} Mean_local"].isna())
        ]
        combined_df[f"{diff_colname} Difference"] = (
            numeric_cases[f"{org_colname} Mean_loho"]
            - numeric_cases[f"{org_colname} Mean_local"]
        )
    return combined_df


def create_average_diff(combined_df):
    """Create average difference dataframe

    Args:
        combined_df (pd.DataFrame): Combined dataframe

    Returns:
        pd.DataFrame: Average difference dataframe
    """
    # Exclude the last row with total values
    combined_df_except_last_row = combined_df.iloc[:-1]

    average_diff_colnames = [
        "Accuracy",
        "AUROC",
        "AUPRC",
        "TN",
        "TN Stay IDs",
        "FP",
        "FP Stay IDs",
        "FN",
        "FN Stay IDs",
        "TP",
        "TP Stay IDs",
    ]
    average_diff_data = {}

    for colname in average_diff_colnames:
        average_diff_data[f"{colname} Difference Mean"] = [
            combined_df_except_last_row[f"{colname} Difference"].mean()
        ]
        average_diff_data[f"{colname} Difference Std"] = [
            combined_df_except_last_row[f"{colname} Difference"].std()
        ]

    average_diff = pd.DataFrame(average_diff_data)
    return average_diff


def convert_cm_string(string):
    """Convert confusion matrix string to NumPy array

    Args:
        string (str): Confusion matrix string

    Returns:
        np.array: Confusion matrix as NumPy array
    """
    numbers = re.findall(r"\d+", string)
    numbers = [int(num) for num in numbers]
    if len(numbers) == 1:
        numbers = [numbers[0], 0, 0, 0]
    matrix = np.array(numbers).reshape((2, 2))
    return matrix


def convert_cm_columns(df):
    """Convert confusion matrix string columns to NumPy array columns

    Args:
        df (pd.DataFrame): Dataframe to process

    Returns:
        pd.DataFrame: Processed dataframe
    """
    df["Confusion Matrix Mean_loho"] = df["Confusion Matrix Mean_loho"].apply(
        convert_cm_string
    )
    df["Confusion Matrix Mean_local"] = df["Confusion Matrix Mean_local"].apply(
        convert_cm_string
    )

    return df


def cml_analysis():
    """Perform CML analysis"""
    # Step 1 - Load data
    analysis_loho_avg, analysis_local_avg = load_csv_files()

    # Step 2 - Combine data
    combined_df = combine_loho_local(analysis_loho_avg, analysis_local_avg)

    # Step 3 - Processing
    combined_df = replace_string_with_nan(
        combined_df,
        [
            "Hospitalid Mean",
            "Confusion Matrix Mean_loho",
            "Confusion Matrix Mean_local",
        ],
    )
    combined_df = convert_cm_columns(combined_df)

    # Step 4 - Get individual differences
    combined_df = calculate_differences(combined_df)
    save_csv(combined_df, "../../results", "cml_comparison_loho_local.csv")

    # Step 5 - Get average differences
    average_diff = create_average_diff(combined_df)
    save_csv(average_diff, "../../results", "cml_comparison_loho_local_avg_diff.csv")


if __name__ == "__main__":
    cml_analysis()
