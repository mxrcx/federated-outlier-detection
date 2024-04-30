import pandas as pd
import numpy as np
import sys
import re
from data.loading import load_csv, load_configuration
from data.saving import save_csv
from training.preparation import reformatting_model_name


def load_csv_files(model_name):
    """Load CSV files with avg metrics

    Args:
        model_name (str): Model name

    Returns:
        pd.DataFrame: Centralized learning metrics avg dataframe
        pd.DataFrame: Local learning metrics avg dataframe
        pd.DataFrame: Federated learning metrics avg dataframe
    """
    results_centralized_avg = load_csv(
        "../../results", f"centralized_{model_name}_metrics_avg.csv"
    )
    results_local_avg = load_csv("../../results", f"local_{model_name}_metrics_avg.csv")
    results_federated_avg = load_csv(
        "../../results", f"federated_{model_name}_metrics_avg.csv"
    )

    return results_centralized_avg, results_local_avg, results_federated_avg


def combine_results(results_logo_avg, results_local_avg, results_federated_avg):
    """Combine result dataframes

    Args:
        results_logo_avg (pd.DataFrame): Leave-one-group-out avg dataframe
        results_local_avg (pd.DataFrame): Local learning avg dataframe
        results_federated_avg (pd.DataFrame): Federated XGBoost avg dataframe

    Returns:
        pd.DataFrame: Combined dataframe
    """
    combined_df = results_logo_avg.join(
        results_local_avg, lsuffix="_logo", rsuffix="_local"
    )
    results_federated_avg = results_federated_avg.add_suffix("_fed")
    combined_df = combined_df.join(results_federated_avg)

    combined_df = combined_df.drop(columns=["Hospitalid Mean_local"])
    combined_df = combined_df.drop(columns=["Hospitalid Mean_fed"])
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
    """Calculate differences between metrics on non NaN values

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
            (~combined_df[f"{org_colname} Mean_logo"].isna())
            & (~combined_df[f"{org_colname} Mean_local"].isna())
            & (~combined_df[f"{org_colname} Mean_fed"].isna())
        ]
        combined_df[f"{diff_colname} Difference LOGO-Local"] = (
            numeric_cases[f"{org_colname} Mean_logo"]
            - numeric_cases[f"{org_colname} Mean_local"]
        )
        combined_df[f"{diff_colname} Difference LOGO-Fed"] = (
            numeric_cases[f"{org_colname} Mean_logo"]
            - numeric_cases[f"{org_colname} Mean_fed"]
        )
        combined_df[f"{diff_colname} Difference Local-Fed"] = (
            numeric_cases[f"{org_colname} Mean_local"]
            - numeric_cases[f"{org_colname} Mean_fed"]
        )
    return combined_df


def create_average_diff(combined_df):
    """Create average difference dataframe

    Args:
        combined_df (pd.DataFrame): Combined dataframe

    Returns:
        pd.DataFrame: Average difference dataframe
    """
    # Exclude the last two rows with total values
    combined_df_pruned = combined_df.iloc[:-2]

    average_diff_colnames = [
        "Accuracy",
        "AUROC",
        "AUPRC",
        "FP",
        "FP Stay IDs",
        "FN",
        "FN Stay IDs",
    ]
    average_diff_data = {}
    comparison_suffixes = ["LOGO-Local", "LOGO-Fed", "Local-Fed"]

    for colname in average_diff_colnames:
        for comparison_suffix in comparison_suffixes:
            average_diff_data[f"{colname} Difference {comparison_suffix} Mean"] = [
                combined_df_pruned[
                    f"{colname} Difference {comparison_suffix}"
                ].mean()
            ]
            average_diff_data[f"{colname} Difference {comparison_suffix}  Std"] = [
                combined_df_pruned[
                    f"{colname} Difference {comparison_suffix}"
                ].std()
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
    df["Confusion Matrix Mean_logo"] = df["Confusion Matrix Mean_logo"].apply(
        convert_cm_string
    )
    df["Confusion Matrix Mean_local"] = df["Confusion Matrix Mean_local"].apply(
        convert_cm_string
    )
    df["Confusion Matrix Mean_fed"] = df["Confusion Matrix Mean_fed"].apply(
        convert_cm_string
    )

    return df


def analysis():
    _path, _filename, config_settings = load_configuration()
    
    # Get model_name
    if len(sys.argv) > 1:
        model_name = reformatting_model_name(sys.argv[1])
    else:
        model_name = reformatting_model_name(config_settings["model"])

    # Step 1 - Load data
    results_centralized_avg, results_local_avg, results_federated_avg = load_csv_files(
        model_name
    )

    # Step 2 - Combine data
    combined_df = combine_results(
        results_centralized_avg, results_local_avg, results_federated_avg
    )

    # Step 3 - Processing
    combined_df = replace_string_with_nan(
        combined_df,
        [
            "Hospitalid Mean",
            # "Confusion Matrix Mean_logo",
            # "Confusion Matrix Mean_local",
            # "Confusion Matrix Mean_fed",
        ],
    )
    # combined_df = convert_cm_columns(combined_df)

    # Step 4 - Get individual differences
    combined_df = calculate_differences(combined_df)
    save_csv(combined_df, "../../results", f"analysis_{model_name}_diff.csv")

    # Step 5 - Get average differences
    average_diff = create_average_diff(combined_df)
    save_csv(average_diff, "../../results", f"analysis_{model_name}_diff_avg.csv")


if __name__ == "__main__":
    analysis()
