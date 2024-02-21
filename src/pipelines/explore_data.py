import os

import pandas as pd
from pyarrow import parquet as pq

import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path, filename):
    data = pq.read_table(os.path.join(path, filename)).to_pandas()
    return data


def get_missing_percentage(data):
    missing_percentage = data.isnull().mean() * 100
    return missing_percentage


def calculate_missingness(data):
    missing_percentage = get_missing_percentage(data)
    missing_info = pd.DataFrame(
        {
            "Feature": missing_percentage.index,
            "MissingPercentage": missing_percentage.values,
        }
    )
    return missing_info


def get_categorical_features(data):
    categorical_features = data.select_dtypes(include=["object", "category"]).columns
    return categorical_features


def create_plots(
    data, missing_percentage, categorical_features, path_results, filename_results
):
    # Set the number of rows and columns for subplots
    num_features = len(data.columns)
    num_cols = min(num_features, 5)
    num_rows = -(-num_features // num_cols)

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    fig.tight_layout(pad=4.0)

    # Loop through each feature and create boxplots
    for i, feature in enumerate(data.columns):
        if feature == "hospitalid":
            continue

        row_index = i // num_cols
        col_index = i % num_cols

        # Specify the axis for the current subplot
        ax = axes[row_index, col_index] if num_rows > 1 else axes[col_index]

        # Subset data for the current feature
        feature_data = data.loc[:, [feature, "hospitalid"]]

        if feature in categorical_features:
            try:
                sns.countplot(x=feature, hue="hospitalid", data=feature_data, ax=ax)
                ax.set_title(
                    f"{feature} - Missing (accross whole dataset): {missing_percentage[feature]:.2f}%"
                )
                ax.set_xlabel(feature)
                # plt.legend(title='Hospital ID', bbox_to_anchor=(1, 1))
            except Exception as e:
                print(f"Error message for CATEGORICAL feature '{feature}': {e}")
        else:
            try:
                sns.boxplot(x="hospitalid", y=feature, data=feature_data, ax=ax)
                ax.set_title(
                    f"{feature} - Missing (accross whole dataset): {missing_percentage[feature]:.2f}%"
                )
                ax.set_xlabel("Hospital ID")
                ax.set_ylabel(feature)
            except Exception as e:
                print(f"Error message for NUMERICAL feature '{feature}': {e}")

    # Save plot to file
    plt.savefig(os.path.join(path_results, filename_results))


def explore_data(path, filename, path_results, filename_results):
    data = load_data(path, filename)
    missing_percentage = get_missing_percentage(data)
    categorical_features = get_categorical_features(data)
    create_plots(
        data, missing_percentage, categorical_features, path_results, filename_results
    )


def explore_concepts(
    path="../../data/interim",
    filename="combined.parquet",
    path_results="../../results",
    filename_results="concepts_exploration.png",
):
    explore_data(path, filename, path_results, filename_results)


def explore_features(
    path="../../data/processed",
    filename="features.parquet",
    path_results="../../results",
    filename_results="features_exploration.png",
):
    explore_data(path, filename, path_results, filename_results)


if __name__ == "__main__":
    explore_concepts()
    explore_features()
