import os
import yaml
import sys

import recipys.recipe, recipys.ingredients, recipys.step, recipys.selector

# append the path of the parent directory
sys.path.append("..")

from data.loading import load_configuration, load_csv, load_cohort_parquet_files, load_parquet
from data.processing import extend_static_data, merge_cohort_data
from data.saving import save_parquet, copy_parquet


def extend_cohort_data(path, filename):
    """
    Extend the static cohort data with additional concepts. Copies the dynamic and outcome data.

    Args:
        path (dict): The path dictionary.
        filename (dict): The filename dictionary.

    Returns:
        None
    """
    raw_patient_data = load_csv(path["raw_data"], filename["raw_patient_data"])
    static_data = load_parquet(path["original_cohorts"], filename["static_data"])
    extended_static_data = extend_static_data(static_data, raw_patient_data)
    save_parquet(extended_static_data, path["cohorts"], filename["static_data"])
    copy_parquet(path["original_cohorts"], path["cohorts"], filename["dynamic_data"])
    copy_parquet(path["original_cohorts"], path["cohorts"], filename["outcome_data"])


def combine_cohort_data(path, filename):
    """
    Combine the static, dynamic and outcome data into one dataframe. Save the combined data as a parquet file.

    Args:
        path (dict): The path dictionary.
        filename (dict): The filename dictionary.

    Returns:
        pandas.DataFrame: The combined cohort data.
    """
    static_data, dynamic_data, outcome_data = load_cohort_parquet_files(path["cohorts"])
    combined_data = merge_cohort_data(static_data, dynamic_data, outcome_data)
    save_parquet(combined_data, path["interim"], filename["combined"])

    return combined_data


def extract_features(data, columns_to_exclude, path, filename):
    """
    Extract features from the cohort data and save the features as a parquet file.

    Args:
        data (pandas.DataFrame): The cohort data.
        columns_to_exclude (list): The columns to exclude.
        path (dict): The path dictionary.
        filename (dict): The filename dictionary.

    Returns:
        None
    """
    sepsis_recipe = recipys.recipe.Recipe(
        data=data,
        predictors=data.drop(columns=columns_to_exclude).columns.tolist(),
    )

    # Add historical features
    sepsis_recipe.add_step(
        recipys.step.StepHistorical(
            sel=recipys.selector.all_numeric_predictors(),
            fun=recipys.step.Accumulator.MEAN,
            role="feature",
        )
    )
    sepsis_recipe.add_step(
        recipys.step.StepHistorical(
            sel=recipys.selector.all_numeric_predictors(),
            fun=recipys.step.Accumulator.MAX,
            role="feature",
        )
    )
    sepsis_recipe.add_step(
        recipys.step.StepHistorical(
            sel=recipys.selector.all_numeric_predictors(),
            fun=recipys.step.Accumulator.MIN,
            role="feature",
        )
    )
    sepsis_recipe.add_step(
        recipys.step.StepHistorical(
            sel=recipys.selector.all_numeric_predictors(),
            fun=recipys.step.Accumulator.VAR,
            role="feature",
        )
    )
    data_with_features = sepsis_recipe.bake()

    # Keep only the features & relevant columns
    columns_to_keep = (
        columns_to_exclude
        + data_with_features.filter(regex="_mean$|_max$|_min$|_var$").columns.tolist()
    )
    data_with_features = data_with_features[columns_to_keep]

    save_parquet(data_with_features, path["features"], filename["features"])


def prepare_cohort_and_extract_features():
    """
    Prepare the cohort data and extract features.

    Returns:
        None
    """
    path, filename, config_settings = load_configuration()
    extend_cohort_data(path, filename)
    data = combine_cohort_data(path, filename)
    extract_features(
        data, config_settings["feature_extraction_columns_to_exclude"], path, filename
    )


if __name__ == "__main__":
    prepare_cohort_and_extract_features()
