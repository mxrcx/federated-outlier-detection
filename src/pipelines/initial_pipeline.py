import os
import yaml

import recipys.recipe, recipys.ingredients, recipys.step, recipys.selector

from data.loading import load_csv, load_cohort_parquet_files, load_parquet
from data.processing import extend_static_data, merge_cohort_data
from data.saving import save_parquet, copy_parquet

# Load the YAML configuration file
with open('configuration.yml') as f:
    config = yaml.safe_load(f)

# Initial configuration
path_to_raw_eicu_data = config['path']['raw_eicu_data']
path_to_original_cohorts = config['path']['original_cohorts']
path_to_cohorts = config['path']['cohorts']
path_to_interim = config['path']['interim']
path_to_features = config['path']['features']
filename_combined = config['filename']['combined']
filename_features = config['filename']['features']

# Extend cohort data with additional concepts
if not os.path.exists(
    os.path.join(path_to_cohorts, "sta.parquet")
    or os.path.join(path_to_cohorts, "dyn.parquet")
    or os.path.join(path_to_cohorts, "outc.parquet")
):
    print("Extending cohort data...", end="", flush=True)
    raw_patient_data = load_csv(path_to_raw_eicu_data, "patient.csv")
    static_data, dynamic_data, outcome_data = load_cohort_parquet_files(
        path_to_original_cohorts
    )
    extended_static_data = extend_static_data(static_data, raw_patient_data)
    save_parquet(extended_static_data, path_to_cohorts, "sta.parquet")
    copy_parquet(path_to_original_cohorts, path_to_cohorts, "dyn.parquet")
    copy_parquet(path_to_original_cohorts, path_to_cohorts, "outc.parquet")
    print("DONE.")
else:
    print("Cohort data already extended.")

# Load the cohort data & merge it if it hasn't been merged yet
if not os.path.exists(os.path.join(path_to_interim, filename_combined)):
    print("Merging cohort data...", end="", flush=True)
    static_data, dynamic_data, outcome_data = load_cohort_parquet_files(path_to_cohorts)
    eICU_cohort_data = merge_cohort_data(static_data, dynamic_data, outcome_data)
    save_parquet(eICU_cohort_data, path_to_interim, filename_combined)
    print("DONE.")
else:
    print("Cohort data already merged.")
    eICU_cohort_data = load_parquet(path_to_interim, "combined.parquet")

# Feature extraction
if not os.path.exists(os.path.join(path_to_features, filename_features)):
    print("Extracting features...", end="", flush=True)
    sepsis_recipe = recipys.recipe.Recipe(
        data=eICU_cohort_data,
        predictors=eICU_cohort_data.drop(
            columns=[
                "stay_id",
                "time",
                "age",
                "sex",
                "height",
                "weight",
                "label",
                "ethnicity",
                "hospitalid",
                "unittype",
                "hospitaladmitoffset",
                "uniquepid",
            ]
        ).columns.tolist(),
    )
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
    eICU_cohort_data_with_features = sepsis_recipe.bake()
    print("DONE.")

    # Keep only the features & relevant columns
    print("Filter and save feature data...", end="", flush=True)
    columns_to_keep = [
        "stay_id",
        "time",
        "age",
        "sex",
        "height",
        "weight",
        "label",
        "ethnicity",
        "hospitalid",
        "unittype",
        "hospitaladmitoffset",
        "uniquepid",
    ] + eICU_cohort_data_with_features.filter(
        regex="_mean$|_max$|_min$|_var$"
    ).columns.tolist()

    eICU_cohort_data_only_features = eICU_cohort_data_with_features[columns_to_keep]
    save_parquet(eICU_cohort_data_only_features, path_to_features, filename_features)
    print("DONE.")
else:
    print("Features already extracted.")
    eICU_cohort_data_only_features = load_parquet(path_to_features, "features.parquet")

print(
    "Initial pipeline completed. Shape of outcome data: ",
    eICU_cohort_data_only_features.shape,
)
