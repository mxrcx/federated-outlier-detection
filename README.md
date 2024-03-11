# Benchmarking Federated Outlier Detection Methods on EHR Data

A repository for code developed as part of the "Benchmarking Federated Outlier Detection Methods on EHR Data" master's thesis by Marco Schaarschmidt.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Scripts](#scripts)
- [Contributing](#contributing)
- [License](#license)

## Introduction
In this project we want to analyze whether it is benefical to 
- (1) treat `Sepsis` detection in ICU units as an outlier detection problem compared to classical machine learning methods.
- (2) use `Federated Machine Learning` to incorporate data from multiple hospitals compared to local machine learning on the hospital's own data.

## Getting Started
To set up the project locally, follow these steps:
- Clone the `federated-outlier-detection` repository.
- Install `conda`, if not installed already.
- Open a terminal and navigate inside the `federated-outlier-detection` repository.
- Create the conda environment with: `conda env create -f environment.yml`
- Activate the conda environment with: `conda activate fedout-det`

## Usage
Change configuration settings (e.g. directory paths, filenames, ...) in the `configuration.yml` file within the `config` subfolder of the repository.

List of available configuration settings:
```
    path:
        raw_data: path to files downloaded and extracted from PhysioNet
        original_cohorts: path to cohort parquet files created with yaib-cohorts
        cohorts: path to directory where the final cohort files should be saved
        interim: path to directory where iterim versions of the data should be saved
        features: path to directory where the extracted features should be saved
        results: path_to_directory where results should be saved

    filename:
        raw_patient_data: name of the eICU CSV file that contains the patient table
        static_data: name of cohort parquet file containing the static data
        dynamic_data: name of cohort parquet file containing the dynamic data
        outcome_data: name of cohort parquet file containing the label data
        combined: name of parquet file where the combined static, dynamic and label data should be saved
        features: name of parquet file where extracted features should be saved

    config_settings:
        random_split_reps: number of experiment repetitions
        test_size: test size for the train-test split
        feature_extraction_columns_to_exclude: columns which should be excluded from the feature extraction
        training_columns_to_drop: columns to drop during training
```

Before running any script or pipeline, make sure to have the environment activated (`conda activate fedout-det`).
- Run a pipeline or script, e.g. with: `python3 src/data/feature_extraction.py`


## Scripts
Here is a list of scripts that can be executed and the order in which they should be run. However, the order is just a suggestion since every pipeline checks if the requirements (e.g. extracted features) are met, and if not, tries to take care of missing steps.

**Note:** Pipeline scripts and SBATCH jobs should be executed from the `src/pipelines` directory.

1. `src/data/feature_extraction.py`: Loading the cohort data, merging and extracting features.
2. `src/pipelines/cml_random_split.py`: Running a classical machine learning pipeline with random data split over the complete data from all hospitals.
3. `src/pipelines/cml_loho_split.py`: Running a classical machine learning pipeline with 'LeaveOneHospitalOut' data split, where in each run data from one hopsital is excluded from the training data and used as test data.
4. `src/pipelines/cml_local.py`: Running a classical machine learning pipeline on each hospital's data individually, splitting data on a patient-stay level.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. We do not own any of the datasets used or included in this repository.