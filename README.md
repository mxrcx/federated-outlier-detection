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
Before running any script or pipeline, make sure to have the environment activated (`conda activate fedout-det`).
- Run a pipeline or script, e.g. with: `python3 src/data/feature_extraction.py`


## Scripts
Here is a list of scripts that can be executed and the order in which they should be run. However, the order is just a suggestion since every pipeline checks if the requirements (e.g. extracted features) are met, and if not, tries to take care of missing steps.

1. `src/data/feature_extraction.py`: Loading the cohort data, merging and extracting features.
2. `src/pipelines/cml_random_split.py`: Running a classical machine learning pipeline with random data split over the complete data from all hospitals.
3. `src/pipelines/cml_loho_split.py`: Running a classical machine learning pipeline with 'LeaveOneHospitalOut' data split, where in each run data from one hopsital is excluded from the training data and used as test data.
4. `src/pipelines/cml_local.py`: Running a classical machine learning pipeline on each hospital's data individually, splitting data on a patient-stay level.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. We do not own any of the datasets used or included in this repository.