These notebooks can be used to:

1. Explore the original cohort files
    - Requirement: This notebook assumes that the raw eICU `.csv` files are available in the `data/raw/eICU` directory and that the cohort `.parquet` files are available in `data/cohorts/sepsis_eicu_robin`.

2. Explore the value distribution and missingness of the original concepts
    - Requirement: This notebook assumes that the `combined.parquet` cohort data file is available in `data/interim/combined.parquet`.

3. Explore the value distribution and missingness of the generated features
    - Requirement: This notebook assumes that the `features.parquet` cohort data file is available in `data/interim/features.parquet`.