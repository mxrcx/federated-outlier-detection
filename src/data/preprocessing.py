import os
import logging
import sys

sys.path.append("..")

from data.loading import load_parquet, load_configuration
from data.saving import save_parquet
from data.feature_extraction import prepare_cohort_and_extract_features
from data.processing import reformat_time_column, encode_categorical_columns


def preprocessing():
    logging.debug("Loading configuration...")
    path, filename, config_settings = load_configuration()

    if not os.path.exists(os.path.join(path["features"], filename["features"])):
        logging.info("Preparing cohort and extracting features...")
        prepare_cohort_and_extract_features()

    logging.debug("Loading data with extracted features...")
    data = load_parquet(path["features"], filename["features"], optimize_memory=True)

    logging.info("Perform preprocessing...")
    data = reformat_time_column(data)
    data = encode_categorical_columns(data, config_settings["training_columns_to_drop"])

    # Drop all rows with missing 'label' values
    data = data.dropna(subset=["label"])

    '''
    # Drop columns with missingness above threshold in any hospital
    cols_to_drop = set(
        data.columns[
            data.groupby("hospitalid")
            .apply(lambda x: x.isnull().mean() > config_settings["missingness_cutoff"])
            .any()
        ]
    )
    '''
    # Drop columns with missingness above threshold in the entire dataset
    cols_to_drop = set(data.columns[data.isnull().mean() > config_settings["missingness_cutoff"]])
    
    data.drop(cols_to_drop, axis=1, inplace=True)

    logging.debug("Save preprocessed data to parquet...")
    save_parquet(data, path["processed"], filename["processed"])


if __name__ == "__main__":
    # Initialize the logging capability
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    preprocessing()
