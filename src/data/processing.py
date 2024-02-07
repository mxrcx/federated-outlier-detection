import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def extend_static_data(static_data, raw_patient_data):
    """
    Extend the static data with additional features.

    Args:
        eICU_cohort_static_data (pandas.DataFrame): The dataframe with static cohort data.

    Returns:
        pandas.DataFrame: A dataframe containing the extended static cohort data.
    """
    extended_data = pd.merge(
        left=static_data,
        right=raw_patient_data.loc[
            :,
            [
                "patientunitstayid",
                "ethnicity",
                "hospitalid",
                "unittype",
                "hospitaladmitoffset",
                "uniquepid",
            ],
        ],
        left_on="stay_id",
        right_on="patientunitstayid",
        how="inner",
    )
    extended_data = extended_data.drop(columns=["patientunitstayid"])
    return extended_data


def merge_cohort_data(
    eICU_cohort_static_data, eICU_cohort_dynamic_data, eICU_cohort_outcome_data
):
    """
    Merge static, dynamic and outcome dataframes into one dataframe.

    Args:
        eICU_cohort_static_data (pandas.DataFrame): The dataframe with static cohort data.
        eICU_cohort_dynamic_data (pandas.DataFrame): The dataframe with dynamic cohort data.
        eICU_cohort_outcome_data (pandas.DataFrame): The dataframe with outcome cohort data.

    Returns:
        pandas.DataFrame: A dataframe containing the combined cohort data.
    """
    eICU_cohort_static_and_dynamic_data = pd.merge(
        eICU_cohort_dynamic_data, eICU_cohort_static_data, on="stay_id", how="left"
    )
    eICU_cohort_complete_data = eICU_cohort_static_and_dynamic_data.join(
        eICU_cohort_outcome_data["label"]
    )
    return eICU_cohort_complete_data


def encode_categorical_columns(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    """
    Encode categorical columns in the dataframe. Exclude columns in columns_to_drop.

    Args:
        df (pd.DataFrame): The input dataframe.
        columns_to_drop (list[str]): The columns to drop from the dataframe.

    Returns:
        pd.DataFrame: The dataframe with encoded categorical columns.
    """
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in columns_to_drop]

    # Encode the categorical columns
    encoder = OneHotEncoder()
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    encoded_col_names = [f"{col}_{category}" for i,col in enumerate(categorical_cols) for category in encoder.categories_[i]]
    encoded_df = pd.DataFrame(
        encoded_cols.toarray(), columns=encoded_col_names
    )

    # Drop categorical columns & concatenate the original dataframe with the encoded columns
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df


def drop_cols_with_all_missing(X_train, X_test):
    """
    Drop columns with all missing values from both X_train and X_test.

    Args:
        X_train (pd.DataFrame): The training data.
        X_test (pd.DataFrame): The test data.

    Returns:
        pd.DataFrame: The training data with columns dropped.
        pd.DataFrame: The test data with columns dropped.
    """
    cols_to_drop = set(X_train.columns[X_train.isnull().all()]) | set(
        X_test.columns[X_test.isnull().all()]
    )
    X_train.drop(cols_to_drop, axis=1, inplace=True)
    X_test.drop(cols_to_drop, axis=1, inplace=True)

    return X_train, X_test


def init_imputer(df: pd.DataFrame) -> ColumnTransformer:
    """
    Initializes the imputer.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        ColumnTransformer: An imputer configured to handle numeric and categorical features.
    """
    # Define the columns for each data type
    numeric_features = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Create transformers for imputation and scaling
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="mean"),
            ),  # Impute missing values with the mean
            ("scaler", StandardScaler()),  # Standardize the data
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),  # Impute missing values with the most frequent category
            # ("onehot", OneHotEncoder()),  # One-hot encode categorical data (Already done in preprocessing step)
        ]
    )

    # Combine transformers using ColumnTransformer
    imputer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return imputer


def reformat_time_column(data):
    """
    Reformat the 'time' column from timedelta to total seconds.

    Args:
        data (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The dataframe with the reformatted 'time' column.
    """
    if "time" in data.columns:
        data["time"] = data["time"].dt.total_seconds()
    return data


