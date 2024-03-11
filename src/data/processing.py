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


def encode_categorical_columns(
    df: pd.DataFrame, columns_to_drop: list[str]
) -> pd.DataFrame:
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
    encoded_col_names = [
        f"{col}_{category}"
        for i, col in enumerate(categorical_cols)
        for category in encoder.categories_[i]
    ]
    encoded_df = pd.DataFrame(encoded_cols.toarray(), columns=encoded_col_names)

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


def drop_cols_with_perc_missing(X_train, X_test, percentage):
    """
    Drop columns with all missing values from both X_train and X_test.

    Args:
        X_train (pd.DataFrame): The training data.
        X_test (pd.DataFrame): The test data.
        percentage (float): The missingness percentage above which a column should be dropped.

    Returns:
        pd.DataFrame: The training data with columns dropped.
        pd.DataFrame: The test data with columns dropped.
    """
    cols_to_drop = set(X_train.columns[X_train.isnull().mean() > percentage]) | set(
        X_test.columns[X_test.isnull().mean() > percentage]
    )
    X_train.drop(cols_to_drop, axis=1, inplace=True)
    X_test.drop(cols_to_drop, axis=1, inplace=True)

    return X_train, X_test


def impute(df):
    """Impute missing values in the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The dataframe with imputed missing values.
    """
    # Impute numerical columns with Forward Fill for each stay_id group
    numerical_columns = df.select_dtypes(include=["number"]).columns
    for col in numerical_columns:
        df.loc[:, col] = df.groupby("stay_id")[col].ffill()
        # Replace any remaining unknown values with -1
        df.loc[:, col] = df[col].fillna(-1)

    # Impute categorical columns with "unknown" for unknown values
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_columns:
        df.loc[:, col] = df[col].fillna("unknown")
    return df


def scale(X_train, X_test):
    """Scale the numerical features in the training and test data.

    Args:
        X_train (pd.DataFrame): The training data (features and target).
        X_test (pd.DataFrame): The test data (features and target).

    Returns:
        pd.DataFrame: The scaled training data.
        pd.DataFrame: The scaled test data.
    """
    scaler = StandardScaler()
    numerical_columns_X_train = X_train.select_dtypes(include=["number"]).columns
    numerical_columns_X_test = X_test.select_dtypes(include=["number"]).columns
    X_train[numerical_columns_X_train] = scaler.fit_transform(
        X_train[numerical_columns_X_train]
    )
    X_test[numerical_columns_X_test] = scaler.transform(
        X_test[numerical_columns_X_test]
    )
    return X_train, X_test


def reformat_time_column(data):
    """
    Reformat the 'time' column from timedelta to total seconds.

    Args:
        data (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The dataframe with the reformatted 'time' column.
    """
    if "time" in data.columns:
        data.loc[:, "time"] = data["time"].dt.total_seconds()
    return data
