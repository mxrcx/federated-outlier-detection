from data.loading import load_configuration, load_parquet


def print_shapes():
    path, filename, _config_settings = load_configuration()
    original_data = load_parquet(
        path["interim"], filename["combined"], optimize_memory=True
    )
    features = load_parquet(
        path["features"], filename["features"], optimize_memory=True
    )
    processed_data = load_parquet(
        path["processed"], filename["processed"], optimize_memory=True
    )

    print(f"Original data shape: {original_data.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Processed data shape: {processed_data.shape}")


def run_test():
    print_shapes()


if __name__ == "__main__":
    run_test()
