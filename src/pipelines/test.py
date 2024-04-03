from data.loading import load_configuration, load_parquet, load_csv
import matplotlib.pyplot as plt
import os


def print_shapes():
    path, filename, _config_settings = load_configuration()
    dynamic_data = load_parquet(
        path["cohorts"], filename["dynamic_data"], optimize_memory=True
    )
    static_data = load_parquet(
        path["cohorts"], filename["static_data"], optimize_memory=True
    )
    original_data = load_parquet(
        path["interim"], filename["combined"], optimize_memory=True
    )
    features = load_parquet(
        path["features"], filename["features"], optimize_memory=True
    )
    processed_data = load_parquet(
        path["processed"], filename["processed"], optimize_memory=True
    )

    print(f"Original data - Dynamic concepts shape: {dynamic_data.shape}")
    print(f"Original data - Static concepts shape: {static_data.shape}")
    print(f"Original data shape: {original_data.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Processed data shape: {processed_data.shape}")


def plot_auprc_histogram():
    path, filename, _config_settings = load_configuration()
    results_centralized = load_csv(
        path["results"], "centralized_xgboostclassifier_metrics_avg.csv"
    )
    results_local = load_csv(path["results"], "local_xgboostclassifier_metrics_avg.csv")
    results_federated = load_csv(
        path["results"], "federated_xgboostclassifier_metrics_avg.csv"
    )

    auprc_centralized = results_centralized["AUPRC Mean"]
    auprc_local = results_local["AUPRC Mean"]
    auprc_federated = results_federated["AUPRC Mean"]

    plt.hist(auprc_centralized, bins=10, alpha=0.5, label="Centralized")
    plt.hist(auprc_local, bins=10, alpha=0.5, label="Local")
    plt.hist(auprc_federated, bins=10, alpha=0.5, label="Federated")

    plt.xlabel("AUPRC")
    plt.ylabel("Frequency")
    plt.title("Histogram of AUPRC Values")
    plt.legend()
    plt.show()
    
    # Save plot to file
    plt.savefig(os.path.join(path["results"], "auprc_plot"))
    


def run_test():
    # print_shapes()
    plot_auprc_histogram()


if __name__ == "__main__":
    run_test()
