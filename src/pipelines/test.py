from data.loading import load_configuration, load_parquet, load_csv
import matplotlib.pyplot as plt
import os
import pandas as pd


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
    
def print_statistics():
    path, filename, _config_settings = load_configuration()
    data = load_parquet(path["features"], filename["features"], optimize_memory=True)
    
    count_sepsis_imps = data[data["label"] == 1].shape[0]
    print(f"Number of IMPs with sepsis: {count_sepsis_imps}")
    
    unique_patients = data["uniquepid"].nunique()
    print(f"Number of unique patients in cohort: {unique_patients}")
    
    data_sepsis = data[data["label"] == 1]
    
    unique_sepsis_patients = data_sepsis["uniquepid"].nunique()
    print(f"Number of unique sepsis patients in cohort: {unique_sepsis_patients}")
    
    missing_percentage = data.isnull().mean() * 100
    missing_info = pd.DataFrame(
        {
            "Feature": missing_percentage.index,
            "MissingPercentage": missing_percentage.values,
        }
    )
    
    lowest_missing = missing_info.sort_values("MissingPercentage").head(20)
    highest_missing = missing_info.sort_values("MissingPercentage", ascending=False).head(20)

    print("Lowest missing values:")
    print(lowest_missing)

    print("Highest missing values:")
    print(highest_missing)
    

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
    # plot_auprc_histogram()
    print_statistics()


if __name__ == "__main__":
    run_test()
