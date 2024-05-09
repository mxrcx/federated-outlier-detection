import sys

sys.path.append("..")

from data.loading import load_configuration, load_parquet, load_csv
from data.processing import reformat_time_column, encode_categorical_columns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import numpy as np


def print_shapes():
    path, filename, config_settings = load_configuration()
    """
    dynamic_data = load_parquet(
        path["cohorts"], filename["dynamic_data"], optimize_memory=True
    )
    static_data = load_parquet(
        path["cohorts"], filename["static_data"], optimize_memory=True
    )
    original_data = load_parquet(
        path["interim"], filename["combined"], optimize_memory=True
    )
    """
    features = load_parquet(
        path["features"], filename["features"], optimize_memory=True
    )
    features = reformat_time_column(features)
    features = encode_categorical_columns(features, config_settings["training_columns_to_drop"])
    processed_data = load_parquet(
        path["processed"], filename["processed"], optimize_memory=True
    )

    """
    print(f"Original data - Dynamic concepts shape: {dynamic_data.shape}")
    print(f"Original data - Static concepts shape: {static_data.shape}")
    print(f"Original data shape: {original_data.shape}")
    
    print(dynamic_data["sbp"].describe())
    """
    
    print(f"Features shape: {features.shape}")
    print(f"Feature column names before: {list(features.columns)}")
    
    print(features["sbp_max"].describe())
    print(features["sbp_max"].isnull().mean())
    print(features["sbp_max"].isnull().sum())
        
    print(features["sbp_mean"].describe())
    print(features["sbp_mean"].isnull().mean())
    print(features["sbp_mean"].isnull().sum())

    print(f"Processed data shape: {processed_data.shape}")
    print(f"Feature column names after: {list(processed_data.columns)}")
    
    removed_features = [item for item in list(features.columns) if item not in list(processed_data.columns)]
    print(f"Removed features: {len(removed_features)} {removed_features}")
    
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
    
    missing_percentage = data.isna().mean() * 100
    missing_info = pd.DataFrame(
        {
            "Feature": missing_percentage.index,
            "MissingPercentage": missing_percentage.values,
        }
    )
    
    lowest_missing = missing_info.sort_values("MissingPercentage").head(200)
    highest_missing = missing_info.sort_values("MissingPercentage", ascending=False).head(40)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
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
    
def plot_sepsis_proportion_hospitals():
    path, filename, _config_settings = load_configuration()
    data = load_parquet(path["features"], filename["features"], optimize_memory=True)
    
    hospital_counts = data.groupby("hospitalid")["label"].sum()
    hospitals_with_only_label_0 = hospital_counts[hospital_counts == 0]
    print(f"Number of unique hospitals with only label 0: {len(hospitals_with_only_label_0)}")
    
    hist, bins, _ = plt.hist(hospital_counts, bins=25)
    plt.xlabel("Number of Sepsis Observations")
    plt.ylabel("Number of Hospitals")
    plt.title("Distribution of Sepsis Observations per Hospital")
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.xticks(bins, rotation=45)
    plt.tight_layout()
    plt.show()
    
    plt.savefig(os.path.join(path["results"], "sepsis_proportion_hospitals.png"))  # Save after showing
    
    total_cases_per_hospital = data.groupby("hospitalid")["label"].count()
    hist, bins, _ = plt.hist(total_cases_per_hospital, bins=25)
    plt.xlabel("Number of all Observations")
    plt.ylabel("Number of Hospitals")
    plt.title("Distribution of all Observations per Hospital")
    plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    plt.xticks(bins, rotation=45)
    plt.tight_layout()
    plt.show()
    
    plt.savefig(os.path.join(path["results"], "total_cases_hospitals.png"))  # Save after showing
    
def test_difference_pos_label():
    path, filename, _config_settings = load_configuration()
    results_xgboost_centralized = load_csv(path["results"], "centralized_xgboostclassifier_metrics_avg.csv")
    results_xgboost_local = load_csv(path["results"], "local_xgboostclassifier_metrics_avg.csv")
    results_xgboost_federated = load_csv(path["results"], "federated_xgboostclassifier_metrics_avg.csv")
    
    # Calculate the differences in the "Positive labels (TP + FN) Mean" column
    diff_centralized = results_xgboost_centralized["Positive labels (TP + FN) Mean"] - results_xgboost_local["Positive labels (TP + FN) Mean"]
    diff_federated = results_xgboost_federated["Positive labels (TP + FN) Mean"] - results_xgboost_local["Positive labels (TP + FN) Mean"]

    # Print the differences
    print("Differences in PosLabels (Centralized - Local):")
    print(diff_centralized)
    print("Differences in PosLabels (Federated - Local):")
    print(diff_federated)
    
    
def plot_score_sampledensity():
    path, filename, _config_settings = load_configuration()
    results_ocsvm_local = load_csv(path["results"], "local_oneclasssvm_metrics_avg.csv")
    results_xgboost_local = load_csv(path["results"], "local_xgboostclassifier_metrics_avg.csv")
    
    results_ocsvm_local = results_ocsvm_local.iloc[1:-1]
    results_xgboost_local = results_xgboost_local.iloc[1:-1]
    
    # Extracting data for OCSVM
    pos_counts_ocsvm = results_ocsvm_local["Positive labels (TP + FN) Mean"]
    neg_counts_ocsvm = results_ocsvm_local["Negative labels (TN + FP) Mean"]
    auprc_scores_ocsvm = results_ocsvm_local["AUPRC Mean"] 
    auroc_scores_ocsvm = results_ocsvm_local["AUROC Mean"]

    # Extracting data for XGBoost
    pos_counts_xgboost = results_xgboost_local["Positive labels (TP + FN) Mean"]
    neg_counts_xgboost = results_xgboost_local["Negative labels (TN + FP) Mean"]
    auprc_scores_xgboost = results_xgboost_local["AUPRC Mean"]
    auroc_scores_xgboost = results_xgboost_local["AUROC Mean"]
    
    
    # Replace all strings or empty entries of the scores with 0
    pos_counts_ocsvm = pos_counts_ocsvm.apply(lambda x: 0 if (isinstance(x, str) and len(x) > 6) or pd.isnull(x) else float(x))
    neg_counts_ocsvm = neg_counts_ocsvm.apply(lambda x: 0 if (isinstance(x, str) and len(x) > 6) or pd.isnull(x)else float(x))
    auprc_scores_ocsvm = auprc_scores_ocsvm.apply(lambda x: 0 if (isinstance(x, str) and len(x) > 6) or pd.isnull(x)else float(x))
    auroc_scores_ocsvm = auroc_scores_ocsvm.apply(lambda x: 0 if (isinstance(x, str) and len(x) > 6) or pd.isnull(x)else float(x))
    pos_counts_xgboost = pos_counts_xgboost.apply(lambda x: 0 if (isinstance(x, str) and len(x) > 6) or pd.isnull(x)else float(x))
    neg_counts_xgboost = neg_counts_xgboost.apply(lambda x: 0 if (isinstance(x, str) and len(x) > 6) or pd.isnull(x)else float(x))
    auprc_scores_xgboost = auprc_scores_xgboost.apply(lambda x: 0 if (isinstance(x, str) and len(x) > 6) or pd.isnull(x)else float(x))
    auroc_scores_xgboost = auroc_scores_xgboost.apply(lambda x: 0 if (isinstance(x, str) and len(x) > 6) or pd.isnull(x)else float(x))

    proportion_positive_ocsvm = pos_counts_ocsvm / (pos_counts_ocsvm + neg_counts_ocsvm)
    proportion_positive_xgboost = pos_counts_xgboost / (pos_counts_xgboost + neg_counts_xgboost)
    
    # Plotting OCSVM
    plt.figure(figsize=(10, 6))  # Adjust figure size
    plt.scatter(pos_counts_ocsvm, auroc_scores_ocsvm, label="OCSVM (AUROC)", s=50, alpha=0.7, color='red')  # Increase marker size
    plt.xlabel("Number of Positive Observations per Hospital")
    plt.ylabel("Performance Score")
    plt.title("OCSVM Performance: Number of Positive Observations per Hospital vs Performance Score")
    plt.grid(True)  # Add grid lines
    plt.legend()  # Add legend
    plt.xticks(ticks=range(0, int(max(pos_counts_ocsvm))+1, 20), rotation=45)  # Specify ticks and rotate x-axis labels for better readability
    plt.yticks(ticks=range(0, int(max(auroc_scores_ocsvm))+1, 10)) 
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    
    plt.savefig(os.path.join(path["results"], "ocsvm_performance_plot_auroc_number.png"))  # Save after showing

    
    plt.figure(figsize=(10, 6))  # Adjust figure size
    plt.scatter(pos_counts_ocsvm, auprc_scores_ocsvm, label="OCSVM (AUPRC)", s=50, alpha=0.7, color='blue')  # Increase marker size
    plt.xlabel("Number of Positive Observations per Hospital")
    plt.ylabel("Performance Score")
    plt.title("OCSVM Performance: Number of Positive Observations per Hospital vs Performance Score")
    plt.grid(True)  # Add grid lines
    plt.legend()  # Add legend
    plt.xticks(ticks=range(0, int(max(pos_counts_ocsvm))+1, 20), rotation=45)  # Specify ticks and rotate x-axis labels for better readability
    plt.yticks(ticks=range(0, int(max(auprc_scores_ocsvm))+1, 2)) 
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    
    plt.savefig(os.path.join(path["results"], "ocsvm_performance_plot_auprc_number.png"))  # Save after showing
    
    plt.figure(figsize=(10, 6))  # Adjust figure size
    plt.scatter(proportion_positive_ocsvm, auroc_scores_ocsvm, label="OCSVM (AUROC)", s=50, alpha=0.7, color='red')  # Increase marker size
    plt.xlabel("Proportion of Positive Observations per Hospital")
    plt.ylabel("Performance Score")
    plt.title("OCSVM Performance: Proportion of Positive Observations per Hospital vs Performance Score")
    plt.grid(True)  # Add grid lines
    plt.legend()  # Add legend
    plt.xlim(0, 0.3)  # Set x-axis limits to create a gap
    plt.xticks(rotation=45)  # Specify ticks and rotate x-axis labels for better readability
    plt.yticks(ticks=range(0, int(max(auroc_scores_ocsvm))+1, 10)) 
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    plt.savefig(os.path.join(path["results"], "ocsvm_performance_plot_auroc_proportion.png"))  # Save after showing
    
    plt.figure(figsize=(10, 6))  # Adjust figure size
    plt.scatter(proportion_positive_ocsvm, auprc_scores_ocsvm, label="OCSVM (AUPRC)", s=50, alpha=0.7, color='blue')  # Increase marker size
    plt.xlabel("Proportion of Positive Observations per Hospital")
    plt.ylabel("Performance Score")
    plt.title("OCSVM Performance: Proportion of Positive Observations per Hospital vs Performance Score")
    plt.grid(True)  # Add grid lines
    plt.legend()  # Add legend
    plt.xlim(0, 0.3)  # Set x-axis limits to create a gap
    plt.xticks(rotation=45)  # Specify ticks and rotate x-axis labels for better readability
    plt.yticks(ticks=range(0, int(max(auprc_scores_ocsvm))+1, 2)) 
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    plt.savefig(os.path.join(path["results"], "ocsvm_performance_plot_auprc_proportion.png"))  # Save after showing

    # Plotting XGBoost
    plt.figure(figsize=(10, 6))  # Adjust figure size
    plt.scatter(pos_counts_xgboost, auroc_scores_xgboost, label="XGBoost (AUROC)", s=50, alpha=0.7, color='red')  # Increase marker size
    plt.xlabel("Number of Positive Observations per Hospital")
    plt.ylabel("Performance Score")
    plt.title("XGBoost Performance: Number of Positive Observations per Hospital vs Performance Score")
    plt.grid(True)  # Add grid lines
    plt.legend()  # Add legend
    plt.xticks(ticks=range(0, int(max(pos_counts_xgboost))+1, 20), rotation=45)  # Specify ticks and rotate x-axis labels for better readability
    plt.yticks(ticks=range(0, int(max(auroc_scores_xgboost))+1, 10)) 
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    
    plt.savefig(os.path.join(path["results"], "xgboost_performance_plot_auroc_number.png"))  # Save after showing
    
    plt.figure(figsize=(10, 6))  # Adjust figure size
    plt.scatter(pos_counts_xgboost, auprc_scores_xgboost, label="XGBoost (AUPRC)", s=50, alpha=0.7, color='blue')  # Increase marker size
    plt.xlabel("Number of Positive Observations per Hospital")
    plt.ylabel("Performance Score")
    plt.title("XGBoost Performance: Number of Positive Observations per Hospital vs Performance Score")
    plt.grid(True)  # Add grid lines
    plt.legend()  # Add legend
    plt.xticks(ticks=range(0, int(max(pos_counts_xgboost))+1, 20), rotation=45)  # Specify ticks and rotate x-axis labels for better readability
    plt.yticks(ticks=range(0, int(max(auprc_scores_xgboost))+1, 2)) 
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    
    plt.savefig(os.path.join(path["results"], "xgboost_performance_plot_auprc_number.png"))  # Save after showing
    
    plt.figure(figsize=(10, 6))  # Adjust figure size
    plt.scatter(proportion_positive_xgboost, auroc_scores_xgboost, label="XGBoost (AUROC)", s=50, alpha=0.7, color='red')  # Increase marker size
    plt.xlabel("Proportion of Positive Observations per Hospital")
    plt.ylabel("Performance Score")
    plt.title("XGBoost Performance: Proportion of Positive Observations per Hospital vs Performance Score")
    plt.grid(True)  # Add grid lines
    plt.legend()  # Add legend
    plt.xlim(0, 0.3)  # Set x-axis limits to create a gap
    plt.xticks(rotation=45)  # Specify ticks and rotate x-axis labels for better readability
    plt.yticks(ticks=range(0, int(max(auroc_scores_xgboost))+1, 10)) 
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    plt.savefig(os.path.join(path["results"], "xgboost_performance_plot_auroc_proportion.png"))  # Save after showing
    
    plt.figure(figsize=(10, 6))  # Adjust figure size
    plt.scatter(proportion_positive_xgboost, auprc_scores_xgboost, label="XGBoost (AUPRC)", s=50, alpha=0.7, color='blue')  # Increase marker size
    plt.xlabel("Proportion of Positive Observations per Hospital")
    plt.ylabel("Performance Score")
    plt.title("XGBoost Performance: Proportion of Positive Observations per Hospital vs Performance Score")
    plt.grid(True)  # Add grid lines
    plt.legend()  # Add legend
    plt.xlim(0, 0.3)  # Set x-axis limits to create a gap
    plt.xticks(rotation=45)  # Specify ticks and rotate x-axis labels for better readability
    plt.yticks(ticks=range(0, int(max(auprc_scores_xgboost))+1, 2)) 
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    plt.savefig(os.path.join(path["results"], "xgboost_performance_plot_auprc_proportion.png"))  # Save after showing


def run_test():
    print_shapes()
    # plot_auprc_histogram()
    print_statistics()
    # plot_score_sampledensity()
    # plot_sepsis_proportion_hospitals()
    # test_difference_pos_label()


if __name__ == "__main__":
    run_test()
