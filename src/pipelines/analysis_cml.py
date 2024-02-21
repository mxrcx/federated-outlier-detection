import pandas as pd
import numpy as np
from data.loading import load_csv
from data.saving import save_csv
from data.saving import save_csv

# Load the csv files
analysis_loho = load_csv("../../results", "cml_loho_split_metrics.csv")
analysis_loho_avg = load_csv("../../results", "cml_loho_split_metrics_avg.csv")
analysis_local_avg = load_csv("../../results", "cml_local_metrics_avg.csv")

# Rename old columns
analysis_loho = analysis_loho.rename(
    columns={
        "Hospitalid": "Hospitalid Mean",
        "Accuracy": "Accuracy Mean",
        "AUROC": "AUROC Mean",
        "AUPRC": "AUPRC Mean",
        "Confusion Matrix": "Confusion Matrix Mean",
        "TN-FP-Sum": "TN-FP-Sum Mean",
        "FPR": "FPR Mean",
    }
)

# Insert new columns
analysis_loho.insert(2, "Accuracy Std", np.tile(0, len(analysis_loho)))
analysis_loho.insert(4, "AUROC Std", np.tile(0, len(analysis_loho)))
analysis_loho.insert(6, "AUPRC Std", np.tile(0, len(analysis_loho)))
analysis_loho.insert(9, "TN-FP-Sum Std", np.tile(0, len(analysis_loho)))
analysis_loho.insert(11, "FPR Std", np.tile(0, len(analysis_loho)))

analysis_loho_avg.insert(0, "Hospitalid Mean", "Total Average")

## Combine analysis_loho and add analysis_loho as a new row
analysis_loho = pd.concat(
    [analysis_loho, analysis_loho_avg],
    ignore_index=True,
)

print(analysis_loho.shape)
print(analysis_loho)

## Combine analysis_loho and analysis_local_avg
combined_df = analysis_loho.join(analysis_local_avg, lsuffix="_loho", rsuffix="_local")
combined_df = combined_df.drop(columns=["Hospitalid Mean_local"])
combined_df = combined_df.rename(
    columns={
        "Hospitalid Mean_loho": "Hospitalid Mean",
    }
)

print(combined_df.shape)
print(combined_df)

# Replace all string values in Dataframe with NaN
combined_df = combined_df.map(
    lambda x: pd.to_numeric(x, errors="coerce") if isinstance(x, str) else x
)

# Calculate differences on non NaN values
numeric_cases_acc = combined_df[
    (~combined_df["Accuracy Mean_loho"].isna())
    & (~combined_df["Accuracy Mean_local"].isna())
]
combined_df["Accuracy Difference"] = (
    numeric_cases_acc["Accuracy Mean_loho"] - numeric_cases_acc["Accuracy Mean_local"]
)

numeric_cases_auroc = combined_df[
    (~combined_df["AUROC Mean_loho"].isna()) & (~combined_df["AUROC Mean_local"].isna())
]
combined_df["AUROC Difference"] = (
    numeric_cases_auroc["AUROC Mean_loho"] - numeric_cases_auroc["AUROC Mean_local"]
)

numeric_cases_auprc = combined_df[
    (~combined_df["AUPRC Mean_loho"].isna()) & (~combined_df["AUPRC Mean_local"].isna())
]
combined_df["AUPRC Difference"] = (
    numeric_cases_auprc["AUPRC Mean_loho"] - numeric_cases_auprc["AUPRC Mean_local"]
)

print(combined_df.shape)
print(combined_df)

# Save the combined dataframe to a csv file
save_csv(combined_df, "../../results", "cml_comparison_loho_local.csv")

# Create new dataframe with the average differences
average_diff = pd.DataFrame(
    {
        "Accuracy Difference Mean": [combined_df["Accuracy Difference"].mean()],
        "Accuracy Difference Std": [combined_df["Accuracy Difference"].std()],
        "AUROC Difference Mean": [combined_df["AUROC Difference"].mean()],
        "AUROC Difference Std": [combined_df["AUROC Difference"].std()],
        "AUPRC Difference Mean": [combined_df["AUPRC Difference"].mean()],
        "AUPRC Difference Std": [combined_df["AUPRC Difference"].std()],
    }
)

# Save the average differences to a csv file
save_csv(average_diff, "../../results", "cml_comparison_loho_local_avg_diff.csv")
