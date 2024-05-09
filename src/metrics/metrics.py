from typing import List, Optional
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


class Metrics:
    """
    Class for calculating metrics.
    """

    METRICS = [
        "Accuracy",
        "AUROC",
        "AUPRC",
        "Positive labels (TP + FN)",
        "Stay IDs with Positive labels (TP + FN)",
        "Negative labels (TN + FP)",
        "Stay IDs with Negative labels (TN + FP)",
        "False Positives",
        "Stay IDs with False Positives",
        "False Negatives",
        "Stay IDs with False Negatives",
        "True Positives",
        "Stay IDs with True Positives",
        "True Negatives",
        "Stay IDs with True Negatives",
    ]

    def __init__(self):
        """
        Constructor. Initialize lists for metrics.
        """
        self.metrics_dict = {
            "Hospitalid": {
                "value": [],
                "mean": [],
            },
            "Random State": {
                "value": [],
                "mean": [],
            },
            "Fold": {
                "value": [],
            },
            "Accuracy": {
                "value": [],
                "mean": [],
                "std": [],
            },
            "AUROC": {
                "value": [],
                "mean": [],
                "std": [],
            },
            "AUPRC": {
                "value": [],
                "mean": [],
                "std": [],
            },
            "Positive labels (TP + FN)": {
                "value": [],
                "mean": [],
                "std": [],
            },
            "Stay IDs with Positive labels (TP + FN)": {
                "value": [],
                "mean": [],
                "std": [],
            },
            "Negative labels (TN + FP)": {
                "value": [],
                "mean": [],
                "std": [],
            },
            "Stay IDs with Negative labels (TN + FP)": {
                "value": [],
                "mean": [],
                "std": [],
            },
            "False Positives": {
                "value": [],
                "mean": [],
                "std": [],
            },
            "Stay IDs with False Positives": {
                "value": [],
                "mean": [],
                "std": [],
            },
            "False Negatives": {
                "value": [],
                "mean": [],
                "std": [],
            },
            "Stay IDs with False Negatives": {
                "value": [],
                "mean": [],
                "std": [],
            },
            "True Positives": {
                "value": [],
                "mean": [],
                "std": [],
            },
            "Stay IDs with True Positives": {
                "value": [],
                "mean": [],
                "std": [],
            },
            "True Negatives": {
                "value": [],
                "mean": [],
                "std": [],
            },
            "Stay IDs with True Negatives": {
                "value": [],
                "mean": [],
                "std": [],
            },
        }

    def add_hospitalid(self, hospitalid):
        """
        Add hospital ID to metrics dictionary.

        Args:
            hospitalid: Hospital ID
        """
        self.metrics_dict["Hospitalid"]["value"].append(hospitalid)

    def add_hospitalid_avg(self, hospitalid):
        """
        Add hospital ID to metrics avg dictionary.

        Args:
            hospitalid: Hospital ID
        """
        self.metrics_dict["Hospitalid"]["mean"].append(hospitalid)

    def add_random_state(self, random_state):
        """
        Add random state to metrics dictionary.

        Args:
            random_state: Random state
        """
        self.metrics_dict["Random State"]["value"].append(random_state)

    def add_random_state_avg(self, random_state):
        """
        Add random state to metrics avg dictionary.

        Args:
            random_state: Random state
        """
        self.metrics_dict["Random State"]["mean"].append(random_state)

    def add_fold(self, fold):
        """
        Add fold to metrics dictionary.

        Args:
            fold: Fold
        """
        self.metrics_dict["Fold"]["value"].append(fold)

    def add_accuracy_value(self, y_true, y_pred):
        """
        Add accuracy value to metrics dictionary.

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        accuracy = accuracy_score(y_true, y_pred) * 100
        self.metrics_dict["Accuracy"]["value"].append(accuracy)

    def add_auroc_value(self, y_true, y_score):
        """
        Add area under receiver operating characteristic curve value to metrics dictionary.

        Args:
            y_true: True labels
            y_score: Predicted probabilities
        """
        # Calculate area under receiver operating characteristic curve
        try:
            if np.sum(y_true) == 0:  # If there are no positive samples in true labels
                auroc = "No Sepsis Occurences"
            else:
                auroc = roc_auc_score(y_true, y_score) * 100
        except ValueError:
            auroc = "Not defined"

        self.metrics_dict["AUROC"]["value"].append(auroc)

    def add_auprc_value(self, y_true, y_score):
        """
        Add area under precision-recall curve value to metrics dictionary.

        Args:
            y_true: True labels
            y_score: Predicted probabilities
        """
        # Calculate average precision
        try:
            if np.sum(y_true) == 0:  # If there are no positive samples in true labels
                auprc = "No Sepsis Occurences"
            else:
                auprc = average_precision_score(y_true, y_score) * 100
        except ValueError:
            auprc = "Not defined"

        self.metrics_dict["AUPRC"]["value"].append(auprc)

    def add_false_positives(self, y_true, y_pred, stay_ids):
        """
        Add false positives total and number of stay IDs with false positives to metrics dictionary.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            stay_ids: Stay IDs
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        try:
            fp = cm[0][1]
        except IndexError:
            fp = 0

        # Create a DataFrame with stay_ids, y_true, and y_pred
        df = pd.DataFrame({"stay_id": stay_ids, "y_true": y_true, "y_pred": y_pred})

        # Group by stay_id and calculate the sum of y_true and y_pred for each group
        group_sums = df.groupby("stay_id").agg({"y_true": "sum", "y_pred": "sum"})

        # Count the number of stay_ids where y_true sum is 0 and y_pred sum is greater than 0
        stay_ids_with_fp = (
            (group_sums["y_true"] == 0) & (group_sums["y_pred"] > 0)
        ).sum()

        self.metrics_dict["False Positives"]["value"].append(fp)
        self.metrics_dict["Stay IDs with False Positives"]["value"].append(
            stay_ids_with_fp
        )

    def add_false_negatives(self, y_true, y_pred, stay_ids):
        """
        Add false negatives total and number of stay IDs with false negatives to metrics dictionary.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            stay_ids: Stay IDs
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        try:
            fn = cm[1][0]
        except IndexError:
            fn = 0

        # Create a DataFrame with stay_ids, y_true, and y_pred
        df = pd.DataFrame({"stay_id": stay_ids, "y_true": y_true, "y_pred": y_pred})

        # Group by stay_id and calculate the sum of y_true and y_pred for each group
        group_sums = df.groupby("stay_id").agg({"y_true": "sum", "y_pred": "sum"})

        # Count the number of stay_ids where y_true sum is greater than 0 and y_pred sum is 0
        stay_ids_with_fn = (
            (group_sums["y_true"] > 0) & (group_sums["y_pred"] == 0)
        ).sum()

        self.metrics_dict["False Negatives"]["value"].append(fn)
        self.metrics_dict["Stay IDs with False Negatives"]["value"].append(
            stay_ids_with_fn
        )

    def add_true_positives(self, y_true, y_pred, stay_ids):
        """
        Add true positives total and number of stay IDs with true positives to metrics dictionary.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            stay_ids: Stay IDs
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        try:
            tp = cm[1][1]
        except IndexError:
            tp = 0

        # Create a DataFrame with stay_ids, y_true, and y_pred
        df = pd.DataFrame({"stay_id": stay_ids, "y_true": y_true, "y_pred": y_pred})

        # Group by stay_id and calculate the sum of y_true and y_pred for each group
        group_sums = df.groupby("stay_id").agg({"y_true": "sum", "y_pred": "sum"})

        # Count the number of stay_ids where y_true sum is greater than 0 and y_pred sum is greater than 0
        stay_ids_with_tp = (
            (group_sums["y_true"] > 0) & (group_sums["y_pred"] > 0)
        ).sum()

        self.metrics_dict["True Positives"]["value"].append(tp)
        self.metrics_dict["Stay IDs with True Positives"]["value"].append(
            stay_ids_with_tp
        )

    def add_true_negatives(self, y_true, y_pred, stay_ids):
        """
        Add true negatives total and number of stay IDs with true negatives to metrics dictionary.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            stay_ids: Stay IDs
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        try:
            tn = cm[0][0]
        except IndexError:
            tn = 0

        # Create a DataFrame with stay_ids, y_true, and y_pred
        df = pd.DataFrame({"stay_id": stay_ids, "y_true": y_true, "y_pred": y_pred})

        # Group by stay_id and calculate the sum of y_true and y_pred for each group
        group_sums = df.groupby("stay_id").agg({"y_true": "sum", "y_pred": "sum"})

        # Count the number of stay_ids where y_true sum is 0 and y_pred sum is 0
        stay_ids_with_tn = (
            (group_sums["y_true"] == 0) & (group_sums["y_pred"] == 0)
        ).sum()

        self.metrics_dict["True Negatives"]["value"].append(tn)
        self.metrics_dict["Stay IDs with True Negatives"]["value"].append(
            stay_ids_with_tn
        )

    def add_positve_negative_label_counts(self):
        self.metrics_dict["Positive labels (TP + FN)"]["value"].append(
            self.metrics_dict["True Positives"]["value"][-1]
            + self.metrics_dict["False Negatives"]["value"][-1]
        )
        self.metrics_dict["Stay IDs with Positive labels (TP + FN)"]["value"].append(
            self.metrics_dict["Stay IDs with True Positives"]["value"][-1]
            + self.metrics_dict["Stay IDs with False Negatives"]["value"][-1]
        )
        self.metrics_dict["Negative labels (TN + FP)"]["value"].append(
            self.metrics_dict["True Negatives"]["value"][-1]
            + self.metrics_dict["False Positives"]["value"][-1]
        )
        self.metrics_dict["Stay IDs with Negative labels (TN + FP)"]["value"].append(
            self.metrics_dict["Stay IDs with True Negatives"]["value"][-1]
            + self.metrics_dict["Stay IDs with False Positives"]["value"][-1]
        )

    def add_individual_confusion_matrix_values(self, y_true, y_pred, stay_ids):
        """
        Add individual confusion matrix values to metrics dictionary.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            stay_ids: Stay IDs
        """
        self.add_false_positives(y_true, y_pred, stay_ids)
        self.add_false_negatives(y_true, y_pred, stay_ids)
        self.add_true_positives(y_true, y_pred, stay_ids)
        self.add_true_negatives(y_true, y_pred, stay_ids)

        self.add_positve_negative_label_counts()

    def _get_metric_list(
        self,
        metric: str,
        mask: Optional[np.ndarray] = None,
        on_mean_data: bool = False,
    ) -> np.ndarray:
        """
        Get metric list from metrics dictionary.

        Args:
            metric: Metric name
            mask: Mask for filtering
            on_mean_data: If True, get metric list from mean data

        Returns:
            np.ndarray: Metric list
        """
        if on_mean_data:
            metric_list = self.metrics_dict[metric]["mean"]
        else:
            metric_list = self.metrics_dict[metric]["value"]

        # If specified: filter elements
        if mask is not None:
            metric_list = [
                value for value, include in zip(metric_list, mask) if include
            ]

        # Remove invalid elements
        metric_list = [
            metric
            for metric in metric_list
            if metric
            not in [
                "Not defined",
                "No Sepsis Occurences",
                "Mean calculation not possible",
                "Std calculation not possible",
            ]
        ]

        return metric_list

    def _calculate_metric_stat(self, metric_list: List[float], stat_type: str):
        """
        Calculate the mean or standard deviation of a metric list.

        Args:
            metric_list: List of metric values
            stat_type: Type of statistic to calculate ("mean" or "std")

        Returns:
            float or str: Calculated statistic or error message
        """
        try:
            if stat_type == "mean":
                return round(sum(metric_list) / len(metric_list), 2)
            elif stat_type == "std":
                return round(np.std(metric_list), 2)
        except Exception:
            return f"{stat_type.capitalize()} calculation not possible"

    def add_metrics_stats(
        self,
        selected_metrics: List[str],
        mask: Optional[np.ndarray] = None,
        on_mean_data: bool = False,
    ):
        """
        Add mean and standard deviation to metrics dictionary for selected metrics.

        Args:
            selected_metrics: List of selected metrics.
            mask: Mask for filtering
            on_mean_data: If True, calculate statistics on mean data
        """
        for metric in selected_metrics:
            metric_list = self._get_metric_list(metric, mask, on_mean_data)
            metric_mean = self._calculate_metric_stat(metric_list, "mean")
            metric_std = self._calculate_metric_stat(metric_list, "std")
            self.metrics_dict[metric]["mean"].append(metric_mean)
            self.metrics_dict[metric]["std"].append(metric_std)

    def calculate_averages_across_random_states(self):
        mask_1 = [
            value > 0
            for value in self.metrics_dict["Positive labels (TP + FN)"]["value"]
        ]
        self.add_random_state_avg("Total Average")
        self.add_metrics_stats(self.METRICS, mask_1)

        mask_2 = self.metrics_dict["Random State"]["value"] != "Total Average"
        mask_3 = [
            (
                self.metrics_dict["Positive labels (TP + FN)"]["value"][i]
                / (
                    self.metrics_dict["Positive labels (TP + FN)"]["value"][i]
                    + self.metrics_dict["Negative labels (TN + FP)"]["value"][i]
                )
            )
            > 0.1
            for i in range(len(self.metrics_dict["Positive labels (TP + FN)"]["value"]))
        ]
        mask = mask_1 & mask_2 & mask_3
        self.add_random_state_avg("Total Average (>0.1 sepsis proportion)")
        self.add_metrics_stats(self.METRICS, mask)

    def calculate_averages_per_hospitalid_across_random_states(self):
        for hospitalid in set(self.metrics_dict["Hospitalid"]["value"]):
            mask_1 = self.metrics_dict["Hospitalid"]["value"] == hospitalid
            mask_2 = [
                value > 0
                for value in self.metrics_dict["Positive labels (TP + FN)"]["value"]
            ]
            mask = mask_1 & mask_2
            self.add_hospitalid_avg(hospitalid)
            self.add_metrics_stats(self.METRICS, mask)

    def calculate_total_averages_across_hospitalids(self):
        mask_1 = [
            value > 0
            for value in self.metrics_dict["Positive labels (TP + FN)"]["value"]
        ]
        self.add_hospitalid_avg("Total Average")
        self.add_metrics_stats(self.METRICS, mask_1, on_mean_data=True)

        mask_2 = np.full(len(self.metrics_dict["Hospitalid"]["value"]), True)
        mask_2[-1] = False
        mask_3 = [
            (
                self.metrics_dict["Positive labels (TP + FN)"]["value"][i]
                / (
                    self.metrics_dict["Positive labels (TP + FN)"]["value"][i]
                    + self.metrics_dict["Negative labels (TN + FP)"]["value"][i]
                )
            )
            > 0.01
            for i in range(len(self.metrics_dict["Positive labels (TP + FN)"]["value"]))
        ]
        print(mask_2)
        mask_1_2 = mask_1 & mask_2
        mask_1_2_3 = mask_1_2 & mask_3
        self.add_hospitalid_avg("Total Average (>0.1 sepsis proportion)")
        self.add_metrics_stats(self.METRICS, mask_1_2_3, on_mean_data=True)

    def get_metrics_dataframe(
        self,
        selected_metrics: Optional[List[str]] = None,
        additional_metrics: Optional[List[str]] = None,
        avg_metrics: bool = False,
    ) -> pd.DataFrame:
        """
        Return a metrics dataframe from the metrics dictionary. Include only the specified metrics.

        Args:
            selected_metrics: List of selected metrics. If none, use the standard metrics.
            additional_metrics: List of metrics to include, in addition to the standard metrics.
            avg_metrics: If True, return the average metrics dataframe.

        Returns:
            pd.DataFrame: The metrics dataframe.
        """
        if selected_metrics is None:
            selected_metrics = self.METRICS

        if additional_metrics is not None:
            selected_metrics = additional_metrics + selected_metrics

        # Create a dictionary to save the filtered metrics data
        filtered_metrics_dict = {}

        # Iterate over the outer keys and extract the sublevels for each key
        for key, value in self.metrics_dict.items():
            if key in selected_metrics:
                if avg_metrics:
                    if "mean" in value:
                        filtered_metrics_dict[key + " Mean"] = value["mean"]
                    if "std" in value:
                        filtered_metrics_dict[key + " Std"] = value["std"]
                else:
                    filtered_metrics_dict[key] = value["value"]

        return pd.DataFrame(filtered_metrics_dict)

    def get_summary_dataframe(self):
        accuracy_prop = f'{self.metrics_dict["Accuracy"]["mean"][-1]} ({self.metrics_dict["Accuracy"]["std"][-1]})'
        auroc_prop = f'{self.metrics_dict["AUROC"]["mean"][-1]} ({self.metrics_dict["AUROC"]["std"][-1]})'
        auprc_prop = f'{self.metrics_dict["AUPRC"]["mean"][-1]} ({self.metrics_dict["AUPRC"]["std"][-1]})'
        positive_labels_prop = f'{self.metrics_dict["Positive labels (TP + FN)"]["mean"][-1]} ({self.metrics_dict["Stay IDs with Positive labels (TP + FN)"]["mean"][-1]})'
        negative_labels_prop = f'{self.metrics_dict["Negative labels (TN + FP)"]["mean"][-1]} ({self.metrics_dict["Stay IDs with Negative labels (TN + FP)"]["mean"][-1]})'
        false_positives_prop = f'{self.metrics_dict["False Positives"]["mean"][-1]} ({self.metrics_dict["Stay IDs with False Positives"]["mean"][-1]})'
        false_negatives_prop = f'{self.metrics_dict["False Negatives"]["mean"][-1]} ({self.metrics_dict["Stay IDs with False Negatives"]["mean"][-1]})'

        accuracy = f'{self.metrics_dict["Accuracy"]["mean"][-2]} ({self.metrics_dict["Accuracy"]["std"][-2]})'
        auroc = f'{self.metrics_dict["AUROC"]["mean"][-2]} ({self.metrics_dict["AUROC"]["std"][-2]})'
        auprc = f'{self.metrics_dict["AUPRC"]["mean"][-2]} ({self.metrics_dict["AUPRC"]["std"][-2]})'
        positive_labels = f'{self.metrics_dict["Positive labels (TP + FN)"]["mean"][-2]} ({self.metrics_dict["Stay IDs with Positive labels (TP + FN)"]["mean"][-2]})'
        negative_labels = f'{self.metrics_dict["Negative labels (TN + FP)"]["mean"][-2]} ({self.metrics_dict["Stay IDs with Negative labels (TN + FP)"]["mean"][-2]})'
        false_positives = f'{self.metrics_dict["False Positives"]["mean"][-2]} ({self.metrics_dict["Stay IDs with False Positives"]["mean"][-2]})'
        false_negatives = f'{self.metrics_dict["False Negatives"]["mean"][-2]} ({self.metrics_dict["Stay IDs with False Negatives"]["mean"][-2]})'

        summary = {
            "Accuracy": [accuracy, accuracy_prop],
            "AUROC": [auroc, auroc_prop],
            "AUPRC": [auprc, auprc_prop],
            "Positive labels (TP + FN)": [positive_labels, positive_labels_prop],
            "Negative labels (TN + FP)": [negative_labels, negative_labels_prop],
            "False Positives": [false_positives, false_positives_prop],
            "False Negatives": [false_negatives, false_negatives_prop],
        }

        summary_df = pd.DataFrame(summary)

        return summary_df
