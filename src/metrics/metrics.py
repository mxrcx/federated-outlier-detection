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
        "True Negatives",
        "Stay IDs with True Negatives",
        "False Positives",
        "Stay IDs with False Positives",
        "False Negatives",
        "Stay IDs with False Negatives",
        "True Positives",
        "Stay IDs with True Positives",
        "TN-FP-Sum",
        "FPR",
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
            "Confusion Matrix": {
                "value": [],
                "mean": [],
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
            "TN-FP-Sum": {
                "value": [],
                "mean": [],
                "std": [],
            },
            "FPR": {
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
        accuracy = round(accuracy_score(y_true, y_pred), 4)
        self.metrics_dict["Accuracy"]["value"].append(accuracy)

    def _get_y_score(self, y_pred_proba):
        """
        Get the probability of the positive class (y_score).

        Args:
            y_pred_proba: Predicted probabilities

        Returns:
            np.ndarray: Probability of the positive class
        """
        try:
            y_score = y_pred_proba[:, 1]
        except IndexError:
            # If all predictions are False(=0), no predictions are True(=1)
            y_score = np.zeros_like(y_pred_proba)
        return y_score

    def add_auroc_value(self, y_true, y_pred_proba):
        """
        Add area under receiver operating characteristic curve value to metrics dictionary.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
        """
        y_score = self._get_y_score(y_pred_proba)

        # Calculate area under receiver operating characteristic curve
        try:
            if np.sum(y_true) == 0:  # If there are no positive samples in true labels
                auroc = "No Sepsis Occurences"
            else:
                auroc = round(roc_auc_score(y_true, y_score), 4)
        except ValueError:
            auroc = "Not defined"

        self.metrics_dict["AUROC"]["value"].append(auroc)

    def add_auprc_value(self, y_true, y_pred_proba):
        """
        Add area under precision-recall curve value to metrics dictionary.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
        """
        y_score = self._get_y_score(y_pred_proba)

        # Calculate average precision
        try:
            if np.sum(y_true) == 0:  # If there are no positive samples in true labels
                auprc = "No Sepsis Occurences"
            else:
                auprc = round(average_precision_score(y_true, y_score), 4)
        except ValueError:
            auprc = "Not defined"

        self.metrics_dict["AUPRC"]["value"].append(auprc)

    def add_confusion_matrix(self, y_true, y_pred):
        """
        Add confusion matrix to metrics dictionary.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            stay_ids: Stay IDs
        """
        cm = confusion_matrix(y_true, y_pred)
        self.metrics_dict["Confusion Matrix"]["value"].append(cm)

    def add_false_positives(self, y_true, y_pred, stay_ids):
        """
        Add false positives total and number of stay IDs with false positives to metrics dictionary.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            stay_ids: Stay IDs
        """
        cm = confusion_matrix(y_true, y_pred)
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
        cm = confusion_matrix(y_true, y_pred)
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
        cm = confusion_matrix(y_true, y_pred)
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
        cm = confusion_matrix(y_true, y_pred)
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

    def _get_last_confusion_matrix(self):
        """
        Get last confusion matrix.

        Returns:
            np.ndarray: Last confusion matrix
        """
        return self.metrics_dict["Confusion Matrix"]["value"][-1]

    def add_tn_fp_sum(self):
        """
        Add sum of true negatives and false positives to metrics dictionary.
        """
        cm = self._get_last_confusion_matrix()
        tn = cm[0][0]
        try:
            fp = cm[0][1]
        except IndexError:
            fp = 0
        tn_fp_sum = tn + fp
        self.metrics_dict["TN-FP-Sum"]["value"].append(tn_fp_sum)

    def add_fpr(self):
        """
        Add false positive rate to metrics dictionary.
        """
        cm = self._get_last_confusion_matrix()
        tn = cm[0][0]
        try:
            fp = cm[0][1]
        except IndexError:
            fp = 0
        fpr = round(fp / (tn + fp), 4)
        self.metrics_dict["FPR"]["value"].append(fpr)

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
                return round(sum(metric_list) / len(metric_list), 4)
            elif stat_type == "std":
                return round(np.std(metric_list), 4)
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

    def add_confusion_matrix_average(
        self, mask: Optional[np.ndarray] = None, on_mean_data: bool = False
    ):
        """
        Add average confusion matrix to metrics dictionary. Consider only the last entries_to_consider entries.

        Args:
            mask: Mask for filtering
            on_mean_data: If True, calculate average on mean data
        """
        if on_mean_data:
            cm_list = self.metrics_dict["Confusion Matrix"]["mean"]
        else:
            cm_list = self.metrics_dict["Confusion Matrix"]["value"]

        # If specified: filter elements
        if mask is not None:
            cm_list = [value for value, include in zip(cm_list, mask) if include]

        cm_avg = sum(cm_list)
        self.metrics_dict["Confusion Matrix"]["mean"].append(cm_avg)

    def calculate_averages_across_random_states(self):
        self.add_random_state_avg("Total Average")
        self.add_metrics_stats(self.METRICS)
        self.add_confusion_matrix_average()

    def calculate_averages_per_hospitalid_across_random_states(self):
        for hospitalid in set(self.metrics_dict["Hospitalid"]["value"]):
            mask = self.metrics_dict["Hospitalid"]["value"] == hospitalid
            self.add_hospitalid_avg(hospitalid)
            self.add_metrics_stats(self.METRICS, mask)
            self.add_confusion_matrix_average(mask)

    def calculate_total_averages_across_hospitalids(self):
        self.add_hospitalid_avg("Total Average")
        self.add_metrics_stats(self.METRICS, on_mean_data=True)
        self.add_confusion_matrix_average(on_mean_data=True)

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
