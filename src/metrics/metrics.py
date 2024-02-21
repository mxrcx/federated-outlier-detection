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
        Add hospital ID to metrics dictionary.

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
        """
        cm = confusion_matrix(y_true, y_pred)
        self.metrics_dict["Confusion Matrix"]["value"].append(cm)

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
        entries_to_consider: Optional[int] = None,
        on_mean_data: bool = False,
    ) -> np.ndarray:
        """
        Get metric list from metrics dictionary.

        Args:
            metric: Metric name
            entries_to_consider: Number of last entries to consider
            on_mean_data: If True, get metric list from mean data

        Returns:
            np.ndarray: Metric list
        """
        if on_mean_data:
            metric_list = self.metrics_dict[metric]["mean"]
        else:
            metric_list = self.metrics_dict[metric]["value"]

        # If specified: Only consider last entries_to_consider elements
        if entries_to_consider is not None:
            metric_list = metric_list[-entries_to_consider:]

        # Remove "Not defined" elements
        metric_list = [metric for metric in metric_list if metric != "Not defined"]

        # Remove "No Sepsis Occurences" elements
        metric_list = [
            metric for metric in metric_list if metric != "No Sepsis Occurences"
        ]
        
        # Remove "Mean calculation not possible" elements
        metric_list = [
            metric for metric in metric_list if metric != "Mean calculation not possible"
        ]
        
        # Remove "Std calculation not possible" elements
        metric_list = [
            metric for metric in metric_list if metric != "Std calculation not possible"
        ]
        
        print(metric_list)

        return metric_list

    def add_metric_mean(
        self,
        metric: str,
        metric_list: List[float],
    ) -> np.ndarray:
        """
        Add metric mean to metrics dictionary. Consider only the last entries_to_consider entries.

        Args:
            metric: Metric name
            metric_list: List of metric values
        """
        try:
            metric_mean = round(sum(metric_list) / len(metric_list), 4)
        except Exception:
            metric_mean = "Mean calculation not possible"

        self.metrics_dict[metric]["mean"].append(metric_mean)

    def add_metrics_mean(
        self,
        selected_metrics: List[str],
        entries_to_consider: Optional[int] = None,
        on_mean_data: bool = False,
    ):
        """
        Add mean to metrics dictionary for selected metrics. Consider only the last entries_to_consider entries.

        Args:
            selected_metrics: List of selected metrics.
            entries_to_consider: Number of last entries to consider
            on_mean_data: If True, calculate mean on mean data
        """
        for metric in selected_metrics:
            metric_list = self._get_metric_list(
                metric, entries_to_consider, on_mean_data
            )
            self.add_metric_mean(metric, metric_list)

    def add_metric_std(
        self,
        metric: str,
        metric_list: List[float],
    ) -> np.ndarray:
        """
        Add metric standard deviation to metrics dictionary. Consider only the last entries_to_consider entries.

        Args:
            metric: Metric name
            metric_list: List of metric values
        """
        try:
            metric_std = round(np.std(metric_list), 4)
        except Exception:
            metric_std = "Std calculation not possible"

        self.metrics_dict[metric]["std"].append(metric_std)

    def add_metrics_std(
        self,
        selected_metrics: List[str],
        entries_to_consider: Optional[int] = None,
        on_mean_data: bool = False,
    ):
        """
        Add standard deviation to metrics dictionary for selected metrics. Consider only the last entries_to_consider entries.

        Args:
            selected_metrics: List of selected metrics.
            entries_to_consider: Number of last entries to consider
            on_mean_data: If True, calculate standard deviation on mean data
        """
        for metric in selected_metrics:
            metric_list = self._get_metric_list(
                metric, entries_to_consider, on_mean_data
            )
            self.add_metric_std(metric, metric_list)

    def add_confusion_matrix_average(
        self, entries_to_consider: Optional[int] = None, on_mean_data: bool = False
    ):
        """
        Add average confusion matrix to metrics dictionary. Consider only the last entries_to_consider entries.

        Args:
            entries_to_consider: Number of last entries to consider
            on_mean_data: If True, calculate average on mean data
        """
        if on_mean_data:
            cm_list = self.metrics_dict["Confusion Matrix"]["mean"]
        else:
            cm_list = self.metrics_dict["Confusion Matrix"]["value"]

        # If specified: Only consider last entries_to_consider elements
        if entries_to_consider is not None:
            cm_list = cm_list[-entries_to_consider:]

        cm_avg = sum(cm_list)
        self.metrics_dict["Confusion Matrix"]["mean"].append(cm_avg)

    def get_metrics_value_dataframe(self, selected_metrics: List[str]) -> pd.DataFrame:
        """
        Return a metrics value dataframe from the metrics dictionary. Include only the specified metrics.

        Args:
            selected_metrics: List of selected metrics.

        Returns:
            pd.DataFrame: The metrics dataframe.
        """
        # Create a dictionary to save the filtered metrics data
        filtered_metrics_value_dict = {}

        # Iterate over the outer keys and extract the sublevels for each key
        for key, value in self.metrics_dict.items():
            if key in selected_metrics:
                if "value" in value:
                    filtered_metrics_value_dict[key] = value["value"]

        return pd.DataFrame(filtered_metrics_value_dict)

    def get_metrics_avg_dataframe(self, selected_metrics: List[str]) -> pd.DataFrame:
        """
        Return a metrics average dataframe from the metrics dictionary. Include only the specified metrics.

        Args:
            selected_metrics: List of selected metrics.

        Returns:
            pd.DataFrame: The metrics dataframe.
        """
        # Create a dictionary to save the filtered metrics data
        filtered_metrics_avg_dict = {}

        # Iterate over the outer keys and extract the sublevels for each key
        for key, value in self.metrics_dict.items():
            if key in selected_metrics:
                if "mean" in value:
                    filtered_metrics_avg_dict[key + " Mean"] = value["mean"]
                if "std" in value:
                    filtered_metrics_avg_dict[key + " Std"] = value["std"]

        return pd.DataFrame(filtered_metrics_avg_dict)
