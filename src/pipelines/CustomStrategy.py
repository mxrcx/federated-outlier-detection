from logging import WARNING
from functools import reduce
from typing import List, Tuple, Union, Optional, Dict, Any, cast
from io import BytesIO
import numpy as np
import json
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    NDArray,
    NDArrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg

class CustomStrategy(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = self.aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (self.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    
        '''
        # Extract the parameters from each client
        parameters_list = [fit_res.parameters for _, fit_res in results]
        
          # Debug: Print the type and content of the parameters
        for i, params in enumerate(parameters_list):
            print(f"Client {i} parameters type: {type(params)}")
            print(f"Client {i} parameters content: {params}")
        
        
        # Convert parameters to ndarrays
        ndarrays_list = [self.parameters_to_ndarrays(params) for params in parameters_list]
        
        # Aggregate the ndarrays
        aggregated_ndarrays = self.aggregate_ndarrays(ndarrays_list)
        
        # Convert the aggregated ndarrays back to Parameters
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        
        # Aggregate custom metrics if required
        aggregated_metrics = self.aggregate_metrics([fit_res.metrics for _, fit_res in results])
        
        return aggregated_parameters, aggregated_metrics
        '''
        
    def aggregate_inplace(self, results: List[Tuple[ClientProxy, FitRes]]) -> NDArrays:
        """Compute in-place weighted average."""
        # Count total examples
        num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)

        # Compute scaling factors for each result
        scaling_factors = [
            fit_res.num_examples / num_examples_total for _, fit_res in results
        ]

        # Let's do in-place aggregation
        # Get first result, then add up each other
        params = [
            scaling_factors[0] * x for x in self.parameters_to_ndarrays(results[0][1].parameters)
        ]
        for i, (_, fit_res) in enumerate(results[1:]):
            res = (
                scaling_factors[i + 1] * x
                for x in self.parameters_to_ndarrays(fit_res.parameters)
            )
            params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]

        return params

    
    def parameters_to_ndarrays(self, parameters: Parameters) -> List[np.ndarray]:
        # Implement your custom logic to convert parameters to ndarrays
        # This example assumes that parameters are a list of numpy arrays
        
        def bytes_to_ndarray(tensor: bytes) -> NDArray:
            """Deserialize NumPy ndarray from bytes."""
            bytes_io = BytesIO(tensor)
            # WARNING: NEVER set allow_pickle to true.
            # Reason: loading pickled data can execute arbitrary code
            # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
            #json_deserialized = json.loads(tensor.decode('utf-8'))
            #ndarray_deserialized = np.array(json_deserialized)
            ndarray_deserialized = np.load(bytes_io, allow_pickle=True)
            return cast(NDArray, ndarray_deserialized)
        
        liste = [bytes_to_ndarray(tensor) for tensor in parameters.tensors]
        print(f"yoooooooooo {liste}")
    
    def aggregate_ndarrays(self, ndarrays_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        # Implement your custom ndarray aggregation logic here
        # This example simply returns the average of the ndarrays
        return [np.mean(np.array(ndarrays), axis=0) for ndarrays in zip(*ndarrays_list)]
    
    def aggregate_metrics(self, metrics: List[Dict[str, Scalar]]) -> Dict[str, Scalar]:
        # Implement your custom metric aggregation logic here
        # This example simply returns the average of each metric
        aggregated_metrics = {}
        for metric_name in metrics[0].keys():
            metric_values = [metric[metric_name] for metric in metrics]
            aggregated_metrics[metric_name] = sum(metric_values) / len(metric_values)
        return aggregated_metrics