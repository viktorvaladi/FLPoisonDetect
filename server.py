import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from typing import Optional, Tuple, List
import flwr as fl
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common.logger import log
from logging import WARNING
from functools import reduce

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from cinic10_ds import get_test_val_ds
from model import create_model
import argparse
from poison_detect import Poison_detect

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poison_counts = {}
        self.total_counts = {}
        self.deviation_sum = {}
        self.acc_history = [[]]
        self.agg_label_final = []
        self.round = 0
        self.vars = []
        self.label_acc_history = []
        self.geti = self.fun(1000)
        self.pointList = {}
        self.mapPoisonClients = {}
        self.model = create_model()
        self.sum_threshold = 0
        self.evclient = FLServer.get_eval_fn2(self.model)
        self.poison_detect = Poison_detect(1.5,1.5,1.5,6)
        self.run = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Optional[fl.common.NDArrays]:
        if len(results) > 0:
            self.round = server_round
            nodes_acc = {}
            label_acc_dict = {}
            varcounter = []
            # evaluates all nodes accuracy and saves in nodes_acc as {nodeName : accuracy}
            # also saves accuracy in a list to count variance
            for i in range(len(results)):
                _,acc,lab_acc = self.evclient(parameters_to_ndarrays(results[i][1].parameters))
                label_acc_dict[results[i][0]] = lab_acc
                nodes_acc[results[i][0]] = acc.get('accuracy')
                varcounter.append(acc.get('accuracy'))
                self.total_counts[results[i][0]] = self.total_counts.get(results[i][0],0)+1
                if (self.mapPoisonClients.get(results[i][0]) is None):
                    self.mapPoisonClients[results[i][0]] = results[i][1].metrics.get("is_poisoned")
            # calculate variance for the current round
            mean = sum(varcounter) / len(varcounter)
            self.vars.append(sum((i - mean) ** 2 for i in varcounter) / len(varcounter))
        part_agg = self.poison_detect.calculate_partitions(results)
        print("PART AGGREGATION DICT HERE!!!!!!")
        print(part_agg)
        aggregated_weights = self.aggregate_fit2(server_round, results, part_agg, failures)
        _,lastacc, agg_label_acc = self.evclient(parameters_to_ndarrays(aggregated_weights[0]))
        print('accuracy here! :)')
        self.acc_history[self.run].append(lastacc.get('accuracy'))
        print(f'acc history: {self.acc_history}')
        sum_run_last = 0
        for elem in self.acc_history:
            sum_run_last += elem[-1]
        print('average final accuracy!:) :')
        print(sum_run_last/len(self.acc_history))
        np.savetxt('test.out', [sum_run_last/len(self.acc_history)], delimiter=',')
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {server_round} aggregated_weights...")
            #np.savez(f"round-{server_round}-weights.npz", *aggregated_weights)
            
            #print accuracy and variance and poison/total visists for clients
            if server_round % 20 == 0 and server_round != 0:
                self.model = create_model()
                aggregated_weights = (ndarrays_to_parameters(self.model.get_weights()), {})
                self.run = self.run+1
                self.acc_history.append([])
                self.agg_label_final.append(agg_label_acc)
                agg_label_avg = None
                for elem in self.agg_label_final:
                    if agg_label_avg is None:
                        agg_label_avg = elem
                    else:
                        for i in range(len(elem)):
                            agg_label_avg[i] = agg_label_avg[i] + elem[i]
                for i in range(len(agg_label_avg)):
                    agg_label_avg[i] = agg_label_avg[i]/len(self.agg_label_final)
                np.savetxt('agg_label_acc_avg.out', agg_label_avg, delimiter=',')
                print('AGG LABEL AVERAGE ALL TURNS!!! :')
                print(agg_label_avg)
            self.totPoisCleanPrint(self.total_counts, label_acc_dict, agg_label_acc)
        return aggregated_weights
    
    def aggregate_fit2(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        part_agg,
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), part_agg.get(name))
            for name, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(self.aggregate2(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
                    
        return parameters_aggregated, metrics_aggregated
    
    def aggregate2(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime
    
    def totPoisCleanPrint(self,totDict, ind_label, agg_label):
        for elem in totDict:
            if self.pointList.get(elem) == None:
                self.pointList[elem] = next(self.geti)
            print(f"individual label: {ind_label.get(elem)}")
        print("aggregated indivudal label accuracy: ")
        print(agg_label)


    def fun(self,x):
        n = 0
        while n < x:
            yield n
            n += 1


class FLServer:
    def __init__(self, rounds, epochs, nr_of_split_per_round):
        self.rounds = rounds
        self.epochs = epochs
        self.nr_of_split_per_round = nr_of_split_per_round

        model = create_model()
        model.summary()

        self.strategy = SaveModelStrategy(
            initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
            on_evaluate_config_fn=FLServer.evaluate_config,
            min_fit_clients=10,
            min_available_clients=10,
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            evaluate_fn=FLServer.get_eval_fn(model),
            on_fit_config_fn=self.on_fit_config,
        )

    def on_fit_config(self, server_round):
        return {
            'current_round': server_round,
            'nr_of_split_per_round' : self.nr_of_split_per_round,
            'epochs': self.epochs,
            'rounds': self.rounds,
        }

    def start(self):
        fl.server.start_server(config=fl.server.ServerConfig(num_rounds=self.rounds), strategy=self.strategy)

    @staticmethod
    def get_eval_fn(model):
        """Return an evaluation function for server-side evaluation."""

        x_test, y_test = get_test_val_ds()
        x_test = x_test[2501:4999]
        y_test = y_test[2501:4999]

        # Load data and model here to avoid the overhead of doing it in `evaluate` itself
        #(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data() #Change to the right dataset
        #x_train = x_train.astype('float32')
        #y_train = np_utils.to_categorical(y_train, 10)
        # Use the last 5k training examples as a validation set
        #x_val, y_val = x_train[45000:50000], y_train[45000:50000]

        # The `evaluate` function will be called after every round
        def evaluate(server_round: int, weights: fl.common.NDArrays, dict) -> Optional[Tuple[float, float]]:
            model.set_weights(weights)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(x_test,y_test)
            return loss, {"accuracy": accuracy}

        return evaluate
    
    @staticmethod
    def get_eval_fn2(model):
        """Return an evaluation function for server-side evaluation."""

        x_test, y_test = get_test_val_ds()
        x_test = x_test[2501:4999]
        y_test = y_test[2501:4999]

        # Load data and model here to avoid the overhead of doing it in `evaluate` itself
        #(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data() #Change to the right dataset
        #x_train = x_train.astype('float32')
        #y_train = np_utils.to_categorical(y_train, 10)
        # Use the last 5k training examples as a validation set
        #x_val, y_val = x_train[45000:50000], y_train[45000:50000]

        # The `evaluate` function will be called after every round
        def evaluate(weights: fl.common.NDArrays) -> Optional[Tuple[float, float]]:
            model.set_weights(weights)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(x_test,y_test)
            preds = model.predict(x_test)
            spec_label_correct_count = [0.0 for i in range(len(y_test[0]))]
            spec_label_all_count = [0.0 for i in range(len(y_test[0]))]
            for i in range(len(preds)):
                pred = np.argmax(preds[i])
                true = np.argmax(y_test[i])
                spec_label_all_count[true] = spec_label_all_count[true] +1
                if true == pred:
                    spec_label_correct_count[true] = spec_label_correct_count[true] +1
            spec_label_accuracy = []
            for i in range(len(spec_label_all_count)):
                spec_label_accuracy.append(spec_label_correct_count[i]/spec_label_all_count[i])
            return loss, {"accuracy": accuracy}, spec_label_accuracy

        return evaluate

    @staticmethod
    def evaluate_config(server_round: int):
        """Return evaluation configuration dict for each round.
        Perform five local evaluation steps on each client (i.e., use five
        batches) during rounds one to three, then increase to ten local
        evaluation steps.
        """
        val_steps = 5 if server_round < 4 else 10
        return {"val_steps": val_steps}