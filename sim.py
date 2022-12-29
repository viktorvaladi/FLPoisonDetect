import os
import math

from model_ascent import create_model_ascent
from krum import Krum

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common.logger import log
from logging import WARNING
from functools import reduce
from client_manager import SimpleClientManager

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

from sim_app import start_simulation

from flwr.server.client_proxy import ClientProxy
from flwr.client.dpfedavg_numpy_client import DPFedAvgNumPyClient
from poison_detect import Poison_detect
from sim_client import FlwrClient
from sim_server import Server
from dpfedavg_adaptive import DPFedAvgAdaptive

import flwr as fl
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
from model import create_model
from cinic10_ds import get_train_ds, get_test_val_ds
import numpy as np
import random

NUM_CLIENTS = 3597
DATA = "femnist"
NUM_ROUNDS = 200
NUM_CPUS = 255
NUM_CLIENTS_PICK = 30
#STRATS = [['krum', 4, 1, True],['krum', 4, 1, False], ['krum', 4, 1.5, True],['krum', 4, 1.5, False], ['krum', 4, 2, True], ['krum', 4, 2, False], ['krum', 4, 2.5, True], ['krum', 4, 2.5, False], ['old', 4, 1, True],['old', 4, 1, False], ['old', 4, 1.5, True],['old', 4, 1.5, False], ['old', 4, 2, True], ['old', 4, 2, False], ['old', 4, 2.5, True], ['old', 4, 2.5, False], ['fedavg', 4, 1, True],['fedavg', 4, 1, False], ['fedavg', 4, 1.5, True],['fedavg', 4, 1.5, False], ['fedavg', 4, 2, True], ['fedavg', 4, 2, False], ['fedavg', 4, 2.5, True], ['fedavg', 4, 2.5, False]]
#STRATS = [['old', 4, 1, False, 5000], ['old', 4, 1, False, 2500], ['old', 4, 1, False, 1250], ['old', 4, 1, False, 600], ['old', 4, 1, False, 300]]
STRATS = [['fedavg', 0, 1, False, 620],['old', 0, 1, False, 620]]
def on_fit_config(server_round):
    return {
        'current_round': server_round,
        'nr_of_split_per_round' : 1,
        'epochs': 10,
        'rounds': 60,
    }

class StaticFunctions():
    @staticmethod
    def get_eval_fn(model,data):
        """Return an evaluation function for server-side evaluation."""

        x_test, y_test = get_test_val_ds(data)
        x_test = x_test[int(len(x_test)/2):int(len(x_test)-1)]
        y_test = y_test[int(len(y_test)/2):int(len(y_test)-1)]

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
    def get_eval_fn2(model, data):
        """Return an evaluation function for server-side evaluation."""

        x_test, y_test = get_test_val_ds(data)
        x_test = x_test[int(len(x_test)/2):int(len(x_test)-1)]
        y_test = y_test[int(len(y_test)/2):int(len(y_test)-1)]

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
            backdoor_count = 0
            backdoor_success = 0
            for i in range(len(preds)):
                pred = np.argmax(preds[i])
                true = np.argmax(y_test[i])
                spec_label_all_count[true] = spec_label_all_count[true] +1
                if true == pred:
                    spec_label_correct_count[true] = spec_label_correct_count[true] +1
                # for backdoor
                if true == 4:
                    backdoor_count += 1
                    if pred == 7:
                        backdoor_success += 1
            spec_label_accuracy = []
            for i in range(len(spec_label_all_count)):
                spec_label_accuracy.append(spec_label_correct_count[i]/spec_label_all_count[i])
            return loss, {"accuracy": accuracy}, spec_label_accuracy, spec_label_accuracy[7]

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

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, data, newold, no_val_elems, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data
        self.newold = newold
        self.poison_counts = {}
        self.total_counts = {}
        self.deviation_sum = {}
        self.acc_history = [[]]
        self.agg_label_final = []
        self.round = 0
        self.label_acc_history = []
        self.geti = self.fun(10000)
        self.pointList = {}
        self.mapPoisonClients = {}
        self.model = create_model(data)
        self.sum_threshold = 0
        self.evclient = StaticFunctions.get_eval_fn2(self.model, self.data)
        self.poison_detect = Poison_detect(2,3,1.5,10, self.data, fraction_boost_iid=0.6, newold=self.newold, val_elems=no_val_elems)
        self.run = 0
        self.agg_history = {}
        self.last_weights = self.model.get_weights()
        self.bd_history = []
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Optional[fl.common.NDArrays]:
        if len(results) > 0:
            self.round = server_round
            # evaluates all nodes accuracy and saves in nodes_acc as {nodeName : accuracy}
            # also saves accuracy in a list to count variance
            for i in range(len(results)):
                self.total_counts[results[i][0]] = self.total_counts.get(results[i][0],0)+1
                if (self.mapPoisonClients.get(results[i][0]) is None):
                    self.mapPoisonClients[results[i][0]] = results[i][1].metrics.get("is_poisoned")
            # calculate variance for the current round
        part_agg, weights_to_add = self.poison_detect.calculate_partitions(results, self.last_weights, server_round)
        print("PART AGGREGATION DICT HERE!!!!!!")
        for elem in part_agg:
            if elem in self.agg_history:
                self.agg_history[elem].append(part_agg.get(elem))
            else:
                self.agg_history[elem] = [part_agg.get(elem)]
        if self.newold == "lfr":
            aggregated_weights = self.aggregate_fit2(server_round, results, part_agg, weights_to_add, failures)
        if self.newold == "new" or self.newold == "old":
            aggregated_weights = self.aggregate_fit2(server_round, results, part_agg, weights_to_add, failures)
        if self.newold == "fedprox" or self.newold == "fedavg":
            aggregated_weights = super().aggregate_fit(server_round, results, failures)

        self.last_weights = parameters_to_ndarrays(aggregated_weights[0])

        _,lastacc, agg_label_acc, bd = self.evclient(parameters_to_ndarrays(aggregated_weights[0]))
        self.bd_history.append(bd)
        print(f"backdoor history: {self.bd_history}")
        print('accuracy here! :)')
        self.acc_history[self.run].append(lastacc.get('accuracy'))
        sum_run_last = 0
        for elem in self.acc_history:
            sum_run_last += elem[-1]
        print('average final accuracy!:) :')
        print(sum_run_last/len(self.acc_history))
        np.savetxt('test.out', self.acc_history, delimiter=',')
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {server_round} aggregated_weights...")
            #np.savez(f"round-{server_round}-weights.npz", *aggregated_weights)
            
            #print accuracy and variance and poison/total visists for clients
            if server_round % 10000 == 0 and server_round != 0:
                self.model = create_model(self.data)
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
            self.totPoisCleanPrint(self.total_counts, agg_label_acc)
            print(f'acc history: {self.acc_history}')
        return aggregated_weights
    
    def aggregate_fit2(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        part_agg,
        weights_to_add,
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        if self.newold == "lfr":
            no_clients = len(results)
            remove = int(round(no_clients*0.3))
            per_client = 1/(no_clients-remove)
            losses = {}
            part_agg = {}
            for elem in results:
                loss, _, _, _ = self.evclient(elem[1].parameters)
                losses[elem[0]] = loss
                part_agg[elem[0]] = per_client
            while remove>0:
                most = 0
                target = ":)"
                for elem in losses:
                    if losses[elem] > most:
                        most = losses[elem]
                        target = elem
                losses[target] = 0
                part_agg[target] = 0
                remove -= 1

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), part_agg.get(name))
            for name, fit_res in results
        ]
        aggregated = self.aggregate2(weights_results)
        if self.newold == "new":
            for i in range(len(aggregated)):
                for elem in weights_to_add:
                    aggregated[i] = np.add(aggregated[i], elem[i])
            
        parameters_aggregated = ndarrays_to_parameters(aggregated)

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
    
    def totPoisCleanPrint(self,totDict, agg_label):
        for elem in totDict:
            if self.pointList.get(elem) == None:
                self.pointList[elem] = next(self.geti)
            print(f"client {self.pointList.get(elem)} is_poisoned = {self.mapPoisonClients.get(elem)} :")
            print("agg_history :")
            print(self.agg_history.get(elem))
            print(f"mean: {np.mean(self.agg_history.get(elem))}")
            #print(f"individual label: {ind_label.get(elem)}")
        print("aggregated indivudal label accuracy: ")
        print(agg_label)


    def fun(self,x):
        n = 0
        while n < x:
            yield n
            n += 1

def get_client_fn(strat, no_poison, pgascaler, dp):
    def client_fn(cid: str) -> fl.client.Client:
        # Load model
        model = create_model(DATA)
        model_ascent = create_model(DATA)
        
        poisoned_list = [i for i in range(no_poison)]
        is_poisoned = False
        if int(cid) in poisoned_list:
            is_poisoned = True
        
        noniid_list = [int(i*NUM_CLIENTS/NUM_CLIENTS_PICK) for i in range(12)]
        is_noniid = False
        if int(cid) in noniid_list:
            is_noniid = True

        # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
        x_train, y_train = get_train_ds(NUM_CLIENTS, int(cid), DATA)
        #x_test, y_test = get_test_val_ds(DATA)
        x_test, y_test = x_train, y_train

        # Create and return client
        client = FlwrClient(model, model_ascent, x_train, y_train, x_test, y_test, is_poisoned, is_noniid, strat, pgascaler, DATA)
        dpClient = None
        if dp:
            dpClient = DPFedAvgNumPyClient(client=client)
        return dpClient if dpClient is not None else client
    return client_fn

def main() -> None:
    # Start Flower simulation
    res = {}
    for elem in STRATS:
        model = create_model(DATA)
        dpstrat = None
        if elem[0] == "krum":
            strat = Krum(
                num_malicious_clients=int(0.5*NUM_CLIENTS_PICK),
                num_clients_to_keep=int(0.5*NUM_CLIENTS_PICK),
                initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
                on_evaluate_config_fn=StaticFunctions.evaluate_config,
                min_fit_clients=NUM_CLIENTS_PICK,
                min_available_clients=NUM_CLIENTS,
                fraction_fit=0.1,
                fraction_evaluate=0.0,
                evaluate_fn=StaticFunctions.get_eval_fn(model, DATA),
                on_fit_config_fn=on_fit_config,
                evclient=StaticFunctions.get_eval_fn2(model, DATA),
                )
        else:
            strat = SaveModelStrategy(
                    data=DATA,
                    newold = elem[0],
                    initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
                    on_evaluate_config_fn=StaticFunctions.evaluate_config,
                    min_fit_clients=NUM_CLIENTS_PICK,
                    min_available_clients=NUM_CLIENTS,
                    fraction_fit=0.1,
                    fraction_evaluate=0.0,
                    evaluate_fn=StaticFunctions.get_eval_fn(model, DATA),
                    on_fit_config_fn=on_fit_config,
                    no_val_elems=elem[4],
                    )
        if elem[3]:
            dpstrat = DPFedAvgAdaptive(strategy=strat,
            num_sampled_clients=NUM_CLIENTS_PICK,
            server_side_noising=False,
            init_clip_norm=50.0,
            noise_multiplier=0.0005,
            clip_norm_target_quantile=0.5,
            )
        serv = Server(client_manager=SimpleClientManager(), strategy=dpstrat if dpstrat is not None else strat)
        start_simulation(
            client_fn=get_client_fn(elem[0],elem[1],elem[2],elem[3]),
            num_clients=NUM_CLIENTS,
            client_resources={"num_cpus": 1, "num_gpus": 0.2},
            ray_init_args= {"num_gpus" : 2},
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=dpstrat if dpstrat is not None else strat,
            server=serv,
        )
        print(elem[0]+str(elem[2])+str(elem[3])+str(elem[4]))
        res[elem[0]+str(elem[2])+str(elem[3])+str(elem[4])] = strat.acc_history
        np.savez('res.npz',**res)
    
    for elem in res:
        print(f"{elem} = {res[elem]}")

if __name__ == "__main__":
    main()