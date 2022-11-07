import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from typing import Optional, Tuple, List
import flwr as fl
import numpy as np
from flwr.common import parameters_to_ndarrays
from cinic10_ds import get_test_val_ds
from model import create_model
import math
from ray.util.multiprocessing import Pool

def multiprocess_evaluate(data, model, weights, x_test, y_test):
    model.set_weights(weights)  # Update model with the latest parameters
    loss, accuracy = model.evaluate(x_test,y_test)
    preds = model.predict(x_test)
    spec_label_correct_count = [0.0 for i in range(len(y_test[0]))]
    spec_label_all_count = [0.0 for i in range(len(y_test[0]))]
    spec_label_loss_count = [0.0 for i in range(len(y_test[0]))]
    for i in range(len(preds)):
        pred = np.argmax(preds[i])
        true = np.argmax(y_test[i])
        spec_label_all_count[true] = spec_label_all_count[true] +1
        spec_label_loss_count[true] += -(math.log(max(preds[i][true],0.0001)))
        if true == pred:
            spec_label_correct_count[true] = spec_label_correct_count[true] +1
    spec_label_accuracy = []
    spec_label_loss = []
    for i in range(len(spec_label_all_count)):
        spec_label_accuracy.append(spec_label_correct_count[i]/spec_label_all_count[i])
        spec_label_loss.append(spec_label_loss_count[i]/spec_label_all_count[i])
    return np.mean(spec_label_loss), {"accuracy": accuracy}, spec_label_accuracy, spec_label_loss

class Poison_detect:
    # md_factor determines by how much more we want to favor the stronger client updates
    def __init__(self, md_overall = 1.5, md_label = 1.5, md_heterogenous = 1.5, ld = 4, data="cifar10", num_cpus=8):
        self.num_cpus = num_cpus
        self.data = data
        self.model = create_model(self.data)
        self.evclient = Poison_detect.get_eval_fn(self.model, self.data)
        x_test, y_test = get_test_val_ds(self.data)
        self.no_labels = len(y_test[0])
        self.x_test = x_test[0:int(len(x_test)/2)]
        self.y_test = y_test[0:int(len(y_test)/2)]
        self.md_overall = md_overall
        self.md_label = md_label
        self.md_heterogenous = md_heterogenous
        self.ld = ld
    
    def calculate_partitions(self,results, last_agg_w, round_nr):
        # part_agg will be between 0-1 how big part each client should take of aggregation
        part_agg = {}
        label_acc_dict, nodes_acc, loss_dict, label_loss_dict, last_loss, last_label_loss = self.calculate_accs(results, last_agg_w, round_nr)
        points, overall_mean = self.get_points_overall(loss_dict, results)
        points = self.get_points_label(label_loss_dict, results, overall_mean, points, last_loss, last_label_loss, round_nr)
        #make sure no client has negative points
        for elem in points:
            points[elem] = max(0,points[elem])
        
        sum_points = 0
        for elem in points:
            sum_points += points[elem]
        for elem in points:
            part_agg[elem] = (points[elem] / sum_points)
        return part_agg
        

    def get_points_overall(self, nodes_acc, results, points = {}):
        #overall points
        # calculate mean absolute deviation for middle 80% of clients
        mean_calc = []
        for elem in nodes_acc:
            mean_calc.append(nodes_acc[elem])
        mean = np.mean(mean_calc)
        all_for_score = []
        for elem in mean_calc:
            #if loss then (mean - elem), if accuracy (mean - elem)
            all_for_score.append(mean - elem)
        mad_calc = all_for_score.copy()
        for i in range(len(mad_calc)):
            mad_calc[i] = abs(mad_calc[i])
        no_elems = round(0.8*len(mad_calc))
        mad_calc.sort()
        mad_calc = mad_calc[:no_elems]
        mad = np.mean(mad_calc)
        slope = 10/(self.md_overall*mad)
        for i in range(len(all_for_score)):
            points[results[i][0]] = points.get(results[i][0],0) + (slope*all_for_score[i]+10)
        #individual label points
        return points, mean
    
    def get_points_label(self, label_acc_dict, results, overall_mean, points, last_loss, last_label_loss, round_nr):
        #individual label points
        for i in range(self.no_labels):
            #calculate mean for label i
            mean_calc = []
            for elem in label_acc_dict:
                mean_calc.append(label_acc_dict.get(elem)[i])
            ## move rest of loop from here to its own function with param (self.md_label,mean_calc)
            mean = np.mean(mean_calc)
            #deviation from mean for label i
            all_for_score = []
            for elem in mean_calc:
                #if loss then (mean - elem), if accuracy (mean - elem)
                all_for_score.append(mean - elem)
            mad_calc = all_for_score.copy()
            for j in range(len(mad_calc)):
                mad_calc[j] = abs(mad_calc[j])
            no_elems = round(0.8*len(mad_calc))
            mad_calc.sort()
            mad_calc = mad_calc[:no_elems]
            mad = np.mean(mad_calc)
            slope = 10/(self.md_label*mad)

            dif = (mean - overall_mean)
            x = ((overall_mean+dif)/overall_mean)
            factor = x**self.ld
            for k in range(len(all_for_score)):
                points[results[k][0]] = points.get(results[k][0],0) + (max(1,factor)*slope*all_for_score[k]+10)
        return points

    def par_results_ev(self, result):
        loss, acc, lab_acc,lab_loss = multiprocess_evaluate(self.data, self.model, parameters_to_ndarrays(result[1].parameters), self.x_test, self.y_test)
        return [result[0], loss, acc, lab_acc, lab_loss]

    #calculates accuracy for each client and return two dicts for label acc and overall acc.
    #calculates variance in data
    # TODO add heterogenity?
    def calculate_accs(self, results, last_weights, round_nr):
        label_acc_dict = {}
        nodes_acc = {}
        loss_dict = {}
        label_loss_dict = {}
        pool = Pool(ray_address="auto")
        evaluated = pool.map(self.par_results_ev, results)
        for elem in evaluated:
            label_acc_dict[elem[0]] = elem[3]
            nodes_acc[elem[0]] = elem[2].get('accuracy')
            loss_dict[elem[0]] = elem[1]
            label_loss_dict[elem[0]] = elem[4]
        #redundant:)
        last_loss = 0
        last_label_loss = 0
        return label_acc_dict, nodes_acc, loss_dict, label_loss_dict, last_loss, last_label_loss

    #calculate accuracies an client
    # TODO add heterogenity?
    # TODO add more metrics - loss
    @staticmethod
    def get_eval_fn(model, data):
        """Return an evaluation function for server-side evaluation."""

        x_test, y_test = get_test_val_ds(data)
        x_test = x_test[0:int(len(x_test)/2)]
        y_test = y_test[0:int(len(y_test)/2)]

        # The `evaluate` function will be called after every round
        def evaluate(weights: fl.common.NDArrays) -> Optional[Tuple[float, float]]:
            model.set_weights(weights)  # Update model with the latest parameters
            loss, accuracy = model.evaluate(x_test,y_test)
            preds = model.predict(x_test)
            spec_label_correct_count = [0.0 for i in range(len(y_test[0]))]
            spec_label_all_count = [0.0 for i in range(len(y_test[0]))]
            spec_label_loss_count = [0.0 for i in range(len(y_test[0]))]
            for i in range(len(preds)):
                pred = np.argmax(preds[i])
                true = np.argmax(y_test[i])
                spec_label_all_count[true] = spec_label_all_count[true] +1
                spec_label_loss_count[true] += -(math.log(max(preds[i][true],0.0001)))
                if true == pred:
                    spec_label_correct_count[true] = spec_label_correct_count[true] +1
            spec_label_accuracy = []
            spec_label_loss = []
            for i in range(len(spec_label_all_count)):
                spec_label_accuracy.append(spec_label_correct_count[i]/spec_label_all_count[i])
                spec_label_loss.append(spec_label_loss_count[i]/spec_label_all_count[i])
            print('mean loss here!!')
            print(np.mean(spec_label_loss))
            return np.mean(spec_label_loss), {"accuracy": accuracy}, spec_label_accuracy, spec_label_loss
        return evaluate
