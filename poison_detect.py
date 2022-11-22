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
import copy

def multiprocess_evaluate(data, model, weights, x_test, y_test):
    model.set_weights(weights)  # Update model with the latest parameters
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
    all_sum = 0
    all_acc_correct = 0
    all_loss_correct = 0
    for i in range(len(spec_label_all_count)):
        all_sum += spec_label_all_count[i]
        spec_label_accuracy.append(spec_label_correct_count[i]/spec_label_all_count[i])
        all_acc_correct += spec_label_correct_count[i]
        spec_label_loss.append(spec_label_loss_count[i]/spec_label_all_count[i])
        all_loss_correct += spec_label_loss_count[i]
    return all_loss_correct/all_sum, {"accuracy": all_acc_correct/all_sum}, spec_label_accuracy, spec_label_loss

class Poison_detect:
    # md_factor determines by how much more we want to favor the stronger client updates
    def __init__(self, md_overall = 1.5, md_label = 1.5, md_heterogenous = 1.5, ld = 4, data="cifar10", fraction_boost_iid=0.2, newold = "new"):
        self.fraction_boost_iid = fraction_boost_iid
        self.newold = newold
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
        self.pre_reset_ld = ld
    
    def calculate_partitions(self,results, last_agg_w, round_nr):
        if self.newold == "fedprox" or self.newold == "fedavg":
            asd = {}
            for elem in results:
                asd[elem[0]] = 0
            return asd, []
        # part_agg will be between 0-1 how big part each client should take of aggregation
        label_acc_dict, nodes_acc, loss_dict, label_loss_dict, last_loss, last_label_loss = self.calculate_accs(results, last_agg_w, round_nr)
        adaptiveMdAccs = []
        adaptiveMdDicts = []
        adaptiveMdTests = []
        adaptiveMdLAccs = []
        adaptiveMdLDicts = []
        adaptiveMdLTests = []
        adaptiveLdAccs = []
        adaptiveLdDicts = []
        # remove all and keep Ld with (reset and reset back to pre reset)
        adaptiveLdTests = [self.ld, max(1,self.ld-0.5), self.ld+0.5, 3, self.pre_reset_ld]
        i = 0
        for elem in adaptiveMdTests:
            self.md_overall = elem
            points = {}
            points, overall_mean = self.get_points_overall(loss_dict, results, points=points)
            points = self.get_points_label(label_loss_dict, results, overall_mean, points, last_loss, last_label_loss, round_nr)
            points_iid = {}
            points_iid = self.get_points_iid(label_loss_dict, results, overall_mean, points_iid, last_loss, last_label_loss, round_nr)
            part_agg = self.points_to_parts(points)
            if self.newold == "new":
                part_agg_iid = self.points_to_parts(points_iid)
                list_norms_to_add = self.norms_from_parts(part_agg_iid, results, last_agg_w)
            else:
                list_norms_to_add = []
            agg_copy_weights = self.agg_copy_weights(results, part_agg, last_agg_w)
            _, loss, _, _ = self.evclient(agg_copy_weights)
            adaptiveMdDicts.append(part_agg)
            adaptiveMdAccs.append(loss) 
            print(f"acc on {i}: {acc}")
            i = i+1
        
        idx_max = np.argmin(adaptiveMdAccs)
        self.md_overall = adaptiveMdTests[idx_max]
        print(f"self.md is now: {self.md_overall}")

        i = 0
        for elem in adaptiveMdLTests:
            self.md_label = elem
            points = {}
            points, overall_mean = self.get_points_overall(loss_dict, results, points=points)
            points = self.get_points_label(label_loss_dict, results, overall_mean, points, last_loss, last_label_loss, round_nr)
            points_iid = {}
            points_iid = self.get_points_iid(label_loss_dict, results, overall_mean, points_iid, last_loss, last_label_loss, round_nr)
            part_agg = self.points_to_parts(points)
            if self.newold == "new":
                part_agg_iid = self.points_to_parts(points_iid)
                list_norms_to_add = self.norms_from_parts(part_agg_iid, results, last_agg_w)
            else:
                list_norms_to_add = []
            agg_copy_weights = self.agg_copy_weights(results, part_agg, last_agg_w)
            _, loss, _, _ = self.evclient(agg_copy_weights)
            adaptiveMdLDicts.append(part_agg)
            adaptiveMdLAccs.append(loss) 
            print(f"acc on {i}: {acc}")
            i = i+1
        
        idx_max = np.argmin(adaptiveMdLAccs)
        self.md_label = adaptiveMdLTests[idx_max]
        print(f"self.mdl is now: {self.md_label}")

        i = 0
        for elem in adaptiveLdTests:
            self.ld = elem
            points = {}
            points, overall_mean = self.get_points_overall(loss_dict, results, points=points)
            points = self.get_points_label(label_loss_dict, results, overall_mean, points, last_loss, last_label_loss, round_nr)
            points_iid = {}
            points_iid = self.get_points_iid(label_loss_dict, results, overall_mean, points_iid, last_loss, last_label_loss, round_nr)
            part_agg = self.points_to_parts(points)
            if self.newold == "new":
                part_agg_iid = self.points_to_parts(points_iid)
                list_norms_to_add = self.norms_from_parts(part_agg_iid, results, last_agg_w)
            else:
                list_norms_to_add = []
            agg_copy_weights = self.agg_copy_weights(results, part_agg, last_agg_w)
            loss, acc, _, _ = self.evclient(agg_copy_weights)
            adaptiveLdDicts.append(part_agg)
            adaptiveLdAccs.append(loss)
            print(f"acc on {i}: {acc}")
            i = i+1
        idx_max = np.argmin(adaptiveLdAccs)
        if idx_max == 3:
            self.pre_reset_ld = self.ld
        self.ld = adaptiveLdTests[idx_max]
        print(f"self.ld is now: {self.ld}")
        return adaptiveLdDicts[idx_max], list_norms_to_add
    
    def agg_copy_weights(self, results, part_agg, last_weights):
        _, norms_dict = self.calculate_avg_norms(results,last_weights)
        ret_weights = []
        for elem in norms_dict:
            for i in range(len(norms_dict[elem])):
                if i < len(ret_weights):
                    ret_weights[i] = np.add(ret_weights[i], norms_dict[elem][i]*part_agg[elem])
                else:
                    ret_weights.append(norms_dict[elem][i]*part_agg[elem])
        for i in range(len(ret_weights)):
            ret_weights[i] = np.add(ret_weights[i], last_weights[i])
        return ret_weights
        
    
    def norms_from_parts(self, parts, results, last_weights):
        avg_norms, norms_dict = self.calculate_avg_norms(results, last_weights)
        no_weights = np.sum([np.prod(list(v.shape)) for v in last_weights])*self.fraction_boost_iid
        norms_list = []
        empty_weights = self.get_empty_weights()
        weights_to_remove_prep = copy.deepcopy(empty_weights)
        weights_to_remove = []
        for i in range(len(weights_to_remove_prep)):
            weights_to_remove.append(np.add(1, weights_to_remove_prep[i]))
        for elem in parts:
            if parts[elem] > 0:
                no_weights_elem = int(no_weights * parts[elem])
                print(no_weights)
                print(no_weights_elem)
                weights_to_add = copy.deepcopy(empty_weights)
                dif_norms = []
                for i in range(len(norms_dict[elem])):
                    dif_norms.append(np.subtract(norms_dict[elem][i],avg_norms[i]))
                # find indexes of n largest absolute values in dif norms..
                while no_weights_elem > 0:
                    # find index of largest dif and set it to 0
                    list_maxes = []
                    index_maxes = []
                    for i in range(len(dif_norms)):
                        index_max = np.unravel_index(abs(dif_norms[i]).argmax(), dif_norms[i].shape)
                        list_maxes.append(abs(dif_norms[i][index_max]))
                        index_maxes.append(index_max)
                    index_list_max = np.argmax(list_maxes)
                    dif_norms[index_list_max][index_maxes[index_list_max]] = 0
                    weights_to_remove[index_list_max][index_maxes[index_list_max]] = weights_to_remove[index_list_max][index_maxes[index_list_max]] - parts[elem]
                    #test
                    if weights_to_remove[index_list_max][index_maxes[index_list_max]] < 0:
                        print("wtf is happening???")
                    weights_to_add[index_list_max][index_maxes[index_list_max]] = parts[elem]*norms_dict[elem][index_list_max][index_maxes[index_list_max]]
                    no_weights_elem = no_weights_elem-1
                norms_list.append(weights_to_add)
        scaled_weights_to_remove = []
        for i in range(len(weights_to_remove)):
            scaled_weights_to_remove.append((-1)*np.multiply(np.subtract(1,weights_to_remove[i]), avg_norms[i]))
        norms_list.append(scaled_weights_to_remove)
        #test
        idx = np.unravel_index(abs(scaled_weights_to_remove[0]).argmax(), scaled_weights_to_remove[0].shape)
        print("norms we add on this:")
        for elem in norms_list:
            print(elem[0][idx])
        print("last weight this:")
        print(last_weights[0][idx])
        print("new weights this:")
        for elem in results:
            print(parameters_to_ndarrays(elem[1].parameters)[0][idx])
        print("norms this: ")
        for elem in norms_dict:
            print(norms_dict[elem][0][idx])
        print("norms times parts this: ")
        for elem in parts:
            print(norms_dict[elem][0][idx]*parts[elem])
        print("avg norms this: ")
        print(avg_norms[0][idx])
        print("scaler of avg norms this")
        print(weights_to_remove[0][idx])
        return norms_list
    
    def get_empty_weights(self):
        empty_model = create_model(self.data)
        w = empty_model.get_weights()
        empty_weights = []
        for elem in w:
            empty_weights.append(np.subtract(elem, elem))
        return empty_weights
    
    def get_norms(self, weights, last_weights):
        norms = []
        for i in range(len(weights)):
            norms.append(np.subtract(weights[i], last_weights[i]))
        return norms
    
    def calculate_avg_norms(self, results, last_weights):
        norms_dict = {}
        norms_list = []
        for elem in results:
            norm = self.get_norms(parameters_to_ndarrays(elem[1].parameters),last_weights)
            norms_dict[elem[0]] = norm
            norms_list.append(norm)
        norms_avg = copy.deepcopy(norms_list[0])
        for w_indx in range(len(norms_list[0])):
            for c_indx in range(1, len(norms_list)):
                norms_avg[w_indx] = np.add(norms_avg[w_indx] , norms_list[c_indx][w_indx])

        
        for i in range(len(norms_avg)):
            norms_avg[i] = norms_avg[i]/len(norms_list)
        return norms_avg, norms_dict
            

    def points_to_parts(self, points):
        part_agg = {}
        #make sure no client has negative points
        for elem in points:
            points[elem] = max(0,points[elem])
        sum_points = 0
        for elem in points:
            sum_points += points[elem]
        sum_points = max(000.1, sum_points)
        for elem in points:
            part_agg[elem] = (points[elem] / sum_points)
        return part_agg

    def get_points_iid(self, label_acc_dict, results, overall_mean, points, last_loss, last_label_loss, round_nr):
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
                points[results[k][0]] = points.get(results[k][0],0) + (factor*slope*all_for_score[k]+10)
        return points

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
                if self.newold == "old":
                    points[results[k][0]] = points.get(results[k][0],0) + (max(1,factor))*slope*all_for_score[k]+10
                else:
                    points[results[k][0]] = points.get(results[k][0],0) + slope*all_for_score[k]+10
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
            all_sum = 0
            all_acc_correct = 0
            all_loss_correct = 0
            for i in range(len(spec_label_all_count)):
                all_sum += spec_label_all_count[i]
                spec_label_accuracy.append(spec_label_correct_count[i]/spec_label_all_count[i])
                all_acc_correct += spec_label_correct_count[i]
                spec_label_loss.append(spec_label_loss_count[i]/spec_label_all_count[i])
                all_loss_correct += spec_label_loss_count[i]
            return all_loss_correct/all_sum, {"accuracy": all_acc_correct/all_sum}, spec_label_accuracy, spec_label_loss
        return evaluate
