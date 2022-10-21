from operator import index
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import argparse
import flwr as fl
import random
from model import create_model
from model_ascent import create_model_ascent
from cinic10_ds import get_test_val_ds
import numpy as np


class FLClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_test, y_test, is_poisoned):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.is_poisoned = is_poisoned

        self.train_count = len(x_train)
        self.test_count = len(x_test)
        self.model = create_model()
        self.model_ascent = create_model_ascent()
        self.lazy_poisoning = False
        self.pga_poisoning = False
        self.pga_poisoning_split = False
        self.epochs = 10
        if is_poisoned:
            if True:
                #self.y_train = self.poisonRandomLabel(y_train = self.y_train, no_labels=3000)
                #self.x_train = self.poisonRandomPixels(x_train = self.x_train)
                #self.y_train = self.poisonSpecificLabel(y_train=self.y_train, part_of_labels=1.0,label=2,to_label='random')
                #heterogen split below
                #self.x_train, self.y_train = self.removeLabels(self.x_train, self.y_train, 7, 7)
                #self.lazy_poisoning = True
                self.pga_poisoning_split = True
                self.pga_poisoning = True
                self.pga_scaler = 0.05
                #self.epochs = 30
                pass
        else:
            #heterogen split below
            #self.x_train, self.y_train = self.removeLabels(self.x_train, self.y_train, 4, 5)
            pass
    
    def removeLabels(self,x_train,y_train,label1,label2):
        print(f"removing labels {label1} , {label2} !!!!!!")
        print(len(x_train))
        x_train = list(x_train)
        y_train = list(y_train)
        x_train_new = []
        y_train_new = []
        for i in range(len(y_train)-1):
            if np.argmax(y_train[i]) == label1 or np.argmax(y_train[i]) == label2:
                pass
            else:
                x_train_new.append(x_train[i])
                y_train_new.append(y_train[i])
        x_train = np.array(x_train_new)
        y_train = np.array(y_train_new)
        print(len(x_train))
        return x_train, y_train
    
    def poisonRandomLabel(self,y_train,no_labels=800):
        print("Poisoning labels!!!!!!!")
        print(len(y_train))
        y_train = list(y_train)
        for i in range(no_labels):
            x = [0.0 for j in range(10)]
            x[random.randint(0,9)] = 1.0
            y_train[random.randint(0,int(len(y_train)/2))-1] = np.array(x)
            y_train[random.randint(int(len(y_train)/2),len(y_train)-1)] = np.array(x)
        y_train = np.array(y_train)
        return y_train
    
    def poison_specific_label(self,y_train,part_of_labels=1.0, label=4, to_label=5):
        print("Poisoning labels!!!!!!!")
        print(len(y_train))
        y_train = list(y_train)
        for i in range(len(y_train)):
            if np.argmax(y_train[i]) == label:
                if random.uniform(0,1) <= part_of_labels:
                    x = [0.0 for j in range(10)]
                    if to_label == 'random':
                        to_label = random.randint(0,9)
                    x[to_label] = 1.0
                    y_train[i] = x
        y_train = np.array(y_train)
        return y_train
    
    def poisonRandomPixels(self, x_train, perc_img=1.0, nr_pixels = 600, th = 0.5):
        print("Poisoning pixels!!!!!!!")
        x_train = list(x_train)
        nr_pictures = int(perc_img*len(x_train))
        index_value = random.sample(list(enumerate(x_train)), nr_pictures)
        for idx, _ in index_value:
            for i in range(nr_pixels):
                position = random.randint(0,len(x_train[idx])-1)
                row_position = random.randint(0,len(x_train[idx][position])-1)
                x_train[idx][position][row_position]
                for j in range(len(x_train[idx][position][row_position])):
                    current = x_train[idx][position][row_position][j]
                    new = np.float32(round(random.uniform(current*(1-th),current*(1+th))))
                    x_train[idx][position][row_position][j] = new
        x_train = np.array(x_train)
        return x_train

    def get_parameters(self, config):
        return self.model.get_weights()
    
    def get_properties(self):
        return {"is_poisoned" : self.is_poisoned}
    
    def find_worst_labels(self, parameters):
        spec_label_acc = self.ev_labels(parameters)
        l1 = np.argmin(spec_label_acc)
        spec_label_acc[l1] = 100
        l2 = np.argmin(spec_label_acc)
        spec_label_acc[l2] = 100
        l3 = np.argmin(spec_label_acc)
        spec_label_acc[l3] = 100
        return l1, l2, l3
    
    def split_data(self, l1, l2, l3, x, y):
        
        pass
    
    def pga_poison(self,parameters, x, y):
        self.model_ascent.set_weights(parameters)
        last_weights = self.model_ascent.get_weights()
        self.model_ascent.fit(x, y, epochs=self.epochs, batch_size=128, validation_split=0.1)
        new_weights = self.model_ascent.get_weights()
        for i in range(len(last_weights)):
            scaled_norm = np.subtract(new_weights[i],last_weights[i])*self.pga_scaler
            last_weights[i] = np.add(last_weights[i],scaled_norm)
        return last_weights, self.train_count, {"is_poisoned" : self.is_poisoned}
    
    def pga_poison_split(self,parameters, x, y):
        l1, l2, l3 = self.find_worst_labels(parameters)
        x_improve, y_improve, x_worsen, y_worsen = self.split_data(l1, l2, l3, x, y)
        self.model_ascent.set_weights(parameters)
        last_weights = self.model_ascent.get_weights()
        self.model_ascent.fit(x, y, epochs=self.epochs, batch_size=128, validation_split=0.1)
        new_weights = self.model_ascent.get_weights()
        for i in range(len(last_weights)):
            scaled_norm = np.subtract(new_weights[i],last_weights[i])*self.pga_scaler
            last_weights[i] = np.add(last_weights[i],scaled_norm)
        return last_weights, self.train_count, {"is_poisoned" : self.is_poisoned}

    def fit(self, parameters, config):
        partition = config.get('current_round') % config.get('nr_of_split_per_round')
        part_size = len(self.x_train)/config.get('nr_of_split_per_round')
        x = self.x_train[int(partition*part_size) : int(((partition+1)*part_size)-1) ]
        y = self.y_train[ int(partition*part_size) : int(((partition+1)*part_size)-1) ]

        if self.pga_poisoning:
            return self.pga_poison(self,parameters, x, y)
        if self.pga_poisoning_split:
            return self.pga_poison_split(self,parameters, x, y)

        self.model.set_weights(parameters)
        if self.lazy_poisoning:
            return self.model.get_weights(), self.train_count, {"is_poisoned" : self.is_poisoned}

        self.model.fit(x, y, epochs=self.epochs, batch_size=128, validation_split=0.1)
        return self.model.get_weights(), self.train_count, {"is_poisoned" : self.is_poisoned}

    def ev_labels(self, parameters):
        x_test, y_test = get_test_val_ds()
        self.model.set_weights(parameters)
        preds = self.model.predict(x_test)
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
        return spec_label_accuracy

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, self.test_count, {"accuracy": accuracy}

    def start(self, server_address):
            fl.client.start_numpy_client(server_address = "127.0.0.1" + ":8080", client=self)
