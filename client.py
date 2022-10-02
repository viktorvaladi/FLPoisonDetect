from operator import index
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import argparse
import flwr as fl
import random
from model import create_model
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
        self.lazy_poisoning = False
        if is_poisoned:
            if True:
                #self.y_train = self.poisonRandomLabel(y_train = self.y_train)
                #self.x_train = self.poisonRandomPixels(x_train = self.x_train)
                #self.y_train = self.poisonSpecificLabel(y_train=self.y_train, part_of_labels=1.0,label=2,to_label='random')
                #heterogen split below
                self.x_train, self.y_train = self.removeLabels(self.x_train, self.y_train, 7, 7)
                #self.lazy_poisoning = True
                pass
        else:
            pass
            #heterogen split below
            self.x_train, self.y_train = self.removeLabels(self.x_train, self.y_train, 2, 3)
    
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

    def fit(self, parameters, config):
        partition = config.get('current_round') % config.get('nr_of_split_per_round')
        part_size = len(self.x_train)/config.get('nr_of_split_per_round')
        x = self.x_train[int(partition*part_size) : int(((partition+1)*part_size)-1) ]
        y = self.y_train[ int(partition*part_size) : int(((partition+1)*part_size)-1) ]
        self.model.set_weights(parameters)
        if self.lazy_poisoning:
            return self.model.get_weights(), self.train_count, {"is_poisoned" : self.is_poisoned}

        self.model.fit(x, y, epochs=10, batch_size=128, validation_split=0.1)

        return self.model.get_weights(), self.train_count, {"is_poisoned" : self.is_poisoned}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, self.test_count, {"accuracy": accuracy}

    def start(self, server_address):
            fl.client.start_numpy_client(server_address = "127.0.0.1" + ":8080", client=self)
