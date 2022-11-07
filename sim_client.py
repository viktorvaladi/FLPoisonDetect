import os

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flwr.common.logger import log
from logging import WARNING

import flwr as fl
import numpy as np
import random

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, model_ascent, x_train, y_train, x_val, y_val, is_poisoned, is_noniid) -> None:
        super().__init__()
        self.model = model
        self.model_ascent = model_ascent
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.is_poisoned = is_poisoned
        self.is_noniid = is_noniid
        self.lazy_poisoning = False
        self.pga_poisoning = False
        self.pga_poisoning_split = False
        self.epochs = 10
        self.train_count = len(x_train)
        self.test_count = len(x_val)
        if is_noniid:
            #self.x_train, self.y_train = self.removeLabels(self.x_train, self.y_train, 7, 7)
            pass
        else:
            #self.x_train, self.y_train = self.removeLabels(self.x_train, self.y_train, 4, 5)
            pass

        if is_poisoned:
            #self.y_train = self.poisonRandomLabel(y_train = self.y_train, no_labels=3000)
            #self.x_train = self.poisonRandomPixels(x_train = self.x_train)
            #self.y_train = self.poisonSpecificLabel(y_train=self.y_train, part_of_labels=1.0,label=2,to_label='random')
            #self.lazy_poisoning = True
            #self.pga_poisoning_split = True
            #self.pga_poisoning = True
            #self.pga_scaler = 0.1
            #self.epochs = 30
            pass
        else:
            #heterogen split below
            #self.x_train, self.y_train = self.removeLabels(self.x_train, self.y_train, 4, 5)
            pass

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        x = self.x_train
        y = self.y_train

        if self.pga_poisoning:
            return self.pga_poison(parameters, x, y)
        if self.pga_poisoning_split:
            return self.pga_poison_split(parameters, x, y)

        self.model.set_weights(parameters)
        if self.lazy_poisoning:
            return self.model.get_weights(), self.train_count, {"is_poisoned" : self.is_poisoned}

        self.model.fit(x, y, epochs=self.epochs, batch_size=128, validation_split=0.1)
        return self.model.get_weights(), self.train_count, {"is_poisoned" : self.is_poisoned}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)
        return loss, len(self.x_val), {"accuracy": acc}
    
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
    
    def find_worst_labels(self, parameters):
        spec_label_acc = self.ev_labels(parameters)
        l1 = np.argmin(spec_label_acc)
        spec_label_acc[l1] = 100
        l2 = np.argmin(spec_label_acc)
        spec_label_acc[l2] = 100
        l3 = np.argmin(spec_label_acc)
        spec_label_acc[l3] = 100
        return [l1, l2, l3]
    
    def split_data(self, l_list, x, y):
        x_improve = []
        y_improve = []
        x_worsen = []
        y_worsen = []
        for i in range(len(y)):
            if np.argmax(y[i]) in l_list:
                x_improve.append(x[i])
                y_improve.append(y[i])
            else:
                x_worsen.append(x[i])
                y_worsen.append(y[i])
        return np.array(x_improve), np.array(y_improve), np.array(x_worsen), np.array(y_worsen)
    
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
        l_list = self.find_worst_labels(parameters)
        x_improve, y_improve, x_worsen, y_worsen = self.split_data(l_list, x, y)
        self.model_ascent.set_weights(parameters)
        self.model.set_weights(parameters)
        last_weights = self.model_ascent.get_weights()
        self.model_ascent.fit(x_worsen, y_worsen, epochs=self.epochs, batch_size=128, validation_split=0.1)
        self.model.fit(x_improve, y_improve, epochs=self.epochs, batch_size=128, validation_split=0.1)
        new_weights = self.model_ascent.get_weights()
        new_weights_good = self.model.get_weights()
        for i in range(len(last_weights)):
            scaled_norm = np.subtract(new_weights[i],last_weights[i])*self.pga_scaler
            good_norm = np.subtract(new_weights_good[i], last_weights[i])
            last_weights[i] = np.add(last_weights[i],scaled_norm)
            last_weights[i] = np.add(last_weights[i],good_norm)
        return last_weights, self.train_count, {"is_poisoned" : self.is_poisoned}
    
    def ev_labels(self, parameters):
        x_test, y_test = self.x_val, self.y_val
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