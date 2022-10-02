from typing import Dict, Optional, Tuple, List

import flwr as fl
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras import datasets, layers, models
from keras.utils import np_utils
from cinic10_ds import get_train_ds, get_test_val_ds, batch_size
import random
import os
from model import create_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
np.set_printoptions(threshold=np.inf)

(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

asd = [[0.25]]
for elem in asd:
    print(elem[-1])

print(":)")


#x = np.array([0.7336,0.7476,0.7280,0.7466,0.7243,0.7339,0.7435,0.7560, 0.7336, 0.7479])
#print(np.mean(x))
#print(np.std(x))
"""
#<class 'tensorflow.python.data.ops.dataset_ops.ShardDataset'>
#<class 'tuple'>
#<class 'tensorflow.python.framework.ops.EagerTensor'>

x = list(x_train.as_numpy_iterator())
features = []
labels = []
for i in range(len(x)):
    for j in range(len(x[i][0])):
        features.append(x[i][0][j])
        labels.append(x[i][1][j])
features = np.array(features)
labels = np.array(labels)
#x = tf.data.Dataset.from_tensor_slices(features,labels)
#x = x.batch(32)

random.seed(0)
for i in range(50):
    x = [0.0 for j in range(10)]
    x[random.randint(0,9)] = 1.0
    labels[random.randint(0,len(labels)-1)] = x

model = Sequential()

model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
"""
