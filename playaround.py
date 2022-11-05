from typing import Dict, Optional, Tuple, List

import flwr as fl
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras import datasets, layers, models
from keras.utils import np_utils
from cinic10_ds import get_train_ds, get_test_val_ds, batch_size
import tensorflow_datasets as tfds
import random
import os
from model import create_model
from model_ascent import create_model_ascent
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
np.set_printoptions(threshold=np.inf)


model = create_model("emnist")

x_train, y_train = get_train_ds(100,1, "emnist")
x_test, y_test = get_test_val_ds("emnist")

model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

#(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
#ds = tfds.load('emnist', split='train', shuffle_files=True)
#train = pd.read_csv('../archive/emnist-balanced-train.csv', header=None)
#test = pd.read_csv('../archive/emnist-balanced-test.csv', header=None)

#x_train = train.iloc[:, 1:]
#y_train = train.iloc[:, 0]
#x_test = test.iloc[:, 1:]
#y_test = test.iloc[:, 0]

#x_train = x_train.values
#y_train = y_train.values
#x_test = x_test.values
#y_test = y_test.values
#del train, test

#def rotate(image):
#    image = image.reshape([28, 28])
#    image = np.fliplr(image)
#    image = np.rot90(image)
#    return image.reshape([28 * 28])
#x_train = np.apply_along_axis(rotate, 1, x_train)
#x_test = np.apply_along_axis(rotate, 1, x_test)

#x_train = x_train.reshape(len(x_train), 28, 28)
#x_test = x_test.reshape(len(x_test), 28, 28)
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#y_test = np_utils.to_categorical(y_test, 47)
#y_train = np_utils.to_categorical(y_train, 47)

# print(tf.version.VERSION)

# x_test, y_test = get_test_val_ds("emnist")
# x_test = x_test[int(len(x_test)/2):int(len(x_test)-1)]
# y_test = y_test[int(len(y_test)/2):int(len(y_test)-1)]
# print(x_test[0])
# print(y_test[0])
#model = create_model("emnist")
#model.summary()
#model.fit(x_train, y_train, epochs=1, batch_size=128, validation_split=0.1)

#model.evaluate(x_test,y_test)

#x_train, y_train = get_train_ds(10, 1)
#x_test, y_test = get_test_val_ds()

#model = create_model_ascent()
#model.fit(x_test, y_test, epochs=1, batch_size=128, validation_split=0.1)
#model.evaluate(x_test,y_test)

#asd = model.get_weights()
#asd2 = model.get_weights()
#print(asd[1])
#r = []
#for i in range(len(asd)):
#    r.append(np.subtract(asd[i],asd2[i]))
#print(r[1])

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
