from cProfile import label
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
import pathlib
import os
import numpy as np
from keras.utils import np_utils

batch_size = 32

def get_train_ds(num_of_clients, client_index):
  (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
  length = len(x_train)
  partition = client_index
  part_size = length/num_of_clients
  x_train = x_train[int(partition*part_size) : int(((partition+1)*part_size)-1) ]
  y_train = y_train[ int(partition*part_size) : int(((partition+1)*part_size)-1) ]
  y_train = np_utils.to_categorical(y_train, 10)
  x_train = x_train.astype('float32')
  return x_train, y_train

def get_test_val_ds():
  (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_test = x_test.astype('float32')
  y_test = np_utils.to_categorical(y_test, 10)
  return x_test,y_test
