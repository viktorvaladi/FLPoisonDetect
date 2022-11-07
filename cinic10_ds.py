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
import numpy as np
from keras.utils import np_utils
import pandas as pd

batch_size = 32

def get_train_ds(num_of_clients, client_index, data):
  if data=="cifar10":
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    length = len(x_train)
    partition = client_index
    part_size = length/num_of_clients
    x_train = x_train[int(partition*part_size) : int(((partition+1)*part_size)-1) ]
    y_train = y_train[ int(partition*part_size) : int(((partition+1)*part_size)-1) ]
    y_train = np_utils.to_categorical(y_train, 10)
    x_train = x_train.astype('float32')
    return x_train, y_train
  if data=="emnist":
    train = pd.read_csv('../emnist/emnist-balanced-train.csv', header=None)
    
    x_train = train.iloc[:, 1:]
    y_train = train.iloc[:, 0]

    x_train = x_train.values
    y_train = y_train.values
    
    del train

    def rotate(image):
        image = image.reshape([28, 28])
        image = np.fliplr(image)
        image = np.rot90(image)
        return image.reshape([28 * 28])
    x_train = np.apply_along_axis(rotate, 1, x_train)

    x_train = x_train.reshape(len(x_train), 28, 28)
    
    length = len(x_train)
    partition = client_index
    part_size = length/num_of_clients
    x_train = x_train[int(partition*part_size) : int(((partition+1)*part_size)-1) ]
    y_train = y_train[ int(partition*part_size) : int(((partition+1)*part_size)-1) ]
    y_train = np_utils.to_categorical(y_train, 47)
    x_train = x_train.astype('float32')
    return x_train, y_train

def get_test_val_ds(data):
  if data=="cifar10":
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32')
    y_test = np_utils.to_categorical(y_test, 10)
    return x_test,y_test
  if data=="emnist":
    test = pd.read_csv('../emnist/emnist-balanced-test.csv', header=None)

    x_test = test.iloc[:, 1:]
    y_test = test.iloc[:, 0]

    x_test = x_test.values
    y_test = y_test.values
    del test

    def rotate(image):
        image = image.reshape([28, 28])
        image = np.fliplr(image)
        image = np.rot90(image)
        return image.reshape([28 * 28])
    
    x_test = np.apply_along_axis(rotate, 1, x_test)

    x_test = x_test.reshape(len(x_test), 28, 28)
    
    x_test = x_test.astype('float32')
    y_test = np_utils.to_categorical(y_test, 47)
    return x_test, y_test
