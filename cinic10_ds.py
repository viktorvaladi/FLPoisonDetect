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


## cifar-10 dirichlet idx map
dirichlet = True
n_parties = 1000
beta = 0.3
(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
min_size = 0
min_require_size = 1
K = 10
N = y_train.shape[0]
np.random.seed(2022)
net_dataidx_map = {}

while min_size < min_require_size:
    idx_batch = [[] for _ in range(n_parties)]
    for k in range(K):
        idx_k = np.where(y_train == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(beta, n_parties))
        # logger.info("proportions1: ", proportions)
        # logger.info("sum pro1:", np.sum(proportions))
        ## Balance
        proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
        # logger.info("proportions2: ", proportions)
        proportions = proportions / proportions.sum()
        # logger.info("proportions3: ", proportions)
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        # logger.info("proportions4: ", proportions)
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])
        # if K == 2 and n_parties <= 10:
        #     if np.min(proportions) < 200:
        #         min_size = 0
        #         break


for j in range(n_parties):
    np.random.shuffle(idx_batch[j])
    net_dataidx_map[j] = idx_batch[j]


##cifar10 

def get_train_ds(num_of_clients, client_index, data):
  if data=="cifar10":
    (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    length = len(x_train)
    partition = client_index
    part_size = length/num_of_clients
    if dirichlet:
      x_ret = []
      y_ret = []
      for idx in net_dataidx_map[client_index]:
        x_ret.append(x_train[idx])
        y_ret.append(y_train[idx])
      x_ret = np.array(x_ret)
      y_ret = np.array(y_ret)
    else:
      x_ret = x_train[int(partition*part_size) : int(((partition+1)*part_size)-1) ]
      y_ret = y_train[ int(partition*part_size) : int(((partition+1)*part_size)-1) ]
    y_ret = np_utils.to_categorical(y_ret, 10)
    x_ret = x_ret.astype('float32')
    return x_ret, y_ret
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
  if data=="femnist":
    train = pd.read_pickle(r'../femnist/'+str(client_index)+'/train.pickle')
    x = train["x"]
    y = train["y"]
    y_train = np_utils.to_categorical(y,62)
    x_train = []
    for elem in x:
      x_train.append(np.asarray(elem))
    val = pd.read_pickle(r'../femnist/'+str(client_index)+'/val.pickle')
    xv = val["x"]
    yv = val["y"]
    y_val = np_utils.to_categorical(yv,62)
    x_val = []
    for elem in xv:
      x_val.append(np.asarray(elem))
    return np.concatenate((np.array(x_train), np.array(x_val)), axis=0), np.concatenate((y_train, y_val), axis=0)

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
  if data=="femnist":
    x_test = np.array([])
    y_test = np.array([])
    for i in range(3597):
      test = pd.read_pickle(r'../femnist/'+str(i)+'/train.pickle')
      x = test["x"]
      y = test["y"]
      y_add = np_utils.to_categorical(y,62)
      x_add = []
      for elem in x:
        x_add.append(np.asarray(elem))
      if i == 0:
        x_test = x_add
        y_test = y_add
      else:
        x_test = np.concatenate((x_test,np.array(x_add)), axis=0)
        y_test = np.concatenate((y_test,y_add),axis=0)
    return x_test, y_test
    
