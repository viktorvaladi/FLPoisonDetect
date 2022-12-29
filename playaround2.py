import numpy as np
import pandas as pd
# from keras.utils import np_utils
# from numpy import asarray

a = np.load('res.npz')
for elem in a:
    print(f"{elem} = {list(a[elem])}")