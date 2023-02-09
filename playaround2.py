import numpy as np
import pandas as pd
from math import comb
import matplotlib.pyplot as plt
import matplotlib
# from keras.utils import np_utils
# from numpy import asarray
plt.rcParams["font.family"] = "Times New Roman"
font = {'size'   : 16}
import matplotlib.ticker as ticker
matplotlib.rc('font', **font)

# a = np.load('res.npz')
# i = 0
# for elem in a:
#     if i == 0:
#         print(f"{elem} = {list(a[elem][-5:])}")
#     i = i+1


def binomial_probability(n, k, p):
  return comb(n, k) * (p**k) * ((1 - p)**(n - k))

# probability of getting 9 or more red elements
prob = sum(binomial_probability(30, x, 0.1) for x in range(12, 31))
print(prob)

x = [1-((1-prob)**i) for i in range(200000)]
print(x[0:5])
title = "Probability of exceeding the threshold for \n malicious clients"
plt.plot(x)
plt.xlabel("Communication rounds")
plt.ylabel("Probability")
plt.xticks([0,50000,100000,150000,200000], 
           ["0", "50K", "100K", "150K", "200K"])
plt.legend(loc='upper left')
# plt.title(title)
plt.show()
