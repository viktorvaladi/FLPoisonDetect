import numpy as np
import pandas as pd
from math import comb
import matplotlib.pyplot as plt
# from keras.utils import np_utils
# from numpy import asarray

# a = np.load('res.npz')
# for elem in a:
#     print(f"{elem} = {list(a[elem])}")


# def binomial_probability(n, k, p):
#   return comb(n, k) * (p**k) * ((1 - p)**(n - k))

# # probability of getting 9 or more red elements
# prob = sum(binomial_probability(30, x, 0.1) for x in range(12, 31))
# print(prob)

# x = [1-((1-prob)**i) for i in range(200000)]
# title = "Probability to get 40% malicious elements \n in at least one pull with 10% malicious clients"
# plt.plot(x)
# plt.xlabel("Communication rounds")
# plt.ylabel("Probability")

# plt.legend(loc='upper left')
# plt.title(title)
# plt.show()
