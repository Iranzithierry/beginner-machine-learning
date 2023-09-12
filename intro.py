# import tensorflow as tf
# t = tf.zeros([5,5,5,5])
# t = tf.reshape(t, [125, -1])
# print(t)
from __future__ import absolute_import, division ,print_function, unicode_literals

import numpy as np #data arrays
import pandas as pd #analytics
import matplotlib.pyplot as plt  #graphs
# from ipython.display import clear_output
# import tensorflow.compact.v2.feature_column as fc

dftrain = pd.read_csv("train.csv")
dfeval = pd.read_csv("eval.csv")
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
# to display the first 5 columns
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# -------------------------------------------------------
# >
# print(dfeval.head())



# this line shows only the first column of our table 
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# -------------------------------------------------------
# >
# print(dftrain.loc[0], y_train.loc[0]) #To check per one row for example here used row 0 means the 1st row of our table




"""to check how many people died based on the number of age"""
#> this create vertical graphy with value based before hist to
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# -------------------------------------------------------
# >
# plt.figure()
# dftrain.male.hist(bins=20)
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.title('Histogram of Age')
# plt.show()




""" to check how many females and males"""
# >This is used to create horizontal graph with different values for example if in the
# >sex column have 2 values male and female they plot graph of how many female and how many males
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# -------------------------------------------------------
# >
# plt.figure()
# dftrain.sex.value_counts().plot(kind='barh')
# plt.show()


"""to check how many people died based on the classes"""
# -------------------------------------------------------
# >
plt.figure()
dftrain['class'].value_counts().plot(kind='barh')
plt.show()
