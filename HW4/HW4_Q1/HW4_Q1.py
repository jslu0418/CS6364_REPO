# Coding solution for CS6364.002's homework4 Q1.
# Filename:      HW4_JiashuaiLu_code_Q1.py
# Name:          Jiashuai Lu
# Student ID:    jxl173630 (2021376746)
# Course name:   Artificial Intelligence
# Course number: CS 6364.002
#
# (Linear Regression): Use the python library (sklearn.linear model) to train a
# linear regression model for the Boston housing dataset:
# https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155.
# Split the dataset to a training set (70% samples) and a testing set
# (30% samples). Report the root mean squared errors (RMSE) on the training and
# testing sets.

from utility import *
import load_data

@format_decorator
def HW4_JiashuaiLu_code_Q1():
    from sklearn import linear_model
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    # %matplotlib inline # Commented since we are not using it in Jupyter.
    import seaborn as sns
    import math

    # Prepare data
    (data, target) = load_data.load_boston()
    n = target.size
    training_percent = 0.7
    training_n = int(n * training_percent)
    testing_n = n - training_n

    # Linear regression
    reg = linear_model.LinearRegression()
    reg.fit(data[:training_n], target[:training_n])
    training_prediction = reg.predict(data[:training_n])
    testing_prediction = reg.predict(data[training_n:])

    from sklearn.metrics import mean_squared_error

    # Compute root mean squared error
    training_rmse = math.sqrt(mean_squared_error(training_prediction,
                                            target[:training_n]))
    testing_rmse = math.sqrt(mean_squared_error(testing_prediction,
                                           target[training_n:]))

    print("Root mean squared errors on Training set: {}".format(training_rmse))
    print("Root mean squared errors on Testing set: {}".format(testing_rmse))

if __name__ == '__main__':
    HW4_JiashuaiLu_code_Q1()
