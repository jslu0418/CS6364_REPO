# Coding solution for CS6364.002's homework4 Q2.
# Filename:      HW4_JiashuaiLu_code_Q2.py
# Name:          Jiashuai Lu
# Student ID:    jxl173630 (2021376746)
# Course name:   Artificial Intelligence
# Course number: CS 6364.002
#
# Implement the following five algorithms to train a linear regression
# model for the Boston housing data set
# https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155
# Split the dataset to a training set (70% samples) and a testing set
# (30% samples). Report the root mean squared errors (RMSE) on the
# training and testing sets.
# 1. The gradient descent algorithm
# 2. The stochastic gradient descent (SGD) algorithm
# 3. The SGD algorithm with momentum
# 4. The SGD algorithm with Nesterov momentum
# 5. The AdaGrad algorithm

from utility import *
import load_data

@format_decorator
def HW4_JiashuaiLu_code_Q2():
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


    # Using our gradient descent algorithm
    from gradient_descent_library import linear_loss, linear_predict, gradient_descent_fixed_learning_rate
    print("*** Beginning of Gradient descent algorithm ***")
    w = gradient_descent_fixed_learning_rate(linear_loss,
                                             data,
                                             target,
                                             1e-6,
                                             max_iter = 50000)
    print("Trained parameters w: ")
    print(w)

    training_prediction = linear_predict(w, data[:training_n])
    testing_prediction = linear_predict(w, data[training_n:])
    report_root_mean_squared_error('Training set',
                                   training_prediction,
                                   target[:training_n])
    report_root_mean_squared_error('Testing set',
                                   testing_prediction,
                                   target[training_n:])
    print("*** Ending Gradient descent algorithm ***")

    # Using our stochastic gradient descent (SGD) algorithm
    from gradient_descent_library import stochastic_gradient_descent_fixed_learning_rate
    print("*** Beginning of Stochastic Gradient descent algorithm ***")
    w = stochastic_gradient_descent_fixed_learning_rate(
        linear_loss, data, target, 1e-6)
    print("Trained parameters w: ")
    print(w)

    training_prediction = linear_predict(w, data[:training_n])
    testing_prediction = linear_predict(w, data[training_n:])
    report_root_mean_squared_error('Training set',
                                   training_prediction,
                                   target[:training_n])
    report_root_mean_squared_error('Testing set',
                                   testing_prediction,
                                   target[training_n:])
    print("*** Ending Stochastic Gradient descent algorithm ***")

    # Using our stochastic gradient descent (SGD) with momentum
    # algorithm
    from gradient_descent_library import stochastic_gradient_descent_fixed_learning_rate_momentum
    print("*** Beginning of Stochastic Gradient descent algorithm ***")
    w = stochastic_gradient_descent_fixed_learning_rate_momentum(
        linear_loss, data, target, 1e-6, max_iter = 50000)
    print("Trained parameters w: ")
    print(w)

    training_prediction = linear_predict(w, data[:training_n])
    testing_prediction = linear_predict(w, data[training_n:])
    report_root_mean_squared_error('Training set',
                                   training_prediction,
                                   target[:training_n])
    report_root_mean_squared_error('Testing set',
                                   testing_prediction,
                                   target[training_n:])
    print("*** Ending Stochastic Gradient descent algorithm ***")

if __name__ == '__main__':
    HW4_JiashuaiLu_code_Q2()
