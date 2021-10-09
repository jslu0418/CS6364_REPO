# Coding solution for CS6364.002's homework4 Q2, Q4.
# Filename:      gradient_descent_library.py
# Name:          Jiashuai Lu
# Student ID:    jxl173630 (2021376746)
# Course name:   Artificial Intelligence
# Course number: CS 6364.002
#
# Functions for:
# 1. The gradient descent algorithm
# 2. The stochastic gradient descent (SGD) algorithm
# 3. The SGD algorithm with momentum
# 4. The SGD algorithm with Nesterov momentum
# 5. The AdaGrad algorithm

import numpy as np
import random

def init_weights(m):
    '''
    Generate random weights in [0, 1]
    '''
    return np.random.rand(m)

def linear_predict(w, X):
    '''
    w: (m + 1) * 1
    X: n * m
    compute X multiply by w[1:] + w[0]
    '''
    return np.matmul(X, w[1:]) + w[0]

def mean_squared_error(prediction, target):
    diff = np.subtract(prediction, target)
    squared_errors = np.square(diff)

    return squared_errors.mean()

def linear_loss(w, X, y):
    '''
    Computer the mean squared error and the gradient
    '''
    # Compute the prediction
    y_p = linear_predict(w, X)

    mse = mean_squared_error(y_p, y)
    diff = np.subtract(y_p, y)
    diff_dot_X = (1/X.shape[0]) * np.matmul(X.transpose(), diff)

    return (mse, np.concatenate((np.array([diff.mean()]), diff_dot_X),
                                axis = 0))

def gradient_descent_fixed_learning_rate(
        loss_func, X, y, alpha = 0.01, max_iter = 10000):
    '''
    Implement the gradient descent algorithm with fixed learning rate.
    X: dataset n * m
    y: target n
    loss_func(w, X, y) -> f as loss function value, g as gradient.
    alpha, learning rate, default to be 0.01.

    return paramaters w
    '''
    # initialized weights (m + 1), m plus one more parameter
    (n, m) = X.shape
    w = init_weights(m + 1)
    pre_f = np.finfo(float).max
    for cur_iter in range(max_iter):
        f, g = loss_func(w, X, y)
        if abs(f - pre_f) < 1e-5:
            break
        w = w - alpha * g
        pre_f = f
    return w

def stochastic_gradient_descent_fixed_learning_rate(
        loss_func, X, y, alpha = 0.01, max_iter = 10000, cond = 1e-5):
    '''
    Implement the stochastic gradient descent algorithm with fixed
    learning rate.
    X: dataset n * m
    y: target n
    loss_func(w, X, y) -> f as loss function value, g as gradient.
    alpha, learning rate, default to be 0.01.
    cond: converge condition for early stop
    return paramaters w
    '''
    # initialized weights (m + 1), m plus one more parameter
    (n, m) = X.shape
    w = init_weights(m + 1)
    pre_f = np.finfo(float).max
    for cur_iter in range(max_iter):
        sample_idx = random.randint(0, X.shape[0]-1)
        f, g = loss_func(w, X[sample_idx:sample_idx + 1],
                         y[sample_idx:sample_idx + 1])
        if abs(f - pre_f) < cond:
            break
        w = w - alpha * g
        pre_f = f
    return w

def stochastic_gradient_descent_fixed_learning_rate_momentum(
        loss_func, X, y, alpha = 0.01, max_iter = 10000,
        cond = 1e-5, eta = 0.9):
    '''
    Implement the stochastic gradient descent algorithm with fixed
    learning rate.
    X: dataset n * m
    y: target n
    loss_func(w, X, y) -> f as loss function value, g as gradient.
    alpha, learning rate, default to be 0.01.
    cond: converge condition for early stop
    return paramaters w
    '''
    # initialized weights (m + 1), m plus one more parameter
    (n, m) = X.shape
    w = init_weights(m + 1)
    sample_idx = random.randint(0, X.shape[0]-1)
    pre_f = np.finfo(float).max
    v = 0
    for cur_iter in range(max_iter):
        sample_idx = random.randint(0, X.shape[0]-1)
        f, g = loss_func(w, X[sample_idx:sample_idx + 1],
                         y[sample_idx:sample_idx + 1])
        if abs(f - pre_f) < cond:
            break
        pre_f = f
        # update momentum
        v = eta * v - alpha * g

        # update weight with momentum
        w = w + v

    return w
