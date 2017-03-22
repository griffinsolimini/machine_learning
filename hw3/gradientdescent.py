import numpy as np
import os
import os.path
import pickle
import matplotlib.pyplot as plt

import setup as s 

(training_X, training_Y, training_N, val_X, val_Y, val_N, test_X, test_Y, test_N) = s.run()

def loss(X, Y, w, N):
    return np.sum(np.square(X * w - Y)) / N

def gradient(X, Y, w, eta, N):
    return w - eta * (2.0 / N) * ((X * w - Y).T * X).T

def train(gradient, T, eta):
    w = np.zeros((785,1))
    for t in range(0, T):
        w = gradient(training_X, training_Y, w, eta, training_N)

    return w

def train_err(loss, w):
    return loss(training_X, training_Y, w, training_N)

def val_err(loss, w):
    return loss(val_X, val_Y, w, val_N)

def test_err(loss, w):
    return loss(test_X, test_Y, w, test_N)

def run_experiment(loss, gradient):
    T = 1000
    eta_list = [3.5e-7, 3e-7, 1e-7, 7e-7, 1e-8, 5e-8, 1e-9, 5e-9, 1e-10, 5e-10]
    best_eta = 0 
    lowest_err = float("inf")

    for eta in eta_list:
        w = train(gradient, T, eta)
        
        tr_err = train_err(loss, w)
        v_err = val_err(loss, w)

        print "eta: " + str(eta)
        print "training error: " + str(tr_err)
        print "validation error: " + str(v_err)
        print
        
        if v_err < lowest_err:
            best_eta = eta
            lowest_err = v_err

    best_T = 1
    lowest_err = float("inf")

    max_T = 100

    T_values = range(1, max_T + 1)
    err_values = [] 
    for T in T_values:
        w = train(gradient, T, best_eta)
        err = val_err(loss, w)
        
        err_values.append(err)

        if err < lowest_err:
            best_T = T
            lowest_err = err

        print str(100.0 * T / max_T) + '%'

    plt.plot(T_values, err_values, '-')

    plt.title('Validation Error vs. T Values')
    plt.xlabel('T Value')
    plt.ylabel('Validation Error')

    plt.show()

    w = train(gradient, best_T, best_eta)
    print test_err(loss, w)

