import numpy as np
import os
import os.path
import pickle
import matplotlib.pyplot as plt

import setup as s 

(training_X, training_Y, training_N, val_X, val_Y, val_N, test_X, test_Y, test_N) = s.run()

def train_increment(gradient, w, eta):
    return gradient(training_X, training_Y, w, eta, training_N)

def train(gradient, T, eta):
    w = np.zeros((785,1))
    for t in range(0, T):
        w = train_increment(gradient, w, eta) 

    return w

def train_err(loss, w):
    return float(loss(training_X, training_Y, w, training_N))

def val_err(loss, w):
    return float(loss(val_X, val_Y, w, val_N))

def test_err(loss, w):
    return float(loss(test_X, test_Y, w, test_N))

def run_experiment(loss, gradient, eta_values):
    T = 1000
    best_eta = 0 
    lowest_err = float("inf")

    for eta in eta_values:
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

    best_T = 0 
    lowest_err = float("inf")

    max_T = 1000

    T_values = range(1, max_T + 1)
    err_values = [] 

    w = np.zeros((785,1))

    for T in T_values:
        w = train_increment(gradient, w, best_eta)
        
        err = val_err(loss, w)
        
        err_values.append(err)

        if err < lowest_err:
            best_T = T
            lowest_err = err


    w = train(gradient, best_T, best_eta)
    
    print "best T: " + str(best_T)
    print "best eta: " + str(best_eta)
    print "test error: " + str(test_err(loss, w))

    plt.plot(T_values, err_values, '-')

    plt.title('Validation Error vs. T Values')
    plt.xlabel('T Value')
    plt.ylabel('Validation Error')

    plt.show()

