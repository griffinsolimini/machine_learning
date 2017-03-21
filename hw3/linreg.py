import numpy as np
import os
import os.path
import pickle

import setup as s 

(training_X, training_Y, training_N, val_X, val_Y, val_N, test_X, test_Y, test_N) = s.run()

#  training_X = data[0]

def train(T, gamma):
    w = np.zeros((785,1))
    for t in range(0, T):
        tmp = w
        
        h = (training_X * w - training_Y).T
        tmp = w - gamma * (2.0/training_N) * (h * training_X).T
        #  for j in range(0, 785):
            #  tmp[j] = w[j] - gamma * (1.0/training_N) * h * training_X[:,j]

        w = tmp

    return w

def train_err(w):
    err = np.square(training_X * w - training_Y)
    loss = np.sum(err) / training_N
    return loss

def val_err(w):
    err = np.square(val_X * w - val_Y)
    loss = np.sum(err) / val_N
    return loss

def test_err(w):
    err = np.square(test_X * w - test_Y)
    loss = np.sum(err) / test_N
    return loss

T = 1000
eta_list = [3.5e-7, 3e-7, 1e-7, 7e-7, 1e-8, 5e-8, 1e-9, 5e-9, 1e-10, 5e-10]
best_eta = 0 
lowest_err = float("inf")

for eta in eta_list:
    w = train(T, eta)
    
    tr_err = train_err(w)
    v_err = val_err(w)

    print "eta: " + str(eta)
    print "training error: " + str(tr_err)
    print "validation error: " + str(v_err)
    print
    
    if v_err < lowest_err:
        best_eta = eta
        lowest_err = v_err

#  best_T = 1
#  lowest_err = float("inf")
#  
#  T_values = range(1, 1001)
#  err_list = []
#  for T in T_values:
    #  print T
#  
    #  w = train(T, best_eta)
    #  err = val_err(w)
#      
    #  err_list.append(err)
#  
    #  if err < lowest_err:
        #  best_T = T
        #  lowest_err = err
#  
#  w = train(best_T, best_eta)
w = train(1000, best_eta)
print test_err(w)

