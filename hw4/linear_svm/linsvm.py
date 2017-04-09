import gradientdescent as gd
import numpy as np

def loss(X, Y, w, l, N):
    loss = 0
    for i in range(N):
        h = float(w.T * X[i].T * Y[i])
        if h < 1:
            loss += 1 - h
    
    loss *= 1.0 / N
    loss += (l / 2) * w.T * w
    
    return loss
    #  h = float(w.T * X.T * Y)
    #  if 1 > h:
        #  return (1.0 / N) * (1 - h) + (l / 2) * w.T * w
    #  else:
        #  return (l / 2) * w.T * w

def gradient(X, Y, w, eta, l, N):
    gsum = 0
    for i in range(N):
        h = float(w.T * X[i].T * Y[i])
        if h < 1:
            gsum += -X[i].T * Y[i]

    return w - eta * ((1.0 / N) * gsum + l * w)
    
    #  h = float(w.T * X.T * Y)
    #  if 1 > h:
        #  return w - eta * ( (1.0 / N) * (-X.T * Y) + l * w ) 
    #  elif 1 < h:
        #  return w - eta * (l * w) 
    #  else:
        #  print ('gradient undefined')
        #  return -1

lambda_values = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
eta_values = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]

lambda_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
eta_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

gd.run_experiment(loss, gradient, eta_values, lambda_values)

