import gradientdescent as gd
import numpy as np

def loss(X, X2, Y, w, l, N):
    K = (X * X2.T + 1) ** 3
    h = float(Y.T * K * w)
    if 1 > h:
        return (1.0 / N) * (1 - h) + (l / 2) * w.T * K * w
    else:
        return (l / 2) * w.T * K * w

def gradient(X, Y, w, eta, l, N):
    K = (X * X.T + 1) ** 3
    print Y.T.shape
    print K.shape
    print w.shape
    h = float(Y.T * K * w)
    if 1 > h:
        return w - eta * ( (1.0 / N) * (-K * Y) + l * K * w ) 
    elif 1 < h:
        return w - eta * (l * K * w) 
    else:
        print ('gradient undefined')
        return -1

lambda_values = [.1, .01, .001, .0001, .00001, .000001, .0000001]
eta_values = [.1, .01, .001, .0001, .00001, .000001, .0000001, .00000001]

gd.run_experiment(loss, gradient, eta_values, lambda_values)

