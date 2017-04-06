import gradientdescent as gd
import numpy as np

def loss(K, KY, w, l, N):
    h = float(w.T * KY)
    if 1 > h:
        return (1.0 / N) * (1 - h) + (l / 2) * w.T * K * w
    else:
        return (l / 2) * w.T * K * w

def gradient(K, KY, w, eta, l, N):
    h = float(w.T * KY)
    if 1 > h:
        return w - eta * ( (1.0 / N) * (-KY) + l * K * w ) 
    else:
        return w - eta * (l * K * w) 

#  lambda_values = [.1, .01, .001, .0001, .00001, .000001, .0000001]
#  eta_values = [.1, .01, .001, .0001, .00001, .000001, .0000001, .00000001]

lambda_values = [.0000001]
eta_values = [.00000001]

gd.run_experiment(loss, gradient, eta_values, lambda_values)

