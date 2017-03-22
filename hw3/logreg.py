import gradientdescent as gd
import numpy as np

def loss(X, Y, w, N):
    print -w.T * X.T * Y 
    return np.log(1 + np.exp(-w.T * X.T * Y)) 

def gradient(X, Y, w, eta, N):
    h = float(np.exp(-w.T * X.T * Y))
    return w - eta * (1.0 / N) * h / (1 + h) * (-X.T * Y) 

eta_values = [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 3.5e-8, 5e-8, 1e-7, 3.5e-7, 5e-7]

gd.run_experiment(loss, gradient, eta_values)

