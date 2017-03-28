import gradientdescent as gd
import numpy as np

def loss(X, Y, w, N):
    return np.log(1 + np.exp(-w.T * X.T * Y)) / N 

def gradient(X, Y, w, eta, N):
    h = float(np.exp(-w.T * X.T * Y))
    return w - eta * (1.0 / N) * h / (1 + h) * (-X.T * Y) 

eta_values = [1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8]

gd.run_experiment(loss, gradient, eta_values)

