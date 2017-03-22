import gradientdescent as gd
import numpy as np

def loss(X, Y, w, N):
    return numpy.log(1 + exp(-w.T * X.T * Y)) 

def gradient(X, Y, w, eta, N):
    return w - eta * (1.0 / N) * np.exp(-w.T * X.T * Y) * (-X.T * Y) / (1 + np.exp(-w.T * X.T * Y))

gd.run_experiment(loss, gradient)

