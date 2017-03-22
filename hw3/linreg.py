import gradientdescent as gd
import numpy as np

def loss(X, Y, w, N):
    return np.sum(np.square(X * w - Y)) / N

def gradient(X, Y, w, eta, N):
    return w - eta * (2.0 / N) * ((X * w - Y).T * X).T

gd.run_experiment(loss, gradient)

