import gradientdescent as gd
import numpy as np

def loss(X, Y, w, N):
    return (X * w - Y).T * (X * w - Y) / N

def gradient(X, Y, w, eta, N):
    return w - eta * (2.0 / N) * ((X * w - Y).T * X).T

eta_values = [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 3.5e-8, 5e-8, 1e-7, 3.5e-7, 5e-7]

gd.run_experiment(loss, gradient, eta_values)

