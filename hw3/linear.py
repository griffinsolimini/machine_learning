from gd_runner import run_gd 
import numpy as np

def gradient(X, Y, w, N):
    return (2 * (X * w - Y).T * X).T / N

run_gd(gradient)
