import gradientdescent as gd
import numpy as np

def loss(K, Y, w, l, N):
    loss = 0
    for i in range(N):
        h = float(Y[i] * (K*w)[i])
        if h < 1:
            loss += 1 - h
    
    loss *= 1.0 / N
    loss += (l / 2) * w.T * K * w

    return loss
    #  h = float(w.T * KY)
    #  if 1 > h:
        #  return (1.0 / N) * (1 - h) + (l / 2) * w.T * K * w
    #  else:
        #  return (l / 2) * w.T * w

def gradient(K, Y, w, eta, l, N):
    gsum = 0
    for i in range(N):
        h = float(Y[i] * (K*w)[i])
        if h < 1:
            gsum += K[i].T * Y[i]

    return w - eta * ((1.0 / N) * gsum + l * K * w)



    #  h = float(w.T * KY)
    #  if 1 > h:
        #  return w - eta * ( (1.0 / N) * (-KY) + l * K * w ) 
    #  else:
        #  return w - eta * (l * K * w) 

#  lambda_values = [.1, .01, .001, .0001, .00001, .000001, .0000001]
#  eta_values = [.1, .01, .001, .0001, .00001, .000001, .0000001, .00000001]

lambda_values = [1e-18]
eta_values = [1e-18]

gd.run_experiment(loss, gradient, eta_values, lambda_values)

