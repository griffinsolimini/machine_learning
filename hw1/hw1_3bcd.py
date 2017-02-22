import math
import numpy as np
import matplotlib.pyplot as pl

test_x = np.random.normal(0, 1, 100)
test_y = test_x * 0.5 + 0.5 + np.random.normal(0, 0.1, 100)

training_x = np.random.normal(0, 1, 100)
training_y = training_x * -0.5 + 0.5 + np.random.normal(0, 0.1, 100)

ones = np.ones(100)

training_x_matrix = np.matrix(np.column_stack((ones, training_x)))
training_y_matrix = np.matrix(training_y)

optimal_matrix = (training_x_matrix.T * training_x_matrix).I * training_x_matrix.T * training_y_matrix.T

pl.plot(training_x, training_y, 'o')

def estimator(x):
    return 0.5 + -0.5 * x

def optimal(x):
    return optimal_matrix.item(1) * x + optimal_matrix.item(0)

x = np.arange(-3, 4, 1)

est_y = estimator(x)
pl.plot(x, est_y, '-', label='estimated')

opt_y = optimal(x)
pl.plot(x, opt_y, '-', label='optimal')

pl.legend()
pl.show()

training_sum = 0
for i in range(0, len(training_x)):
    training_sum += (training_y[i] - (training_x[i] * -0.5 + 0.5)) ** 2

training_err = training_sum / len(training_x)

test_sum = 0
for i in range(0, len(test_x)):
    test_sum += (test_y[i] - (test_x[i] * optimal_matrix.item(1) + optimal_matrix.item(0))) ** 2

test_err = test_sum / len(test_x)

print("training error: " + str(training_err))
print("testing error: " + str(test_err))

