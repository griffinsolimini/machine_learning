import numpy
import matplotlib.pyplot as plt
import math

mean_pos = [1, 0]
mean_neg = [-1, 0]
cov_pos = [[1, 0], [0, 1]]
cov_neg = [[1, .5], [.5, 1]]


# Calculate sample mean and sample covariance for positive class
x, y = .5 * numpy.random.multivariate_normal(mean_pos, cov_pos, 50).T

sample_mean_pos = numpy.matrix([numpy.mean(x), numpy.mean(y)]).T

sample_cov_pos = numpy.matrix([[0.0, 0.0],[0.0, 0.0]])
for i, j in zip(x, y):
    tmp = numpy.matrix([i, j]).T
    sample_cov_pos += (tmp - sample_mean_pos) * (tmp - sample_mean_pos).T
sample_cov_pos /= 50.0

# Calculate sample mean and sample covariance for negative class
x, y = .5 * numpy.random.multivariate_normal(mean_neg, cov_neg, 50).T

sample_mean_neg = numpy.matrix([numpy.mean(x), numpy.mean(y)]).T

sample_cov_neg = numpy.matrix([[0.0, 0.0],[0.0, 0.0]])
for i, j in zip(x, y):
    tmp = numpy.matrix([i, j]).T
    sample_cov_neg += (tmp - sample_mean_neg) * (tmp - sample_mean_neg).T
sample_cov_neg /= 50.0

Cw = sample_cov_pos + sample_cov_neg

print Cw.I * (sample_mean_pos - sample_mean_neg)

