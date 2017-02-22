import numpy
import matplotlib.pyplot as plt
import math

# sample properties
mean_pos = [1, 0]
mean_neg = [-1, 0]
cov_pos = [[1, 0], [0, 1]]
cov_neg = [[1, .5], [.5, 1]]

# bayesian decision bound equation
def decisionBound((x,y)):
    return (-1/3 * x**2) - (14/3 * x) + (7/3 * y**2) + (4/3 * y) + (4/3 * x * y) - (1/3) + 2 * math.log(2 / math.sqrt(3))

# data generation
x, y = .5 * numpy.random.multivariate_normal(mean_pos, cov_pos, 50).T
plt.plot(x, y, '+')

pos_pts = zip(x, y)

x, y = .5 * numpy.random.multivariate_normal(mean_neg, cov_neg, 50).T
plt.plot(x, y, 'r_')

neg_pts = zip(x, y)

# calculate bayesian decision bound test error
correct = 0
for pt in pos_pts:
    if decisionBound(pt) <= 0:
        correct += 1

for pt in neg_pts:
    if decisionBound(pt) > 0:
        correct += 1

print "bayesian test error: " + str(correct / 100.0)

# calculate LDA
# Calculate sample mean and sample covariance for positive class
x, y = zip(*pos_pts)
sample_mean_pos = numpy.matrix([numpy.mean(x), numpy.mean(y)]).T

sample_cov_pos = numpy.matrix([[0.0, 0.0],[0.0, 0.0]])
for i, j in pos_pts:
    tmp = numpy.matrix([i, j]).T
    sample_cov_pos += (tmp - sample_mean_pos) * (tmp - sample_mean_pos).T
sample_cov_pos /= 50.0

# Calculate sample mean and sample covariance for negative class
x, y = zip(*neg_pts)
sample_mean_neg = numpy.matrix([numpy.mean(x), numpy.mean(y)]).T

sample_cov_neg = numpy.matrix([[0.0, 0.0],[0.0, 0.0]])
for i, j in zip(x, y):
    tmp = numpy.matrix([i, j]).T
    sample_cov_neg += (tmp - sample_mean_neg) * (tmp - sample_mean_neg).T
sample_cov_neg /= 50.0

Cw = sample_cov_pos + sample_cov_neg
direction = Cw.I * (sample_mean_pos - sample_mean_neg)
const = -.5 * sample_mean_pos.T * Cw.I * sample_mean_pos + 0.5 * sample_mean_neg.T * Cw.I * sample_mean_neg 

print direction
print const

# show plot
#  plt.ylim(-6, 6)
#  plt.show()

