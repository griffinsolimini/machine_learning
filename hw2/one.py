import numpy
import matplotlib.pyplot as plt
import math

# sample properties
mean_pos = [1, 0]
mean_neg = [-1, 0]
cov_pos = [[1, 0], [0, 1]]
cov_neg = [[1, .5], [.5, 1]]

# bayesian decision bound equation
def bayesianDecisionBound((x,y)):
    return (1/6 * x**2) + (7/3 * x) + (1/6 * y**2) - (2/3 * y) - (2/3 * x * y) + (1/6) - math.log(2 / math.sqrt(3)) >= 0

# data generation
x, y = .5 * numpy.random.multivariate_normal(mean_pos, cov_pos, 50).T
plt.plot(x, y, '+', label='Positive')

pos_pts = zip(x, y)

x, y = .5 * numpy.random.multivariate_normal(mean_neg, cov_neg, 50).T
plt.plot(x, y, 'r_', label='Negative')

neg_pts = zip(x, y)

# calculate bayesian decision bound test error

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

# equation for lda line
def lda_bound(x):
    return (float(direction[0]) * x + float(const)) / -float(direction[1])

# lda decision bound given a point
def ldaDecision((x,y)):
   return float(direction[0]) * x + float(direction[1]) * y + float(const) >= 0

# calcualte bayesian test error
bayesian_correct = 0
for pt in pos_pts:
    if bayesianDecisionBound(pt):
        bayesian_correct += 1

for pt in neg_pts:
    if not bayesianDecisionBound(pt):
        bayesian_correct += 1

print "bayesian test error: " + str(1 - (bayesian_correct / 100.0))

# calculate lda test error
lda_correct = 0
for pt in pos_pts:
    if ldaDecision(pt):
        lda_correct += 1

for pt in neg_pts:
    if not ldaDecision(pt):
        lda_correct += 1

print "lda test error: " + str(1 - (lda_correct / 100.0))

# plot lda line
x = numpy.arange(-3, 4, 1)
bound = lda_bound(x)

plt.plot(x, bound, '-', label='LDA Decision Boundary')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Bound')
plt.legend(loc='upper left')
plt.xlim(-2, 2)
plt.ylim(-4, 4)
plt.show()

