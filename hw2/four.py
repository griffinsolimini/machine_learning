import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import sys

# equation that returns the theoretical error 
def g_bound(N, d, delta):
    return math.sqrt(2*d*math.log(2.7182*N/d)/N)+math.sqrt(math.log(1/delta)/(2*N))

# function that performs algorithm 1 with N points
def algorithm1(N, plot):

    # Randomly generate rectangle
    rect_x = float(numpy.random.rand(1, 1))
    rect_y = float(numpy.random.rand(1, 1))

    rect_width = 1.0
    while rect_width + rect_x > 1.0:
        rect_width = float(numpy.random.rand(1, 1))

    rect_height = 1.0
    while rect_height + rect_y > 1.0:
        rect_height = float(numpy.random.rand(1, 1))
   
    # Randomly generate training points
    pts = numpy.random.rand(N, 2)

    # Classify points
    pos_pts = []
    neg_pts = []
    for pt in pts:
        x, y = pt
        if x >= rect_x and x <= rect_x + rect_width and y >= rect_y and y <= rect_y + rect_height:
            pos_pts.append((x, y))
        else:
            neg_pts.append((x, y))
    
    # If no positive points, exit
    if len(pos_pts) == 0:
        return -1
    
    # Generate smallest possible rectangle
    min_x = float("inf")
    max_x = 0.0
    min_y = float("inf")
    max_y = 0.0

    for pt in pos_pts:
        x, y = pt
        if x > max_x:
            max_x = x
        if x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        if y < min_y:
            min_y = y

    # Classify test points based on algorithm 1 rectangle 
    correct = 0
    test_pts = numpy.random.rand(N, 2)
    for pt in test_pts:
        x, y = pt
        if x >= rect_x and x <= rect_x+rect_width and y >= rect_y and y <= rect_y+rect_height:
            if x >= min_x and x <= max_x and y >= min_y and y <= max_y:
                correct += 1
        else:
            if x < min_x or x > max_x or y < min_y or y > max_y:
                correct += 1
   
    # Plot if flag set
    if plot:
        currentAxis = plt.gca()
        currentAxis.add_patch(patches.Rectangle((rect_x, rect_y), 
                                                rect_width,
                                                rect_height, 
                                                alpha=1, 
                                                facecolor='none'))

        currentAxis.add_patch(patches.Rectangle((min_x, min_y),
                                            max_x - min_x,
                                            max_y - min_y,
                                            alpha=1,
                                            facecolor='none',
                                            edgecolor="magenta"))

        x, y = zip(*pos_pts)
        plt.plot(x, y,'+', label="Positive")

        x, y = zip(*neg_pts)
        plt.plot(x, y, 'r_', label="Negative")
        
        plt.plot([],[], 'm-', label="Algorithm 1")

        plt.plot([],[], 'k-', label="Concept")

        plt.xlim(-0.25, 1.25)
        plt.ylim(-0.25, 1.25)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Algorithm 1')
        plt.legend(loc="upper left")
        plt.show()
   
    # return test error
    return (1 - correct / float(N))


# Perform algorithm once
result = -1
while result == -1:
    result = algorithm1(100, True)

print "test error from one trial, N=100: " + str(result)

# build histogram for T=100 N=100
results = []
i = 0
while i < 100:
   result = algorithm1(100, False)
   if result != -1:
       results.append(result)
       i += 1

plt.hist(results, 10)
plt.xlabel('Test Error')
plt.ylabel('Frequency')
plt.title('Algorithm 1 with N=100')
plt.show()

print "theoretical error for T=100 N=100: " + str(g_bound(100, 4.0, 0.01))

# build histogram for T=100 N=50
results = []
i = 0
while i < 100:
   result = algorithm1(50, False)
   if result != -1:
       results.append(result)
       i += 1

plt.hist(results, 10)
plt.xlabel('Test Error')
plt.ylabel('Frequency')
plt.title('Algorithm 1 with N=50')
plt.show()

print "theoretical error for T=100 N=50: " + str(g_bound(50, 4.0, 0.01))

# build histogram for T=100 N=200
results = []
i = 0
while i < 100:
   result = algorithm1(200, False)
   if result != -1:
       results.append(result)
       i += 1

plt.hist(results, 10)
plt.xlabel('Test Error')
plt.ylabel('Frequency')
plt.title('Algorithm 1 with N=200')
plt.show()

print "theoretical error for T=100 N=200: " + str(g_bound(200, 4.0, 0.01))

