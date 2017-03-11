import numpy as np

def run_gd(gradient):
    test_file = open('testData.txt', 'r')
    test_labels = open('testLabels.txt', 'r')

    X = None
    mat_str = '' 
    for line in test_file:
        line = line.replace(',', ' ')
        mat_str += str(line) + ';'

    mat_str = mat_str[0:len(mat_str)-1]
    X = np.matrix(mat_str)

    Y = None
    N = 0
    mat_str = ''
    for line in test_labels:
        N += 1
        line = line.replace(',', ' ')
        mat_str += str(line) + ';'

    mat_str = mat_str[0:len(mat_str)-1]
    Y = np.matrix(mat_str)

    w = np.zeros((784,1))

    print X.shape
    print Y.shape
    print w.shape

    print (X*w).shape

    #  def gradient():
        #  return (2 * (X * w - Y).T * X).T / N

    T = 1 
    gamma = 1 
    for i in range(0, T):
        w = w - (gamma * gradient(X, Y, w, N))

    # Validation 
    validation_data = open('validationData.txt', 'r')
    validation_labels = open('validationLabels.txt', 'r')

    X = None
    mat_str = '' 
    for line in validation_data:
        line = line.replace(',', ' ')
        mat_str += str(line) + ';'

    mat_str = mat_str[0:len(mat_str)-1]
    X = np.matrix(mat_str)

    Y = None
    N = 0
    mat_str = ''
    for line in validation_labels:
        N += 1
        line = line.replace(',', ' ')
        mat_str += str(line) + ';'

    mat_str = mat_str[0:len(mat_str)-1]
    Y = np.matrix(mat_str)

    err = np.absolute(Y - np.sign(X * w))

    print "validation error: " + str(np.sum(err) / err.size)

    # Testing 
    test_data = open('testData.txt', 'r')
    test_labels = open('testLabels.txt', 'r')

    X = None
    mat_str = '' 
    for line in test_data:
        line = line.replace(',', ' ')
        mat_str += str(line) + ';'

    mat_str = mat_str[0:len(mat_str)-1]
    X = np.matrix(mat_str)

    Y = None
    N = 0
    mat_str = ''
    for line in test_labels:
        N += 1
        line = line.replace(',', ' ')
        mat_str += str(line) + ';'

    mat_str = mat_str[0:len(mat_str)-1]
    Y = np.matrix(mat_str)

    err = np.absolute(Y - np.sign(X * w))

    print "test error: " + str(np.sum(err) / err.size)

