import numpy as np
import os
import os.path
import pickle

def run():
    training_X = None
    training_Y = None
    training_N = 0

    val_X = None
    val_Y = None
    val_N = 0

    test_X = None
    test_Y = None
    test_N = 0

    if not os.path.exists('pickle'):
        os.makedirs('pickle')

    if os.path.isfile('pickle/training_X.pickle') and os.path.isfile('pickle/training_Y.pickle') and os.path.isfile('pickle/training_N.pickle'):
        
        print 'using saved training matrix'
        
        with open('pickle/training_X.pickle') as f:
            training_X = pickle.load(f)
        with open('pickle/training_Y.pickle') as f:
            training_Y = pickle.load(f)
        with open('pickle/training_N.pickle') as f:
            training_N = pickle.load(f)
    else:
        print "building training matrix..."
            
        training_data = open('data/trainingData.txt', 'r')
        training_labels = open('data/trainingLabels.txt', 'r')

        training_X = None
        mat_str = '' 
        for line in training_data:
            line = line.replace(',', ' ')
            mat_str += '1.0 ' + str(line) + ';'

        mat_str = mat_str[0:len(mat_str)-1]
        training_X = np.matrix(mat_str)

        training_Y = None
        training_N = 0
        mat_str = ''
        for line in training_labels:
            training_N += 1
            line = line.replace(',', ' ')
            mat_str += str(line) + ';'

        mat_str = mat_str[0:len(mat_str)-1]
        training_Y = np.matrix(mat_str)

        with open('pickle/training_X.pickle', 'w') as f:
            pickle.dump(training_X, f)
        with open('pickle/training_Y.pickle', 'w') as f:
            pickle.dump(training_Y, f)
        with open('pickle/training_N.pickle', 'w') as f:
            pickle.dump(training_N, f)

    if os.path.isfile('pickle/val_X.pickle') and os.path.isfile('pickle/val_Y.pickle') and os.path.isfile('pickle/val_N.pickle'):

        print 'using saved validation matrix'

        with open('pickle/val_X.pickle') as f:
            val_X = pickle.load(f)
        with open('pickle/val_Y.pickle') as f:
            val_Y = pickle.load(f)
        with open('pickle/val_N.pickle') as f:
            val_N = pickle.load(f)
    else:
        print "building validation matrix..."

        validation_data = open('data/validationData.txt', 'r')
        validation_labels = open('data/validationLabels.txt', 'r')

        val_X = None
        mat_str = '' 
        for line in validation_data:
            line = line.replace(',', ' ')
            mat_str += '1.0 ' + str(line) + ';'

        mat_str = mat_str[0:len(mat_str)-1]
        val_X = np.matrix(mat_str)

        val_Y = None
        val_N = 0
        mat_str = ''
        for line in validation_labels:
            val_N += 1
            line = line.replace(',', ' ')
            mat_str += str(line) + ';'

        mat_str = mat_str[0:len(mat_str)-1]
        val_Y = np.matrix(mat_str)
        
        with open('pickle/val_X.pickle', 'w') as f:
            pickle.dump(val_X, f)
        with open('pickle/val_Y.pickle', 'w') as f:
            pickle.dump(val_Y, f)
        with open('pickle/val_N.pickle', 'w') as f:
            pickle.dump(val_N, f)

    if os.path.isfile('pickle/test_X.pickle') and os.path.isfile('pickle/test_Y.pickle') and os.path.isfile('pickle/test_N.pickle'):

        print 'using saved test matrix'

        with open('pickle/test_X.pickle') as f:
            test_X = pickle.load(f)
        with open('pickle/test_Y.pickle') as f:
            test_Y = pickle.load(f)
        with open('pickle/test_N.pickle') as f:
            test_N = pickle.load(f)
    else:
        print "building test matrix..."

        test_data = open('data/testData.txt', 'r')
        test_labels = open('data/testLabels.txt', 'r')

        test_X = None
        mat_str = '' 
        for line in test_data:
            line = line.replace(',', ' ')
            mat_str += '1.0 ' + str(line) + ';'

        mat_str = mat_str[0:len(mat_str)-1]
        test_X = np.matrix(mat_str)

        test_Y = None
        test_N = 0
        mat_str = ''
        for line in test_labels:
            test_N += 1
            line = line.replace(',', ' ')
            mat_str += str(line) + ';'

        mat_str = mat_str[0:len(mat_str)-1]
        test_Y = np.matrix(mat_str)
        
        with open('pickle/test_X.pickle', 'w') as f:
            pickle.dump(test_X, f)
        with open('pickle/test_Y.pickle', 'w') as f:
            pickle.dump(test_Y, f)
        with open('pickle/test_N.pickle', 'w') as f:
            pickle.dump(test_N, f)

    return (training_X, training_Y, training_N, val_X, val_Y, val_N, test_X, test_Y, test_N)


