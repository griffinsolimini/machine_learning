import numpy as np
import os.path
import pickle

training_X = None
training_Y = None
training_N = 0

val_X = None
val_Y = None

test_X = None
test_Y = None

if os.path.isfile('pickle/training_X.pickle') and os.path.isfile('pickle/training_Y.pickle') and os.path.isfile('pickle/training_N.pickle'):
    with open('pickle/training_X.pickle') as f:
        training_X = pickle.load(f)
    with open('pickle/training_Y.pickle') as f:
        training_Y = pickle.load(f)
    with open('pickle/training_N.pickle') as f:
        training_N = pickle.load(f)
else:
    print "building training matrix..."
        
    training_data = open('trainingData.txt', 'r')
    training_labels = open('trainingLabels.txt', 'r')

    training_X = None
    mat_str = '' 
    for line in training_data:
        line = line.replace(',', ' ')
        mat_str += str(line) + ';'

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

if os.path.isfile('pickle/val_X.pickle') and os.path.isfile('pickle/val_Y.pickle'):
    with open('pickle/val_X.pickle') as f:
        val_X = pickle.load(f)
    with open('pickle/val_Y.pickle') as f:
        val_Y = pickle.load(f)
else:
    print "building validation matrix..."

    validation_data = open('validationData.txt', 'r')
    validation_labels = open('validationLabels.txt', 'r')

    val_X = None
    mat_str = '' 
    for line in validation_data:
        line = line.replace(',', ' ')
        mat_str += str(line) + ';'

    mat_str = mat_str[0:len(mat_str)-1]
    val_X = np.matrix(mat_str)

    val_Y = None
    mat_str = ''
    for line in validation_labels:
        line = line.replace(',', ' ')
        mat_str += str(line) + ';'

    mat_str = mat_str[0:len(mat_str)-1]
    val_Y = np.matrix(mat_str)
    
    with open('pickle/val_X.pickle', 'w') as f:
        pickle.dump(val_X, f)
    with open('pickle/val_Y.pickle', 'w') as f:
        pickle.dump(val_Y, f)

if os.path.isfile('pickle/test_X.pickle') and os.path.isfile('pickle/test_Y.pickle'):
    with open('pickle/test_X.pickle') as f:
        test_X = pickle.load(f)
    with open('pickle/test_Y.pickle') as f:
        test_Y = pickle.load(f)
else:
    print "building test matrix..."

    test_data = open('testData.txt', 'r')
    test_labels = open('testLabels.txt', 'r')

    test_X = None
    mat_str = '' 
    for line in test_data:
        line = line.replace(',', ' ')
        mat_str += str(line) + ';'

    mat_str = mat_str[0:len(mat_str)-1]
    test_X = np.matrix(mat_str)

    test_Y = None
    mat_str = ''
    for line in test_labels:
        line = line.replace(',', ' ')
        mat_str += str(line) + ';'

    mat_str = mat_str[0:len(mat_str)-1]
    test_Y = np.matrix(mat_str)
    
    with open('pickle/test_X.pickle', 'w') as f:
        pickle.dump(test_X, f)
    with open('pickle/test_Y.pickle', 'w') as f:
        pickle.dump(test_Y, f)

print training_X.shape
print training_Y.shape
print training_N

        #  w = w - (gamma * (2 * (training_X * w - training_Y).T * training_X).T / training_N)
def train(T, gamma):
    w = np.zeros((784,1))
    print gamma
    for i in range(0, T):
        #  print (-2 / training_N) * ( (training_Y - training_X * w ).T * training_X).T
        print w.item(0)
        print (gamma * (-2 / training_N) * ( (training_Y - training_X * w ).T * training_X).T).item(0)
        w -= (gamma * (-2.0/training_N) * ((training_Y - training_X * w).T * training_X).T)
    return w

def validate(w):
    print (val_X * w)
    err = np.absolute(np.sign(val_Y - np.sign(val_X * w)))
    return np.sum(err) / err.size

def test(w):
    err = np.absolute(np.sign(test_Y - np.sign(test_X * w)))
    return np.sum(err) / err.size

w = train(10, .0001)
print validate(w)

#  print "calculating best gamma..."
#  
#  gamma = [.00000001, .0000001]
#  T = 1000 
#  
#  best_gamma = .0001
#  lowest_err = float("inf")
#  
#  for g in gamma:
    #  w = train(T, g)
    #  err = validate(w)
#      
    #  if err < lowest_err:
        #  best_gamma = g
        #  lowest_err = err
#  
#  print "calculating best T..."
#  
#  best_T = 1
#  lowest_err = float("inf")
#  
#  err_plot = []
#  for T in range(1,1001):
    #  w = train(T, best_gamma)
    #  err = validate(w)
#      
    #  err_plot.append(err)
#  
    #  if err < lowest_err:
        #  best_T = T
        #  lowest_err = err
#  
#  w = train(best_T, best_gamma)
#  best_error = validate(w)
#  
#  print "best T: " + str(best_T)
#  print "best gamma: " + str(best_gamma)
#  print "best error: " + str(best_error)

