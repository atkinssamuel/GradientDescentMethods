from data_utils import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import psutil
import copy
np.random.seed(0)
x_train, x_valid, x_test, y_train, y_valid, y_test = ([[],[],[],[],[]], [[],[],[],[],[]],
                                                     [[],[],[],[],[]], [[],[],[],[],[]],
                                                     [[],[],[],[],[]], [[],[],[],[],[]])
x_train[0], x_valid[0], x_test[0], y_train[0], y_valid[0], y_test[0] = list(load_dataset('mauna_loa'))
x_train[1], x_valid[1], x_test[1], y_train[1], y_valid[1], y_test[1] = list(load_dataset('rosenbrock', n_train=5000, d=2))
x_train[2], x_valid[2], x_test[2], y_train[2], y_valid[2], y_test[2] = list(load_dataset('pumadyn32nm'))
x_train[3], x_valid[3], x_test[3], y_train[3], y_valid[3], y_test[3] = list(load_dataset('iris'))
x_train[4], x_valid[4], x_test[4], y_train[4], y_valid[4], y_test[4] = list(load_dataset('mnist_small'))
inf = 10000000

def RMSE(measurements, actuals):
    n = len(measurements)
    sum = 0
    for i in range(n):
        sum += (measurements[i] - actuals[i])**2

    return math.sqrt(sum/n)


def gradientDescent():
    xTrain = copy.deepcopy(x_train[2][:1000])
    xTrain = np.c_[np.ones(xTrain.shape[0]), xTrain]
    yTrain = copy.deepcopy(y_train[2][:1000])

    learningRates = [[0.1, 'b'], [0.01, 'g'], [0.001, 'm'], [0.0001, 'c']]
    maxIterations = 1000
    iterationDomain = [i for i in range(maxIterations)]

    grad = lambda x, y, w: np.expand_dims(sum(2*(np.dot(x, w) - y)*x)/y.shape[0], axis=1)

    for learningRate, colorCode in learningRates:

        weightsFull = np.zeros(len(xTrain[0]))
        weightsFull = np.expand_dims(weightsFull, axis=1)

        weightsStoch = np.zeros(len(xTrain[0]))
        weightsStoch = np.expand_dims(weightsStoch, axis=1)
        
        gradErrors = []
        stochErrors = []
        minGradError = inf
        minStochError = inf

        for i in range(maxIterations):
            # Full gradient descent:
            weightsFull = weightsFull - learningRate * grad(xTrain, yTrain, weightsFull)
            yPredictions = np.matmul(xTrain, weightsFull)
            error = RMSE(yPredictions, yTrain)
            gradErrors.append(error)
            if error < minGradError:
                minGradError = error

            # Stochastic gradient descent:
            randomIndex = int(np.random.random()*len(yTrain)) 
            weightsStoch = weightsStoch - learningRate * grad(xTrain[randomIndex], yTrain[randomIndex], weightsStoch)
            yPredictions = np.matmul(xTrain, weightsStoch)
            error = RMSE(yPredictions, yTrain)
            stochErrors.append(error)
            if error < minStochError:
                minStochError = error
        
        # Gradient Plotting:
        fig, ax = plt.subplots()
        ax.plot(iterationDomain, gradErrors, '-{}'.format(colorCode), markersize=1, \
            label='Learning Rate = {}, Minimum Error = {}'\
            .format(learningRate, round(minGradError, 4)))
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.legend()
        plt.title('Gradient Descent Error vs Iterations')
        fig.savefig('results/GD/GDRate_{}.png'.format(learningRate))

        # Stochastic Plotting:
        fig, ax = plt.subplots()
        ax.plot(iterationDomain, stochErrors, '-{}'.format(colorCode), markersize=1, \
            label='Learning Rate = {} Minimum Error = {}'\
            .format(learningRate, round(minStochError, 8)))
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.legend()
        plt.title('Stochastic Gradient Descent Error vs Iterations')
        fig.savefig('results/SGD/SGDRate_{}.png'.format(learningRate))
       
    return 

if __name__ == "__main__":
    gradientDescent()
    