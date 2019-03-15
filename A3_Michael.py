"""
ROB313 Assignment 3
 - Mar. 14th, 2019
 - Michael Shanks-Marshall, 1003363451
"""

import numpy as np
from matplotlib import pyplot
from data_utils import load_dataset

np.random.seed(0)

### Extra Functions

def least_sq_loss(x, w, y):
    f_hat = np.dot(x, np.transpose(w))
    f_hat = f_hat.reshape(-1, 1)
    loss = sum((f_hat - y)**2)/y.shape[0]
    return loss
    
def least_sq_grad(x, w, y):
    f_hat = np.dot(x, np.transpose(w))
    f_hat = f_hat.reshape(-1, 1)
    grad = sum(2*(f_hat - y)*x)/y.shape[0]
    return grad

def Fsigmoid(x, w):
    z = np.dot(x, w)
    sig = (1 + np.exp(-z))**-1
    return sig

def log_likelihood(x, w, y):
    f_hat = Fsigmoid(x, w)
    f_hat = f_hat.reshape(-1, 1)
    lkhood = 0
    for i in range(len(y)):
        f_hat_i = f_hat[i]
        if f_hat_i != 1:
            lkhood += y[i]*np.log(f_hat_i) + (1 - y[i])*np.log(1 - f_hat_i)
    return lkhood/len(y)

def log_like_grad(x, w, y):
    f_hat = Fsigmoid(x, w)
    grad = np.dot(np.transpose(x), (y - f_hat))
    return grad

def accuracy(x, w, y):
    f_hat = Fsigmoid(x, w)
    f_hat = f_hat > 0.5
    total = f_hat == y
    return sum(total)[0]/len(total)*100

### Question 1 - Gradient and Sochastic Descent

def gradDescent(name):
    '''
    
    '''
    
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(name)
    x_train = x_train[:1000]
    x_train = np.c_[np.ones(x_train.shape[0]), x_train]     # Add a dummy variable as a column of ones
    y_train = y_train[:1000]

    all_weights_full_opt = []
    all_RMSE_full = []
    
    all_weights_soch_opt = []
    all_RMSE_soch = []
    
    # SVD_RMSE = 0.862
    N = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    figNum = 1
    
    for n in N:     # Iterate over many different learning rates to find the best rate for a given model
        
        weights_full = np.zeros(x_train.shape[1])
        weights_full_opt = weights_full
        RMSE_full = float('Inf')
        
        weights_soch = np.zeros(x_train.shape[1])
        weights_soch_opt = weights_soch
        RMSE_soch = float('Inf')
        
        RMSE_full_plot = []
        RMSE_soch_plot = []
        iters = 1
        
        while iters < 2000:
            
            weights_full = weights_full - n*least_sq_grad(x_train, weights_full, y_train)   # Make a step forward
            RMSE_tmp = (least_sq_loss(x_train, weights_full, y_train))**0.5
            if RMSE_tmp < RMSE_full:            # Evaluate the best found weights by comparing the RMSE
                RMSE_full = RMSE_tmp
                weights_full_opt = weights_full
            
            RMSE_full_plot.append(RMSE_tmp)
            
            randIndex = int(np.random.random()*y_train.shape[0])    # Choose a random point to evaluate the sochastic gradient
            weights_soch = weights_soch - n*least_sq_grad(x_train[randIndex], weights_soch, y_train[randIndex]) # Make a step forward
            RMSE_tmp = (least_sq_loss(x_train, weights_soch, y_train))**0.5
            if RMSE_tmp < RMSE_soch:
                RMSE_soch = RMSE_tmp
                weights_soch_opt = weights_soch
            
            RMSE_soch_plot.append(RMSE_tmp)
            
            iters += 1
            
        all_weights_full_opt.append(weights_full_opt)
        all_RMSE_full.append(RMSE_full[0])
        
        all_weights_soch_opt.append(weights_soch_opt)
        all_RMSE_soch.append(RMSE_soch[0])
        
        x_range = [i for i in range(iters-1)]       # Plot over the range of iterations
        
        pyplot.figure(figNum)
        pyplot.plot(x_range, RMSE_full_plot, color='blue', label='Full-Batch')
        pyplot.plot(x_range, RMSE_soch_plot, color='red', label='Sochastic')
        pyplot.title("Full-Batch and Sochastic Descent for n={}".format(n))
        pyplot.xlabel("iteration")
        pyplot.ylabel("RMSE")
        pyplot.legend()
        pyplot.show()
        
        figNum += 1
    
    best_RMSE_full = min(all_RMSE_full)                 # Find the best weights and learning rate for the full-batch model
    full_index = all_RMSE_full.index(best_RMSE_full)
    best_weights_full = all_weights_full_opt[full_index]
    best_N_full = N[full_index]
    
    best_RMSE_soch = min(all_RMSE_soch)                 # Find the best weights and learning rate for the sochastic model
    soch_index = all_RMSE_soch.index(best_RMSE_soch)
    best_weights_soch = all_weights_soch_opt[soch_index]
    best_N_soch = N[soch_index]
    
    f_hat_full = np.dot(np.c_[np.ones(x_test.shape[0]), x_test], best_weights_full)
    f_hat_soch = np.dot(np.c_[np.ones(x_test.shape[0]), x_test], best_weights_soch)
    
    test_RMSE_full = ((least_sq_loss(np.c_[np.ones(x_test.shape[0]), x_test], best_weights_full, y_test))**0.5)[0]
    test_RMSE_soch = ((least_sq_loss(np.c_[np.ones(x_test.shape[0]), x_test], best_weights_soch, y_test))**0.5)[0]
    
    return [best_RMSE_full, best_weights_full, best_N_full, test_RMSE_full], [best_RMSE_soch, best_weights_soch, best_N_soch, test_RMSE_soch]
    
### Question 2 - Gradient and Sochastic Descent for Logistic Regression

def gradDescentLogistic(name):
    '''
    
    '''
    
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(name)
    x_train = np.c_[np.ones(x_train.shape[0]), x_train]     # Add a dummy variable as a column of ones
    y_train = y_train[:, (1,)]

    all_weights_full_opt = []
    all_lkhood_full = []
    
    all_weights_soch_opt = []
    all_lkhood_soch = []
    
    N = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    figNum = 1
    
    for n in N:     # Iterate over many different learning rates
        
        weights_full = np.zeros(x_train.shape[1]).reshape(-1, 1)
        weights_full_opt = weights_full
        lkhood_full = float('Inf')
        
        weights_soch = np.zeros(x_train.shape[1]).reshape(-1, 1)
        weights_soch_opt = weights_soch
        lkhood_soch = float('Inf')
        
        lkhood_full_plot = []
        lkhood_soch_plot = []
        iters = 1
        
        while iters < 2000:
            
            weights_full = weights_full + n*log_like_grad(x_train, weights_full, y_train)       # Make a step forward (in the positive 
                                                                                                #   direction, to minimize the negative]
                                                                                                #   log likelihood)
            lkhood_tmp = -log_likelihood(x_train, weights_full, y_train)
            if lkhood_tmp < lkhood_full:        # Evaluate best weights by comparing the model with best likelihood
                lkhood_full = lkhood_tmp
                weights_full_opt = weights_full
            
            lkhood_full_plot.append(lkhood_tmp)
            
            randIndex = int(np.random.random()*y_train.shape[0])        # Choose a random point to evaluate the sochastic gradient
            weights_soch = weights_soch + n*log_like_grad(x_train[randIndex].reshape(1, -1), weights_soch, y_train[randIndex].reshape(-1, 1))       # Make a step forward
            lkhood_tmp = -log_likelihood(x_train, weights_soch, y_train)
            if lkhood_tmp < lkhood_soch:
                lkhood_soch = lkhood_tmp
                weights_soch_opt = weights_soch
            
            lkhood_soch_plot.append(lkhood_tmp)
            
            iters += 1
            
        all_weights_full_opt.append(weights_full_opt)
        all_lkhood_full.append(lkhood_full[0])
        
        all_weights_soch_opt.append(weights_soch_opt)
        all_lkhood_soch.append(lkhood_soch[0])
        
        x_range = [i for i in range(iters-1)]       # Plot over the range of iterations
        
        pyplot.figure(figNum)
        pyplot.plot(x_range, lkhood_full_plot, color='blue', label='Full-Batch')
        pyplot.plot(x_range, lkhood_soch_plot, color='red', label='Sochastic')
        pyplot.title("Full-Batch and Sochastic Descent for n={}".format(n))
        pyplot.xlabel("iteration")
        pyplot.ylabel("Negative Log Likelihood")
        pyplot.legend()
        pyplot.show()
        
        figNum += 1
    
    best_lkhood_full = min(all_lkhood_full)                 # Find the best weights and learning rate for the full-batch model
    full_index = all_lkhood_full.index(best_lkhood_full)
    best_weights_full = all_weights_full_opt[full_index]
    best_N_full = N[full_index]
    
    best_lkhood_soch = min(all_lkhood_soch)                 # Find the best weights and learning rate for the sochastic model
    soch_index = all_lkhood_soch.index(best_lkhood_soch)
    best_weights_soch = all_weights_soch_opt[soch_index]
    best_N_soch = N[soch_index]
    
    f_hat_full = Fsigmoid(np.c_[np.ones(x_test.shape[0]), x_test], best_weights_full)
    f_hat_soch = Fsigmoid(np.c_[np.ones(x_test.shape[0]), x_test], best_weights_soch)
    
    y_test = y_test[:, (1,)]
    
    test_lkhood_full = -(log_likelihood(np.c_[np.ones(x_test.shape[0]), x_test], best_weights_full, y_test))[0]     # Find the
                                                                                                                    #   likelihood and
                                                                                                                    #   accuracy
    test_Acc_full = accuracy(np.c_[np.ones(x_test.shape[0]), x_test], best_weights_full, y_test)
    
    test_lkhood_soch = -(log_likelihood(np.c_[np.ones(x_test.shape[0]), x_test], best_weights_soch, y_test))[0]     # Find the
                                                                                                                    #   likelihood and
                                                                                                                    #   accuracy
    test_Acc_soch = accuracy(np.c_[np.ones(x_test.shape[0]), x_test], best_weights_soch, y_test)
    
    return [best_lkhood_full, best_weights_full, best_N_full, test_lkhood_full, test_Acc_full], [best_lkhood_soch, best_weights_soch, best_N_soch, test_lkhood_soch, test_Acc_soch]  
    
    
if __name__ == '__main__':
    
    #Q1Results = gradDescent('pumadyn32nm')
    
    Q2Results = gradDescentLogistic('iris')