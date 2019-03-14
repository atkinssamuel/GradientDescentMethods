# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:18:48 2019

@author: datta
"""
import array
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import numpy.linalg as la 
import time

def RMSEfunction(y_test,y_predict):
    
    RMSE = 0 
    for i in range(len(y_predict)):
        RMSE += pow((y_predict[i][0]-y_test[i]),2)
    RMSE = math.sqrt(RMSE/len(y_predict))
    
    return RMSE

def update_weights(x_train,y_train,y_predict,alpha,curr_weights,curr_b):
    grad_weights = np.zeros(len(x_train[0]))
    grad_b = 0
    for i in range(len(x_train)):
        grad_weights += x_train[i]*(y_train[i]-y_predict[i])
        grad_b += y_train[i]-y_predict[i]

    grad_weights = -2*grad_weights/len(x_train)
    grad_b = -2*grad_b/len(x_train)
    new_weights = curr_weights - alpha*grad_weights
    new_b = curr_b - alpha*grad_b
    
    return new_weights, new_b

def stochastic_update_weights(x_train,y_train,y_predict,alpha,curr_weights,curr_b):
    i = random.randint(0,len(x_train)-1)
    grad_weights = x_train[i]*(y_train[i]-y_predict[i])
    grad_b = y_train[i]-y_predict[i]
    grad_weights = -2*grad_weights
    grad_b = -2*grad_b
    new_weights = curr_weights - alpha*grad_weights
    new_b = curr_b - alpha*grad_b
    
    return new_weights, new_b

def gradientdescent(x_train,y_train,alpha,weights,b,iterations):
    L = []
    k=0 
    curr_weights = weights
    curr_b = b
    min = 1000
    for i in range(iterations):
        y_predict = [(curr_weights.T).dot(x_train[i])+curr_b for i in range(len(x_train))]
        L.append(RMSEfunction(y_predict,y_train))
        curr_weights, curr_b = update_weights(x_train,y_train,y_predict,alpha,curr_weights,curr_b)
       
        k+=1
    
    return curr_weights, L, curr_b

def stochastic_gradientdescent(x_train,y_train,alpha,weights,b,iterations):
    L = []
    k=0 
    curr_weights = weights
    curr_b = b
    min = 1000
    for i in range(iterations):
        y_predict = [(curr_weights.T).dot(x_train[i])+curr_b for i in range(len(x_train))]
        L.append(RMSEfunction(y_predict,y_train))
        curr_weights, curr_b = stochastic_update_weights(x_train,y_train,y_predict,alpha,curr_weights,curr_b)
       
        k+=1
    
    return curr_weights, L, curr_b

def sigmoid(z):
    return 1/(1+np.exp(-z))

def f(x,w):
    f = sigmoid(np.dot(x,w))
    return f


def negloglikelihood(x_train,y_train,w):
    sum = 0
    for i in range(len(x_train)):
        h = f(x_train[i],w)
        y = 1*y_train[i]
        sum += y*math.log(h) + (1-y)*math.log(1-h)    
        
    return -sum  
    

def classification_update_weights(x_train,y_train,alpha,curr_weights):
    grad_weights = np.zeros(len(x_train[0]))
    
    for i in range(len(x_train)):
    
        grad_weights += (y_train[i]-f(x_train[i],curr_weights))*x_train[i]

    new_weights = curr_weights + alpha*grad_weights
    
    return new_weights

def stochastic_classification_update_weights(x_train,y_train,alpha,curr_weights):
   # grad_weights = np.zeros(len(x_train[0]))
    i = random.randint(0,len(x_train)-1)
    grad_weights = (y_train[i]-f(x_train[i],curr_weights))*x_train[i]

    new_weights = curr_weights + alpha*grad_weights
    
    return new_weights

def classification_gradientdescent(x_train,y_train,alpha,weights,iterations):
    likelihoods = []
    k=1 
    curr_weights = weights
    
    for i in range(iterations):
        likelihoods.append(negloglikelihood(x_train,y_train,curr_weights))
        curr_weights = classification_update_weights(x_train,y_train,alpha,curr_weights)
    
        k+=1
    
    return curr_weights, likelihoods

def stochastic_classification_gradientdescent(x_train,y_train,alpha,weights,iterations):
    likelihoods = []
    k=1 
    curr_weights = weights
    
    for i in range(iterations):
        likelihoods.append(negloglikelihood(x_train,y_train,curr_weights))
        curr_weights = stochastic_classification_update_weights(x_train,y_train,alpha,curr_weights)
    
        k+=1
    
    return curr_weights, likelihoods
#******************************************************************************
#Question 1
np.random.seed(10)    
from data_utils import load_dataset 
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('pumadyn32nm')
x_train, y_train = x_train[:1000], y_train[:1000]


N = 100
k = [i for i in range(N)]
results = []
learningrates = [1,0.5,0.1,0.01,0.001,0.0001]
for rate in learningrates:
    weights = np.zeros(len(x_train[0]))
    b = 0
    print(rate)
    weights, RMSEs,b = gradientdescent(x_train,y_train,rate,weights,b,N)
    results.append(RMSEs)
    plt.plot(k,RMSEs)
    plt.title('RMSE Loss vs iteration number; Learning rate = ' + str(rate))
    plt.xlabel('Iteration number')
    plt.ylabel('RMSE Loss')
    plt.figure()

#error diverges for rate = 1, coverges quickly for 0.7 and 0.5. As rate decreases loss vs iteration curve approaches a line
L = results[1:]

for r in L:
    plt.plot(k,r)
    plt.title('RMSE Loss vs iteration number')
    plt.xlabel('Iteration number')
    plt.ylabel('RMSE Loss')
plt.legend(["rate=0.5","rate=0.1","rate=0.01","rate=0.001","rate=0.0001"])
plt.figure()

#from the above, a good learning rate is 0.5 which converges to the min loss within very few iterations
weights, RMSEs,b = gradientdescent(x_test,y_test,0.5,weights,b,N)


y_predict = [(weights.T).dot(x_test[i])+b for i in range(len(x_test))]
RMSE = RMSEfunction(y_predict,y_test)
print("(learning rate = 0.5) Test RMSE:" + str(RMSE))
###############################################################################
#Stochastic
    
N = 1000
k = [i for i in range(N)]
results = []
learningrates = [1,0.5,0.1,0.01,0.001,0.0001,0.00001]
for rate in learningrates:
    weights = np.zeros(len(x_train[0]))
    b = 0
    print(rate)
    weights, RMSEs,b = stochastic_gradientdescent(x_train,y_train,rate,weights,b,N)
    results.append(RMSEs)
    plt.plot(k,RMSEs)
    plt.title('RMSE Loss vs iteration number (stochastic); Learning rate = ' + str(rate))
    plt.xlabel('Iteration number')
    plt.ylabel('RMSE Loss')
    plt.figure()
#error overflows for rate = 1 and 0.5. rate = 0.01 gives very erratic behavior  
L = results[3:]

for r in L:
    plt.plot(k,r)
    plt.title('RMSE Loss vs iteration number (stochastic)')
    plt.xlabel('Iteration number')
    plt.ylabel('RMSE Loss')
plt.legend(["rate=0.01","rate=0.001","rate=0.0001","rate=0.00001"])
plt.figure()

#from the above, a good learning rate is 0.001 which converges much more quickly than the other rates
weights, RMSEs,b = stochastic_gradientdescent(x_test,y_test,0.001,weights,b,N)


y_predict = [(weights.T).dot(x_test[i])+b for i in range(len(x_test))]
RMSE = RMSEfunction(y_predict,y_test)
print("Stochastic (learning rate = 0.001) Test RMSE:" + str(RMSE))


#*******************************************************************************
#Question 2 
    
from data_utils import load_dataset 
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
y_train, y_valid, y_test = y_train[:,(1,)], y_valid[:,(1,)], y_test[:,(1,)]
x_train = np.vstack([x_valid, x_train])
y_train = np.vstack([y_valid, y_train])
#insert dummy column of ones into x-data
ones = np.ones((x_train.shape[0], 1))
x_train = np.concatenate((ones, x_train), axis=1)
ones = np.ones((x_test.shape[0], 1))
x_test = np.concatenate((ones, x_test), axis=1)

#intialize weights to 0
weights = np.zeros(len(x_train[0]))
N = 1000
k = [i for i in range(N)]

#compute weights and likelihoods

results = []
learningrates = [0.01,0.001,0.0001,0.00001]
for rate in learningrates:
    print(rate)
    weights = np.zeros(len(x_train[0]))
    weights, likelihoods = classification_gradientdescent(x_train,y_train,rate,weights,N)
    results.append(likelihoods)
    
for r in results:
    plt.plot(k,r)
    plt.title('Negative Log Likelihood vs iteration number')
    plt.xlabel('Iteration number')
    plt.ylabel('Negative Log Likelihood')
plt.legend(["rate=0.01","rate=0.001","rate=0.0001","rate=0.00001"])
plt.figure()
# from the above it appears that 0.01 is an appropriate learning rate that converges very quickly  

weights, likelihoods = classification_gradientdescent(x_train,y_train,0.01,weights,N)

#predict probabilities
y_predict = sigmoid(np.dot(x_test,weights))
y_predict = y_predict>=0.5

correct = 0
for i in range(len(y_predict)):
    if y_predict[i] == y_test[i]:
        correct+=1
test_accuracy = correct/len(y_predict)

likelihood = negloglikelihood(x_test,y_test,weights)

print("Test accuracy: " +str(test_accuracy))
print("Test Negative Log Likelihood: " + str(likelihood))

###############################################################################
##Stochastic

#intialize weights to 0
weights = np.zeros(len(x_train[0]))
N = 1000
k = [i for i in range(N)]

#compute weights and likelihoods

results = []
learningrates = [0.01,0.001,0.0001,0.00001]
for rate in learningrates:
    print(rate)
    weights = np.zeros(len(x_train[0]))
    weights, likelihoods = stochastic_classification_gradientdescent(x_train,y_train,rate,weights,N)
    results.append(likelihoods)
    
for r in results:
    plt.plot(k,r)
    plt.title('Negative Log Likelihood vs iteration number (stochastic)')
    plt.xlabel('Iteration number')
    plt.ylabel('Negative Log Likelihood')
plt.legend(["rate=0.01","rate=0.001","rate=0.0001","rate=0.00001"])
plt.figure()
# from the above it appears that 0.01 is an appropriate learning rate that converges very quickly  

weights, likelihoods = stochastic_classification_gradientdescent(x_train,y_train,0.01,weights,N)

#predict probabilities
y_predict = sigmoid(np.dot(x_test,weights))
y_predict = y_predict>=0.5

correct = 0
for i in range(len(y_predict)):
    if y_predict[i] == y_test[i]:
        correct+=1
test_accuracy = correct/len(y_predict)

likelihood = negloglikelihood(x_test,y_test,weights)

print("Stochastic Test accuracy: " +str(test_accuracy))
print("Stochastic Test Negative Log Likelihood: " + str(likelihood))