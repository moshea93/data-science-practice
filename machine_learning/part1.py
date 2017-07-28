#Single Variable Linear Regression
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_cost(X, y, theta):
    hypotheses = (X*theta).sum(axis=1)
    squared_errors = hypotheses.sub(y, axis=0)**2
    cost = (float(1)/(2*y.shape[0]))*squared_errors.sum(axis=0)
    return cost

def get_partial_derivatives(X, y, theta):
    partials = [0]*len(theta)

    hypotheses = (X*theta).sum(axis=1)
    errors = hypotheses.sub(y, axis=0)
    coef = float(1)/y.shape[0]

    partials[0] = coef*errors.sum(axis=0)
    for i in range(1, len(theta)):
        partials[i] = coef*((errors*X.iloc[:,i]).sum())
    
    return partials

def gradient_descent(X, y, theta, alpha, iters):
    cost = [0]*iters
    
    for i in range(iters):
        cost[i] = get_cost(X, y, theta)
        partials = get_partial_derivatives(X, y, theta)
        for i in range(len(theta)):
            theta[i] = theta[i] - alpha*partials[i]
    
    return theta, cost

def graph_data(data, theta):
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = theta[0] + (theta[1] * x)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x, f, 'y', label='Prediction')
    ax.scatter(data['Population'], data['Profit'], label='Training Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    
    plt.show()

def graph_cost(iters, cost):
    fig, ax = plt.subplots(figsize=(12,8))
    #to get axes to scale better, not plotting large cost of first 10
    ax.plot([i for i in range(10,iters)], cost[10:])
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()

def myimplementation():
    data = pd.read_csv('data/ex1data1.txt', header=None, names=['Population', 'Profit'])
    
    #add theta_0 term
    data.insert(0, 'Ones', 1)
    #initialize theta
    theta = [0]*(data.shape[1] - 1)
    
    X = data.iloc[:,0:-1]
    y = data.iloc[:,-1]
    
    alpha = .02
    iters = 1000
    
    theta, cost = gradient_descent(X, y, theta, alpha, iters)
    #graph_data(data, theta)
    graph_cost(iters, cost)

if __name__ == '__main__':
    myimplementation()
