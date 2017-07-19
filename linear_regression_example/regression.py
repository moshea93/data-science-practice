import pandas as pd
import numpy as np
from sklearn import linear_model

pd.set_option('display.max_columns', None)

def read(csvfile):
    return pd.read_csv(csvfile)

def prep(standard, advanced):
    standard = pd.read_csv('fangraphs_standard.csv')
    advanced = pd.read_csv('fangraphs_advanced.csv')
    
    #get rid of duplicate columns
    cols_to_use = advanced.columns.difference(standard.columns)
    advanced = pd.concat([advanced[cols_to_use], advanced['playerid']], axis=1)
    
    combined = pd.merge(standard, advanced, on='playerid')
    return combined

def wOBA_variables(combined):
    #target wOBA formula: 
    # .69*uBB + .72*HBP + .89*1B + 1.27*2B + 1.62*3B + 2.10*HR) /
    # (AB + BB - IBB + SF + HBP)

    X = combined[['G', 'AB', 'PA', 'H', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'BB', 'IBB', 'SO', 'HBP', 'SF', 'SH', 'GDP', 'SB', 'CS']]
    X = combined[['BB', 'HBP', '1B', '2B', '3B', 'HR']]
    #wOBA numerator
    y = combined['wOBA']*(combined['AB']+combined['BB']-combined['IBB']+combined['SF']+combined['HBP'])
    return X, y

def computeCost(X, y, theta):
    squarederror = np.power(X*theta.T - y, 2)
    return np.sum(squarederror) / (2 * len(y))

def gradient_descent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = theta.shape[1]
    cost = np.zeros(iters)

    for i in range(iters):
        error = X*theta.T - y
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - (alpha / len(X)) * np.sum(term)

        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta

def manual_linear_regression(X, y):
    features = X.columns
    X = np.matrix(X)
    y = np.matrix(y)
    #reads in as 1 row instead of 1 column
    y = y.T
    theta = np.matrix(np.zeros(X.shape[1]))
    alpha = .000003
    iters = 100000
    theta = gradient_descent(X, y, theta, alpha, iters)
    print '*** Manual Linear Regression ***'
    for i in range(len(features)):
        print features[i], theta[-1, i]

def sklearn_linear_regression(X, y):
    model = linear_model.LinearRegression()
    model.fit(X, y)
    impact = model.coef_
    print '*** SciKit Learn Linear Regression ***'
    for i in range(len(X.columns)):
        print X.columns[i], impact[i]

if __name__ == '__main__':
    standard = read('fangraphs_standard.csv')
    advanced = read('fangraphs_advanced.csv')
    combined = prep(standard, advanced)
    X, y = wOBA_variables(combined)
    sklearn_linear_regression(X, y)
    manual_linear_regression(X, y)
