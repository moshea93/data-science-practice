import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn import linear_model

def computeCost(X, y, theta):
	squarederror = np.power(X*theta.T - y, 2)
	return np.sum(squarederror) / (2 * len(y))

def gradientDescent(X, y, theta, alpha, iters):
	temp = np.matrix(np.zeros(theta.shape))
	parameters = int(theta.ravel().shape[1])
	cost = np.zeros(iters)
	
	for i in range(iters):
		error = X*theta.T - y

		for j in range(parameters):
			term = np.multiply(error, X[:,j])
			temp[0,j] = theta[0,j] - (alpha / len(X)) * np.sum(term)

		theta = temp
		cost[i] = computeCost(X, y, theta)

	return theta, cost

def plot_data_and_prediction(data, f):
	x = np.linspace(data.Population.min(), data.Population.max(), 100)
	f = g[0,0] + g[0, 1] * x
	
	fig, ax = plt.subplots(figsize=(12,8))
	ax.plot(x, f, 'r', label='Prediction')
	ax.scatter(data.Population, data.Profit, label='Training Data')
	ax.legend(loc=2)
	ax.set_xlabel('Population')
	ax.set_ylabel('Profit')
	ax.set_title('Predicted Profit vs. Population Size')
	plt.show()	

def plot_cost_by_iteration(cost, iters):
	fig, ax = plt.subplots(figsize=(12,8))
	ax.plot(np.arange(iters), cost, 'r')
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Cost')
	ax.set_title('Error vs Iteration')
	
	plt.show()
	
def single_variable_linear_regression(textfile):
	path = os.getcwd() + textfile
	data = pd.read_csv(path, header=None, names = ['Population', 'Profit'])
	data.insert(0, 'Ones', 1)
	
	X = np.matrix(data.loc[:,['Ones', 'Population']])
	y = np.matrix(data.loc[:,['Profit']])
	theta = np.matrix(np.array([0,0]))
	
	#alpha = .01
	#iters = 1000
	
	#g, cost = gradientDescent(X, y, theta, alpha, iters)
	#print g
	
	#plot_data_and_prediction(data, g)
	#plot_cost_by_iteration(cost, iters)

	model = linear_model.LinearRegression()
	model.fit(X, y)
	x = np.array(X[:, 1].A1)  
	f = model.predict(X).flatten()

	fig, ax = plt.subplots(figsize=(12,8))  
	ax.plot(x, f, 'r', label='Prediction')  
	ax.scatter(data.Population, data.Profit, label='Training Data')  
	ax.legend(loc=2)  
	ax.set_xlabel('Population')  
	ax.set_ylabel('Profit')  
	ax.set_title('Predicted Profit vs. Population Size')
	plt.show()
	
if __name__ == '__main__':
	single_variable_linear_regression('/data/ex1data1.txt')
