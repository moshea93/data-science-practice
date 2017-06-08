import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import os
import scipy.optimize as opt

def plot_admitted(positive, negative):
	fig, ax = plt.subplots(figsize=(12,8))  
	ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')  
	ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
	ax.legend()
	ax.set_xlabel('Exam 1 Score')  
	ax.set_ylabel('Exam 2 Score')
	plt.show()
	
def plot_sigmoid_output():
	nums = np.arange(-10, 10, step=.2)
	fig, ax = plt.subplots(figsize = (12,8))
	ax.plot(nums, sigmoid(nums), 'r')
	plt.show()

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)
	
	predictions = sigmoid(X * theta.T)
	first = np.multiply(-y, np.log(predictions))
	second = np.multiply(-(1 - y), np.log(1 - predictions))
	return float(1) / len(X) * np.sum(first + second)

def gradient(theta, X, y):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)

	parameters = int(theta.ravel().shape[1])
	grad = np.zeros(parameters)

	error = sigmoid(X * theta.T) - y

	for i in range(parameters):
		term = np.multiply(error, X[:,i])
		grad[i] = np.sum(term) / len(X)
	
	return grad

def predict(theta, X):
	probability = sigmoid(X * theta.T)
	return [1 if x >= .5 else 0 for x in probability]

path = os.getcwd() + '/data/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

positive = data[data['Admitted'] == 1]
negative = data[data['Admitted'] == 0]

#plot_admitted(positive, negative)
#plot_sigmoid_output()

data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = np.matrix(data.iloc[:,0:cols-1].values)
y = np.matrix(data.iloc[:,cols-1:cols].values)
theta = np.matrix(np.zeros(3))

result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b== 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(correct) % len (correct))
print accuracy
print 'accuracy = {0}%'.format(accuracy)
