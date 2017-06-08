import matplotlib as plt
import pandas as pd
import numpy as np
import os
from ex1pt1 import gradientDescent, plot_cost_by_iteration
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def normalize_features(feature):
	feature = (feature - feature.mean()) / feature.std()
	return feature

path = os.getcwd() + '/data/ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

data = (data - data.mean()) / data.std()
data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = np.matrix(data.iloc[:,0:cols-1].values)
y = np.matrix(data.iloc[:,cols-1:cols].values)

theta = np.matrix(np.array([0,0,0]))

alpha = .01
iters = 1000

g, cost = gradientDescent(X, y, theta, alpha, iters)
plot_cost_by_iteration(cost, iters)

