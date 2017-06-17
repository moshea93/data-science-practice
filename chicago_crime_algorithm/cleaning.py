import pandas as pd
import os
import numpy as np
from sklearn import linear_model

def clean(csvfile):
    data = pd.read_csv(csvfile)
    data = data.iloc[:,:9]
    age_mappings = {'less than 20': 0,
    	            '20-30': 1,
    		    '30-40': 2,
		    '40-50': 3,
		    '50-60': 4,
		    '60-70': 5,
		    '70-80': 6}
    data = data[data['PREDICTOR RAT AGE AT LATEST ARREST'].notnull()]
    data['PREDICTOR RAT AGE AT LATEST ARREST'] = data['PREDICTOR RAT AGE AT LATEST ARREST'].apply(lambda x: age_mappings[x])
    return data

def linear_regression(data):
    data.insert(1, 'Ones', 1)
    X = np.matrix(data.iloc[:,1:])
    y = data.iloc[:,:1]
    reg = linear_model.LinearRegression()
    reg.fit(X, y)
    
    characteristics = [' '.join(x.split(' ')[2:]) for x in list(data)[1:]]
    impact = reg.coef_[0]
    for i in range(len(characteristics)):
        print characteristics[i], impact[i]

if __name__ == '__main__':
    data = clean('Strategic_Subject_list.csv')
    linear_regression(data)
    #clean('wip.csv')
