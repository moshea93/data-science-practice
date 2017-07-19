import pandas as pd
from sklearn import linear_model

pd.set_option('display.max_columns', None)

def gradient_descent(X, y, theta):
    return

def linear_regression(X, y):
    theta = [0]*X.shape[1]
    print theta

standard = pd.read_csv('fangraphs_standard.csv')
advanced = pd.read_csv('fangraphs_advanced.csv')

#get rid of duplicate columns
cols_to_use = advanced.columns.difference(standard.columns)
advanced = pd.concat([advanced[cols_to_use], advanced['playerid']], axis=1)

combined = pd.merge(standard, advanced, on='playerid')

#target wOBA formula: 
# .69*uBB + .72*HBP + .89*1B + 1.27*2B + 1.62*3B + 2.10*HR) /
# (AB + BB - IBB + SF + HBP)

X = combined[['G', 'AB', 'PA', 'H', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'BB', 'IBB', 'SO', 'HBP', 'SF', 'SH', 'GDP', 'SB', 'CS']]
#wOBA numerator
y = combined['wOBA']*(combined['AB']+combined['BB']-combined['IBB']+combined['SF']+combined['HBP'])

model = linear_model.LinearRegression()
model.fit(X, y)
impact = model.coef_
#print X.columns
for i in range(len(X.columns)):
    print X.columns[i], impact[i]

#linear_regression(X, y)

#print combined.head()
