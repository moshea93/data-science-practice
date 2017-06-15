import pandas as pd
import os

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
	data = (data - data.mean()) / data.std()
	print data.head()

if __name__ == '__main__':
	clean('Strategic_Subject_list.csv')
	#clean('wip.csv')
