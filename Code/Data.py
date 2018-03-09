import numpy as np

data = []

def load(filename):
	global data
	data = np.genfromtxt(filename,delimiter=',',names=True)

def getCol(name):
	return data[name]

def getRow(row):
	return np.asarray(data[row].tolist())


load('../Data/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv')