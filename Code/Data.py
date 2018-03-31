import numpy as np

data = []
import random

def load(filename):
	global data
	data = np.genfromtxt(filename,delimiter=',',names=True)

def getCol(name):
	return data[name]

def getRow(row):
	return np.asarray(data[row].tolist())

def getNames():
	return data.dtype.names

def getLength():
	return len(data)


def randomSample(count):
	endIndex = random.randint(count, data.shape[0]) 
	beginIndex = endIndex - count
	rawChunk=data[beginIndex:endIndex]
	chunk=[]
	for index, row in enumerate(rawChunk):
		chunk.append(row.tolist())     
	return chunk


load('../Data/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv')
