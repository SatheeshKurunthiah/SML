import numpy as np
import constants as cns
import os

_data = []
_PICKLE_PATH = '../Data/coinbase_pickle.npy'


def load():
    global _data
    # _
    if(os.path.exists(_PICKLE_PATH) and os.path.isfile(_PICKLE_PATH)):
    	_data = np.load(_PICKLE_PATH)
    else:
    	print "Reading from csv..."
    	_data = np.genfromtxt('../Data/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv',delimiter=',',names=True)
    	pickle_data()


def pickle_data():
    print "Pickling _data..."
    np.save(_PICKLE_PATH, _data)
    print "Done"

def getRawData():
    return _data.view(np.float64).reshape(_data.shape + (-1,))


def getCol(name):
	return _data[name]

def getRow(row):
	return np.asarray(_data[row].tolist())

def getNames():
	return _data.dtype.names

def getLength():
	return len(_data)




load()
# load('../Data/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv')
