from matplotlib import pyplot
import numpy as np


def plotArray(array):
	x=range(len(array))
	pyplot.plot(x, array)
	pyplot.show()



def plotData():
	import Data
	for index,name in enumerate(Data.getNames()):
		pyplot.subplot(4,2,index+1)
		col = Data.getCol(name)
		x=range(len(col))
		pyplot.plot(x, col, label=name)
		pyplot.ylabel(name)
	pyplot.show()


