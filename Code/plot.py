from matplotlib import pyplot
import numpy as np

data = []

with open('../Data/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv', 'r') as fp:
    for line in fp:
        l = line.rstrip('\r\n')
        data.append(l.split(','))

data_array = np.array(data)
data_array = data_array[1:]
data_array = data_array.astype(float)

data = []
with open('../Data/Processed.csv', 'r') as fp:
    for line in fp:
        l = line.rstrip('\r\n')
        data.append(l.split(','))

data_array2 = np.array(data)
data_array2 = data_array2[1:]
data_array2 = data_array2.astype(float)

x = range(1, len(data_array) + 1)
x2 = range(1, len(data_array2) + 1)

pyplot.plot(x, data_array[:, 4], label='original')
pyplot.plot(x2, data_array2[:, 4], label='processed')
pyplot.legend()
pyplot.show()
