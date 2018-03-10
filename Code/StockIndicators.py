import numpy as np
def calcEMA(data,time_period):

	data = np.asarray(data,float)

	ema=np.zeros(len(data))

	ema[time_period-1]=np.average(data[0:time_period])

	for row in range(time_period,len(data)):
		ema[row] = data[row] * (2/time_period + 1.0) + ema[row-1]*(1-(2/(time_period+1.0)))

	return ema


def calcMACD(closing_price_list):
	closing_price_list = np.asarray(closing_price_list,float)

	ema12 = calcEMA(closing_price_list,12)
	ema26 = calcEMA(closing_price_list,26)

	macd = ema12 - ema26
	macd[0:25] = 0 # Should this be ?

	return macd

def calcSignalLine(closing_price_list):

	macd = calcMACD(closing_price_list)
	signal = calcEMA(macd,9)

	return signal

def calcKDS(data,lookback_period):
	K = np.zeros(data.getLength())
	D = np.zeros(data.getLength())

	for row in range(lookback_period,data.getLength()):
		lowest_low = np.min(data.getCol('Low')[row-lookback_period:row+1])
		highest_high = np.max(data.getCol('High')[row-lookback_period:row+1])
		current_close = data.getCol('Close')[row]
		K[row]=(current_close - lowest_low) / (highest_high - lowest_low) * 100
		D[row]=np.average(K[row-3:row])
	#TODO: Handle NaN 
	return(K,D)


