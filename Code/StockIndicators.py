import numpy as np
def calcEMA(data,time_period):

	data = np.asarray(data,float)

	ema=np.zeros(len(data))

	ema[time_period-1]=np.average(data[0:time_period])

	for row in range(time_period,len(data)):
		ema[row] = data[row] * (2/time_period + 1.0) + ema[row-1]*(1-(2/(time_period+1.0)))

	return ema


def calcMACD(closing_price_list, time_period_start, time_period_end):
	assert(time_period_end > time_period_start), "time_period_start must be less than time_period_end"
	closing_price_list = np.asarray(closing_price_list,float)

	ema_start = calcEMA(closing_price_list,time_period_start)
	ema_end = calcEMA(closing_price_list,time_period_end)

	macd = ema_start - ema_end
	macd[0:time_period_end - 1] = 0 # Should this be ?

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
	return K, D


def RSI(data, periods):
    data = data[:, 4]
    rsi = []
    sum_loss = 0
    sum_gain = 0
    idx = 0
    for i in range(0, periods - 1):
        change = data[i + 1] - data[i]
        if change < 0:
            sum_loss -= change
        else:
            sum_gain += change
        rsi.append(0.0)
        idx += 1
    avg_loss = sum_loss / periods
    avg_gain = sum_gain / periods
    rsi.append(100 - (100 / (1 + (avg_gain / avg_loss))))

    while idx < len(data) - 1:
        change = data[idx + 1] - data[idx]
        if change < 0:
            avg_gain = (avg_gain * (periods - 1)) / periods
            avg_loss = ((avg_loss * (periods - 1)) - change) / periods
        else:
            avg_loss = avg_loss * (periods - 1) / periods
            avg_gain = (avg_gain * (periods - 1) + change) / periods

        rsi.append(100 - (100 / (1 + (avg_gain / avg_loss))))
        idx += 1

    return np.array(rsi)


def OBV(data):
    idx = 0
    obv = [0]
    while idx < len(data) - 1:
        change = data[idx + 1, 4] - data[idx, 4]
        if change > 0:
            obv.append(obv[idx] + data[idx + 1, 5])
        elif change < 0:
            obv.append(obv[idx] - data[idx + 1, 5])
        else:
            obv.append(0)
        idx += 1

    return np.array(obv)


