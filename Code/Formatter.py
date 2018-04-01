import Data as data
import StockIndicators as sind
import grouping_changecalc as grp
import constants as cns
import numpy as np


class PeriodSample:
    def __init__(self, period_size):
        self._period_size = period_size
        self._data = data.getRawData()
        self._seed = 1000

    def __getSample(self, train_periods, test_periods):
        grouped_data = grp.group_data(self._data, self._period_size)
        return grp.randomSample(self._seed, grouped_data[self._period_size:], train_periods, test_periods)

    def classifyBins(self, value):
        bin = -1
        if value<0:
            bin = 3
            for i in cns.TARGET_BINS:
                if value<(-1)*i:
                    bin -= 1
        else:
            bin = 4
            for i in cns.TARGET_BINS:
                if value>i:
                    bin += 1

        return bin

    def getChangeVolData(self, train_periods, test_periods):
        train, test = self.__getSample(train_periods, test_periods)
        return train[:,[5,8]], self.classifyBins(grp.total_change(test))

    def getIndicatorData(self, train_periods, test_periods):
        gap_slab = max(cns.KDS_PERIOD, cns.RSI_PERIOD, cns.MACD_PERIOD, cns.MACD_SIGNAL_PERIOD)
        indicators = np.zeros((train_periods+gap_slab,4))
        train, test = self.__getSample(train_periods+gap_slab, test_periods)
        test = grp.total_change(test)
        # indicators[:, 0] = sind.calcEMA(train, cns.KDS_PERIOD)
        # indicators[:, 1] = sind.calcKDS(train, cns.KDS_PERIOD)
        indicators[:, 2] = sind.RSI(train, cns.RSI_PERIOD)
        indicators[:, 3] = sind.OBV(train)

        return indicators[gap_slab:], self.classifyBins(test)
