import Data
import Formatter
import grouping_changecalc as grp
print(Data.getNames())

formatter = Formatter.PeriodSample(10)

formatter.getChangeVolData(50,5)
#
# # train, test = Data.randomSample(10,2)
# print(Data.getNames())
