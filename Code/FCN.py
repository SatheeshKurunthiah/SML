import Formatter
import ANN as nn
import numpy as np

train_period = 24*7  # 7 days
test_period = 24  # 1 day
bin_count = 8

period_class = Formatter.PeriodSample(60)
target = []
change_data = []
matrix = [period_class.getChangeVolData(train_period, test_period) for i in range(900000)]

for index in range(0, len(matrix), 1):
    change_data.append(matrix[index][0][:, 1])
    bin_no = np.zeros([bin_count], dtype=float)
    bin_no[matrix[index][1]] = 1.0
    target.append(bin_no)

change_data = np.array(change_data, dtype=float)


model = nn.ann()
cost = model.train(change_data,target)
# plt.plot(cost)
print(cost)
# plt.show()
# print(cost)
print((model.test(change_data,target)))