import os
import pickle

import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, LSTM, Merge
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler

import Formatter

json_path = '../Data/RNN/lstm_model.json'
h5_path = '../Data/RNN/lstm_model.h5'
tb_path = '../Data/RNN/Graph'
volume_scale_path = '../Data/RNN/volume_scale.sav'
change_scale_path = '../Data/RNN/change_scale.sav'
scale = MinMaxScaler(feature_range=(0, 1))
volume_scale = None
change_scale = None
period_class = Formatter.PeriodSample(60)  # 1 hour
train_period = 24 * 7  # 7 days
test_period = 24  # 1 day
bin_count = 8  # no of bins
model = None


def save_model(lstm_model):
    model_json = lstm_model.to_json()
    with open(json_path, 'w') as json_file:
        json_file.write(model_json)

    print("Saved model to disk")


def save_scale(file_name, scaler):
    pickle.dump(scaler, open(file_name, 'wb'))


def load_scale(file_name):
    return pickle.load(open(file_name, 'rb'))


def load_model():
    global volume_scale, change_scale

    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(h5_path)
    print("Loaded model from disk")

    volume_scale = load_scale(volume_scale_path)
    change_scale = load_scale(change_scale_path)

    return loaded_model


def load_data():
    global volume_scale, change_scale
    matrix = period_class.getRnnData(train_period, test_period)

    volume_data = []
    change_data = []
    target = []

    for index in xrange(0, len(matrix), 1):
        volume_data.append(matrix[index][0][:, 0])
        change_data.append(matrix[index][0][:, 1])
        bin_no = np.zeros([bin_count], dtype=float)
        bin_no[matrix[index][1]] = 1.0
        target.append(bin_no)

    volume_data = np.array(volume_data, dtype=float)
    change_data = np.array(change_data, dtype=float)

    volume_scale = scale.fit(volume_data)
    volume_data = volume_scale.transform(volume_data)
    save_scale(volume_scale_path, volume_scale)

    change_scale = scale.fit(change_data)
    change_data = change_scale.transform(change_data)
    save_scale(change_scale_path, change_scale)

    volume_data = volume_data.reshape((len(matrix), 1, train_period))
    change_data = change_data.reshape((len(matrix), 1, train_period))
    target = np.reshape(target, [len(matrix), 1, bin_count])

    return [volume_data, change_data], target


def build_model(data_points, target):
    cost = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    volume = Sequential()
    volume.add(LSTM(train_period, input_shape=(1, train_period), dropout=0.2, return_sequences=True))
    volume.add(LSTM(train_period, return_sequences=True, dropout=0.2))

    change = Sequential()
    change.add(LSTM(train_period, input_shape=(1, train_period), dropout=0.2, return_sequences=True))
    change.add(LSTM(train_period, return_sequences=True, dropout=0.2))

    output = Sequential()
    output.add(Merge([volume, change], mode='mul'))
    output.add(Dense(bin_count, activation='softmax'))

    plot_graph = TensorBoard(log_dir=tb_path, histogram_freq=0, write_graph=True, write_images=True)
    checkpoint = ModelCheckpoint(h5_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint, plot_graph]
    output.compile(loss='categorical_crossentropy', optimizer=cost, metrics=['acc'])
    output.fit(data_points, target, epochs=50, verbose=0, validation_split=0.1, shuffle=True, callbacks=callbacks_list)

    return output


def check_model(iter_count, lstm_model):
    global volume_scale, change_scale
    count = 0
    for index in xrange(0, iter_count, 1):
        test_data = period_class.getChangeVolData(train_period, test_period)

        volume_data = np.reshape(test_data[0][:, 0], (1, train_period))
        volume_data = volume_scale.transform(volume_data)
        volume_data = volume_data.reshape(1, 1, train_period)

        change_data = np.reshape(test_data[0][:, 1], (1, train_period))
        change_data = change_scale.transform(change_data)
        change_data = change_data.reshape(1, 1, train_period)

        prediction = lstm_model.predict([volume_data, change_data])[0][0]
        prediction = np.argmax(prediction)
        actual = test_data[1]
        if prediction == actual:
            count += 1

    print 'Accuracy is : ' + str((count / 150.0))


if not (os.path.isfile(json_path) and os.path.isfile(h5_path)):
    data = load_data()
    model = build_model(data[0], data[1])
    save_model(model)
else:
    model = load_model()

# for i in xrange(0, 100, 1):
#     check_model(150, model)
