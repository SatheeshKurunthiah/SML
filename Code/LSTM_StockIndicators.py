import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, LSTM, Merge
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop
import keras



bin_count = 8  # no of bins
BATCH_SIZE = 1000
import Formatter

period_sample = Formatter.PeriodSample(60)
INPUT_SIZE = 24



def createModel(train_period,target):
    cost = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    EMA_lstm = Sequential()
    EMA_lstm.add(LSTM(INPUT_SIZE, input_shape=(INPUT_SIZE,1),  batch_input_shape=(BATCH_SIZE,INPUT_SIZE,1), dropout=0.2, return_sequences=False))

    K_lstm = Sequential()
    K_lstm.add(LSTM(INPUT_SIZE, input_shape=(INPUT_SIZE,1),batch_input_shape=(BATCH_SIZE,INPUT_SIZE,1), dropout=0.2, return_sequences=False))

    D_lstm = Sequential()
    D_lstm.add(LSTM(INPUT_SIZE, input_shape=(INPUT_SIZE,1),batch_input_shape=(BATCH_SIZE,INPUT_SIZE,1), dropout=0.2, return_sequences=False))

    RSI_lstm = Sequential()
    RSI_lstm.add(LSTM(INPUT_SIZE, input_shape=(INPUT_SIZE,1),batch_input_shape=(BATCH_SIZE,INPUT_SIZE,1), dropout=0.2, return_sequences=False))


    OBV_lstm = Sequential()
    OBV_lstm.add(LSTM(INPUT_SIZE, input_shape=(INPUT_SIZE,1),batch_input_shape=(BATCH_SIZE,INPUT_SIZE,1), dropout=0.2, return_sequences=False))

    network_model = Sequential()
    network_model.add(Merge([EMA_lstm, K_lstm,D_lstm,RSI_lstm,OBV_lstm], mode='mul'))
    network_model.add(Dense(bin_count))

    network_model.compile(loss='categorical_crossentropy', optimizer=cost, metrics=['acc'])
    train_period,target = getBatch()
    network_model.fit(train_period, target,batch_size=BATCH_SIZE, epochs=5, verbose=0, validation_split=0.1, shuffle=True, callbacks=None)
    # network_model.fit_generator(generator=gen(),steps_per_epoch=1000)
def getBatch():
    out = []
    for i in range(BATCH_SIZE):
        data = period_sample.getIndicatorData(INPUT_SIZE, 1)
        features = data[0]
        features = np.transpose(features)
        # output_bin = np.zeros(bin_count)
        # output_bin[data[1]]=1
        output_bin = keras.utils.to_categorical(data[1], num_classes=bin_count)
        out.append([[np.asarray(features[0]).reshape(1,24,1),
               np.asarray(features[1]).reshape(1,24,1),
               np.asarray(features[2]).reshape(1,24,1),
               np.asarray(features[3]).reshape(1,24,1),
               np.asarray(features[4]).reshape(1,24,1)],np.asarray(output_bin).reshape(1,8)])
    out = np.asarray(out)
    return out[:,0],out[:,1]


def main():
    # nn_input = []
    # for i in range(BATCH_SIZE):
    #     data = period_sample.getIndicatorData(INPUT_SIZE, 1)
    #     features = data[0]
    #     features = np.transpose(features)
    #     output_bin = data[1]
    #     ary = [features,output_bin]
    #     nn_input.append(ary)
    # nn_input = np.asarray(nn_input)

    createModel(None,None)


if __name__ == '__main__':
    main()
