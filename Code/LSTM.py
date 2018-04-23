import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
import Formatter

period_sample = Formatter.PeriodSample(10)


TINY = 1e-6 # to avoid NaNs in logs
INPUT_SIZE =4 
OUTPUT_SIZE = 8
LSTM_CELL_UNITS = 20
LEARNING_RATE = 0.005
STOCK_INDICATOR_COUNT = 5 
inputs = tf.placeholder(tf.float32, shape = (None, None, INPUT_SIZE*STOCK_INDICATOR_COUNT)) #(time, batch, in) , multiply by 5 , because of 5 stock indicators
outputs = tf.placeholder(tf.float32, shape = (None, None, OUTPUT_SIZE))


cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_CELL_UNITS, state_is_tuple=True)

batch_size = tf.shape(inputs)[1]
initial_state = cell.zero_state(batch_size, tf.float32)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)
final_projection = lambda x: layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)
# rnn_outputs = tf.Print(rnn_outputs,[rnn_outputs],"rnn_outputs: ")
predicted_outputs = tf.map_fn(final_projection, rnn_outputs)
# predicted_outputs = tf.Print(predicted_outputs,[predicted_outputs],"predicted_outputs : ")
# print("outputs : {}",outputs)
error = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
error = tf.reduce_mean(error)



# optimize
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

# assuming that absolute difference between output and correct answer is 0.5 or less we can round it to the correct output.
accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))



def generate_batch(num_bits,batch_size):
	out = period_sample.getIndicatorData(INPUT_SIZE,3)
	x=out[0]
	bin_num = out[1]
	y = np.zeros(8)
	y[bin_num-1] = 1

	x=x.reshape(1,1,INPUT_SIZE*STOCK_INDICATOR_COUNT)
	y=y.reshape(1,1,OUTPUT_SIZE)
	# print ("Input: {}".format(x))

	return x,y




NUM_BITS = 10
ITERATIONS_PER_EPOCH = 100
BATCH_SIZE = 16

valid_x, valid_y = generate_batch(num_bits=NUM_BITS, batch_size=100)

session = tf.Session()
# For some reason it is our job to do this:
session.run(tf.initialize_all_variables())
errors = []
from matplotlib import pyplot
for epoch in range(100):
    epoch_error = 0
    for _ in range(ITERATIONS_PER_EPOCH):
        # here train_fn is what triggers backprop. error and accuracy on their
        # own do not trigger the backprop.
        x, y = generate_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)
        # print (x)
        # print ("--")
        # print (y)
        epoch_error += session.run([error, train_fn], {
            inputs: x,
            outputs: y,
        })[0]
    epoch_error /= ITERATIONS_PER_EPOCH
    errors.append(epoch_error)
    print (epoch_error)
    valid_accuracy = session.run(accuracy, {
        inputs:  valid_x,
        outputs: valid_y,
    })

pyplot.plot(errors)
pyplot.show()
# print "Epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, valid_accuracy * 100.0)