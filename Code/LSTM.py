import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers

TINY = 1e-6 # to avoid NaNs in logs
INPUT_SIZE =20 
OUTPUT_SIZE = 20
LSTM_CELL_UNITS = 20
LEARNING_RATE = 0.01
inputs = tf.placeholder(tf.float32, shape = (None, None, INPUT_SIZE)) #(time, batch, in)
outputs = tf.placeholder(tf.float32, shape = (None, None, OUTPUT_SIZE))


cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_CELL_UNITS, state_is_tuple=True)

batch_size = tf.shape(inputs)[1]
initial_state = cell.zero_state(batch_size, tf.float32)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)
final_projection = lambda x: layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)
predicted_outputs = tf.map_fn(final_projection, rnn_outputs)

# compute elementwise cross entropy.
error = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
error = tf.reduce_mean(error)

# optimize
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

# assuming that absolute difference between output and correct answer is 0.5
# or less we can round it to the correct output.
accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))



def generate_batch(num_bits,batch_size):
	x = np.random.random_integers(0,10,20)
	y = np.concatenate((np.zeros(10),np.ones(10)),axis=0)
	x=x.reshape(1,1,20)
	y=y.reshape(1,1,20)
	return x,y




NUM_BITS = 10
ITERATIONS_PER_EPOCH = 100
BATCH_SIZE = 16

valid_x, valid_y = generate_batch(num_bits=NUM_BITS, batch_size=100)

session = tf.Session()
# For some reason it is our job to do this:
session.run(tf.initialize_all_variables())

for epoch in range(1000):
    epoch_error = 0
    for _ in range(ITERATIONS_PER_EPOCH):
        # here train_fn is what triggers backprop. error and accuracy on their
        # own do not trigger the backprop.
        x, y = generate_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)
        epoch_error += session.run([error, train_fn], {
            inputs: x,
            outputs: y,
        })[0]
    epoch_error /= ITERATIONS_PER_EPOCH
    print (epoch_error)
    valid_accuracy = session.run(accuracy, {
        inputs:  valid_x,
        outputs: valid_y,
    })
# print "Epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, valid_accuracy * 100.0)