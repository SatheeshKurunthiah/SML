import tensorflow as tf
import numpy as np

class ann:

    def __init__(self):
        self.input_size = 168
        self.output_size = 8
        self.learning_rate = 0.05
        self.sess = tf.Session()
        self.input = tf.placeholder(tf.float32, [None, self.input_size])
        self.target = tf.placeholder(tf.float32, [None, self.output_size])
        self.model = self.build_model(self.input)

        self.disp_loss = tf.losses.huber_loss(self.model, self.target)
        self.loss = tf.reduce_mean(self.disp_loss)
        self.reducer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def build_model(self,observation_input):
        layer1_nodes = 48
        layer2_nodes = 32

        weights = tf.random_normal_initializer(0., 0.3)
        biases = tf.constant_initializer(0.1)

        w1 = tf.get_variable('w1', [self.input_size, layer1_nodes], initializer=weights)
        b1 = tf.get_variable('b1', [1, layer1_nodes], initializer=biases)
        l1 = tf.nn.relu(tf.matmul(observation_input, w1) + b1)

        # second hidden layer
        w2 = tf.get_variable('w2', [layer1_nodes, layer2_nodes], initializer=weights)
        b2 = tf.get_variable('b2', [1, layer2_nodes], initializer=biases)
        l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

        # output layer
        w3 = tf.get_variable('w3', [layer2_nodes, self.output_size], initializer=weights)
        b3 = tf.get_variable('b3', [1, self.output_size], initializer=biases)

        return tf.nn.softmax(tf.matmul(l2, w3) + b3)

    def indice(self,val,size):
        arr = [0.0]*size
        for i in range(size):
            if i == val:
                arr[i] = 1.0

        return arr



    def train(self,data,target):
        _, cost = self.sess.run([self.reducer, self.loss],
                                feed_dict={self.input: data,
                                           self.target: target})

        return cost

    def test(self,data,target):
        predictions = self.sess.run(self.model,
                      feed_dict={self.input: data})
        predictions = [np.argmax(i) for i in predictions]
        target = [np.argmax(i) for i in target]
        # return (np.array(target)==np.array(predictions)).sum()
        return (abs(np.array(target)-np.array(predictions))).mean()