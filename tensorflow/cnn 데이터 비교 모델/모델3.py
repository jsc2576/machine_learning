# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

tf.reset_default_graph()

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

learning_rate = 0.0005
training_epochs = 30
batch_size = 100
display_step = 1

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

dropout_rate = tf.placeholder("float")

W1 = tf.get_variable(name="W1", shape=[784, 1024], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable(name="W2", shape=[1024, 512], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable(name="W3", shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable(name="W4", shape=[256, 128], initializer=tf.contrib.layers.xavier_initializer())
W5 = tf.get_variable(name="W5", shape=[128, 64], initializer=tf.contrib.layers.xavier_initializer())
W6 = tf.get_variable(name="W6", shape=[64, 32], initializer=tf.contrib.layers.xavier_initializer())
W7 = tf.get_variable(name="W7", shape=[1056, 10], initializer=tf.contrib.layers.xavier_initializer())

B1 = tf.get_variable(name="B1", shape=[1024], initializer=tf.contrib.layers.xavier_initializer())
B2 = tf.get_variable(name="B2", shape=[512], initializer=tf.contrib.layers.xavier_initializer())
B3 = tf.get_variable(name="B3", shape=[256], initializer=tf.contrib.layers.xavier_initializer())
B4 = tf.get_variable(name="B4", shape=[128], initializer=tf.contrib.layers.xavier_initializer())
B5 = tf.get_variable(name="B5", shape=[64], initializer=tf.contrib.layers.xavier_initializer())
B6 = tf.get_variable(name="B6", shape=[32], initializer=tf.contrib.layers.xavier_initializer())
B7 = tf.get_variable(name="B7", shape=[10], initializer=tf.contrib.layers.xavier_initializer())

_L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L1 = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
L2 = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), B3))
L3 = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), B4))
L4 = tf.nn.dropout(_L4, dropout_rate)
_L5 = tf.nn.relu(tf.add(tf.matmul(L4, W5), B5))
L5 = tf.nn.dropout(_L5, dropout_rate)
_L6 = tf.nn.relu(tf.add(tf.matmul(L5, W6), B6))
L6 = tf.nn.dropout(_L6, dropout_rate)
result = tf.add(tf.matmul(tf.concat([L6, L1], 1), W7), B7)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    start_time = time.time()

   for epoch in range(training_epochs):
        avg_cost = 0
        
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict = {X:batch_xs, Y:batch_ys, dropout_rate:0.6})
            print(batch_ys)
            avg_cost += c/total_batch

        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        
    correct_prediction = tf.equal(tf.argmax(result, 1), tf.argmax(Y, 1))
        
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels, dropout_rate: 0.6}))
    sess.close()
    
    end_time = time.time()

    print("time: ", end_time-start_time)
