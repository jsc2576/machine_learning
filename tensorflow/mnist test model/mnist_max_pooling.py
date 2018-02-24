# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

tf.reset_default_graph()

start_time = time.time()

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

learning_rate = 0.01
training_epochs = 15
batch_size = 100
display_step = 1

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

dropout_rate = tf.placeholder("float")

W1 = tf.get_variable(name="W1", shape=[3,3,1,32], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable(name="W2", shape=[3,3,32,64], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable(name="W3", shape=[3,3,64,128], initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable(name="W4", shape=[3,3,128,256], initializer=tf.contrib.layers.xavier_initializer())
W5 = tf.get_variable(name="W5", shape=[4*4*128, 625], initializer=tf.contrib.layers.xavier_initializer())
W6 = tf.get_variable(name="W6", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())

B4 = tf.get_variable(name="B4", shape=[625], initializer=tf.contrib.layers.xavier_initializer())

L = tf.reshape(X, [-1, 28, 28, 1])

L1_c = tf.nn.conv2d(L, W1, strides=[1,1,1,1], padding='SAME') # ?, 28, 28, 32
L1_r = tf.nn.relu(L1_c)
L1_d = tf.nn.max_pool(L1_r, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # ?, 14, 14, 32
L1 = tf.nn.dropout(L1_d, dropout_rate)

L2_c = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME') # ?, 14, 14, 64
L2_r = tf.nn.relu(L2_c)
L2_d = tf.nn.max_pool(L2_r, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # ?, 7, 7, 64
L2 = tf.nn.dropout(L2_d, dropout_rate)

L3_c = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME') # ?, 7, 7, 128
L3_r = tf.nn.relu(L3_c)
L3_d = tf.nn.max_pool(L3_r, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # ?, 4, 4, 128
L3 = tf.nn.dropout(L3_d, dropout_rate)

L3_re = tf.reshape(L3, [-1, 4*4*128])
L4 = tf.nn.relu(tf.matmul(L3_re, W5) + B4)
L4_d = tf.nn.dropout(L4, dropout_rate)

L6_result = tf.matmul(L4_d, W6)
L6_d = tf.nn.dropout(L6_result, dropout_rate)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L6_d, labels=Y))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

init = tf.global_variables_initializer()

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0
        
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict = {X:batch_xs, Y:batch_ys, dropout_rate:1.0})
            
            avg_cost += c/total_batch

        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        
    correct_prediction = tf.equal(tf.argmax(L6_d, 1), tf.argmax(Y, 1))
        
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels, dropout_rate: 1.0}))
    sess.close()
    
end_time = time.time()

print("time: ", end_time-start_time)