# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:48:23 2018

@author: jsc5565
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
tf.reset_default_graph()

mnist = input_data.read_data_sets("mnist", one_hot=True)

learning_rate = 0.001
train_epochs = 20
train_size = 100
display_step = 1
filter_size = 20
img_size = 14

x = tf.placeholder(tf.float32, [None, 28*28])
y = tf.placeholder(tf.float32, [None, 10])

row_w = tf.get_variable(name="row_w", shape=[img_size, filter_size], initializer=tf.ones_initializer())
col_w = tf.get_variable(name="col_w", shape=[img_size, filter_size], initializer=tf.ones_initializer())
filter_w = tf.Variable(tf.random_normal([7,7,1,filter_size], stddev=0.01))
output = tf.get_variable(name="output", shape=[img_size*filter_size, 10], initializer=tf.ones_initializer())
logit = tf.get_variable(name="logit", shape=[img_size*img_size, 10], initializer=tf.ones_initializer())


x_re = tf.reshape(x, [-1, 28, 28, 1])
x_conv = tf.nn.conv2d(x_re, filter_w, strides=[1,1,1,1], padding='SAME')
x_pool = tf.nn.max_pool(x_conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') #?, 14, 14, filter_size

"""first conv2d"""
col_sum = tf.reduce_sum(x_pool, axis=1) * col_w
row_sum = tf.reduce_sum(x_pool, axis=2) * row_w

col_sum_re = tf.reshape(col_sum, [-1, img_size*filter_size])
row_sum_re = tf.reshape(row_sum, [-1, img_size*filter_size])

total_sum = col_sum_re + row_sum_re
relu_add = tf.nn.relu(total_sum)

result = tf.matmul(relu_add, output)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


"""sesson start"""

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    for epoch in range(train_epochs):
        avg_cost = 0
        
        total_epochs = int(mnist.train.num_examples/ train_size)
        
        for i in range(total_epochs):
            batch_xs, batch_ys = mnist.train.next_batch(train_size)
            _,c, row= sess.run([optimizer, cost, row_w], {x: batch_xs, y: batch_ys})
            avg_cost += c/total_epochs
            
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:9f}".format(avg_cost))
            
    prediction = tf.equal(tf.argmax(result, 1), tf.argmax(y, 1))            
            
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
    
    sess.close()
    
    end_time = time.time()
    
    print('time:', end_time-start_time)
