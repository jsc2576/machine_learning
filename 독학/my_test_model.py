# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:09:46 2018

@author: jsc5565
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

mnist = input_data.read_data_sets("mnist", one_hot=True)

learning_rate = 0.0005
train_epochs = 20
train_size = 100
display_step = 1

x = tf.placeholder(tf.float32, [None, 28*28])
y = tf.placeholder(tf.float32, [None, 10])

row_w = tf.get_variable(name="row_w", shape=[28], initializer=tf.ones_initializer())
col_w = tf.get_variable(name="col_w", shape=[28], initializer=tf.ones_initializer())
sum_row_w = tf.get_variable(name="sum_row_w", shape=[28, 10], initializer=tf.ones_initializer())
sum_col_w = tf.get_variable(name="sum_col_w", shape=[28, 10], initializer=tf.ones_initializer())
logit = tf.get_variable(name="logit", shape=[28*28, 10], initializer=tf.ones_initializer())
bias = tf.get_variable(name="b", shape=[28], initializer=tf.ones_initializer())

x_re = tf.reshape(x, [-1, 28, 28])

#mul1 = tf.multiply(x_re, row_w)
mul1 = x_re * row_w
transpose = tf.transpose(mul1, perm=[0, 2, 1])
mul2 = tf.multiply(transpose, col_w)


add = tf.add(mul2, bias)
add_re = tf.reshape(add, [-1, 28*28])

size = tf.shape(add_re)
one = tf.ones([size[0], 28, 1])
col_sum = tf.matmul(add, one)


re_transpose = tf.transpose(add, perm=[0,2,1])
row_sum = tf.matmul(re_transpose, one)

col_sum_re = tf.reshape(col_sum, [-1, 28])
row_sum_re = tf.reshape(row_sum, [-1, 28])

col_relu = tf.nn.relu(col_sum_re)
row_relu = tf.nn.relu(row_sum_re)

col_mat = tf.matmul(col_relu, sum_col_w)
row_mat = tf.matmul(row_relu, sum_row_w)

result = tf.add(col_mat, row_mat)

#result = tf.matmul(add_re, logit)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(train_epochs):
        avg_cost = 0
        
        total_epochs = int(mnist.train.num_examples/ train_size)
        
        for i in range(total_epochs):
            batch_xs, batch_ys = mnist.train.next_batch(train_size)
            _,c,row, col= sess.run([optimizer, cost, col_sum, row_sum], {x: batch_xs, y: batch_ys})
            
            avg_cost += c/total_epochs
            
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:9f}".format(avg_cost))
            print("row:", row.shape, ", col:", col.shape)
            
    prediction = tf.equal(tf.argmax(result, 1), tf.argmax(y, 1))            
            
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
    
    sess.close()