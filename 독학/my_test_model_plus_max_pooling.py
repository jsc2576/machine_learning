# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:46:53 2018

@author: jsc5565
"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

mnist = input_data.read_data_sets("mnist", one_hot=True)

learning_rate = 0.01
train_epochs = 15
train_size = 200
display_step = 1

x = tf.placeholder(tf.float32, [None, 28*28])
y = tf.placeholder(tf.float32, [None, 10])

""" original """
row_w = tf.get_variable(name="row_w", shape=[28], initializer=tf.ones_initializer())
col_w = tf.get_variable(name="col_w", shape=[28], initializer=tf.ones_initializer())
sum_row_w = tf.get_variable(name="sum_row_w", shape=[49, 10], initializer=tf.ones_initializer())
sum_col_w = tf.get_variable(name="sum_col_w", shape=[49, 10], initializer=tf.ones_initializer())
logit = tf.get_variable(name="logit", shape=[49*2, 10], initializer=tf.ones_initializer())
bias = tf.get_variable(name="b", shape=[28], initializer=tf.ones_initializer())

x_re = tf.reshape(x, [-1, 28, 28])

#mul1 = tf.multiply(x_re, row_w)
mul1 = x_re * row_w
transpose = tf.transpose(mul1, perm=[0, 2, 1])
mul2 = transpose*col_w


add = tf.add(mul2, bias)

#add_re = tf.reshape(add, [-1, 14])

size = tf.shape(add)
one = tf.ones([size[0], 28, 1])
col_sum = tf.matmul(add, one)


re_transpose = tf.transpose(add, perm=[0,2,1])
#tran_re = tf.reshape(re_transpose, [-1, 14])
row_sum = tf.matmul(re_transpose, one)

""" max pooling """

pool_row_w = tf.get_variable(name="pool_row_w", shape=[14], initializer=tf.ones_initializer())
pool_col_w = tf.get_variable(name="pool_col_w", shape=[14], initializer=tf.ones_initializer())
pool_bias = tf.get_variable(name="pool_b", shape=[14], initializer=tf.ones_initializer())

pool_x_re_temp = tf.reshape(x, [-1, 28, 28, 1])
pool_x_pool = tf.nn.max_pool(pool_x_re_temp, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
pool_x_re = tf.reshape(pool_x_pool, [-1, 14, 14])

#mul1 = tf.multiply(x_re, row_w)
pool_mul1 = pool_x_re * pool_row_w
pool_transpose = tf.transpose(pool_mul1, perm=[0, 2, 1])
pool_mul2 = pool_transpose*pool_col_w


pool_add = tf.add(pool_mul2, pool_bias)

#add_re = tf.reshape(add, [-1, 14])

pool_size = tf.shape(pool_add)
pool_one = tf.ones([pool_size[0], 14, 1])
pool_col_sum = tf.matmul(pool_add, pool_one)


pool_re_transpose = tf.transpose(pool_add, perm=[0,2,1])
#tran_re = tf.reshape(re_transpose, [-1, 14])
pool_row_sum = tf.matmul(pool_re_transpose, pool_one)

""" max pooling2 """

pool2_row_w = tf.get_variable(name="pool2_row_w", shape=[7], initializer=tf.ones_initializer())
pool2_col_w = tf.get_variable(name="pool2_col_w", shape=[7], initializer=tf.ones_initializer())
pool2_bias = tf.get_variable(name="pool2_b", shape=[7], initializer=tf.ones_initializer())

pool2_x_pool = tf.nn.max_pool(pool_x_pool, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
pool2_x_re = tf.reshape(pool2_x_pool, [-1, 7, 7])

#mul1 = tf.multiply(x_re, row_w)
pool2_mul1 = pool2_x_re * pool2_row_w
pool2_transpose = tf.transpose(pool2_mul1, perm=[0, 2, 1])
pool2_mul2 = pool2_transpose*pool2_col_w


pool2_add = tf.add(pool2_mul2, pool2_bias)

#add_re = tf.reshape(add, [-1, 14])

pool2_size = tf.shape(pool2_add)
pool2_one = tf.ones([pool2_size[0], 7, 1])
pool2_col_sum = tf.matmul(pool2_add, pool2_one)


pool2_re_transpose = tf.transpose(pool2_add, perm=[0,2,1])
#tran_re = tf.reshape(re_transpose, [-1, 14])
pool2_row_sum = tf.matmul(pool2_re_transpose, pool2_one)

""" end"""

col_sum_re = tf.concat([tf.squeeze(col_sum), tf.squeeze(pool_col_sum), tf.squeeze(pool2_col_sum)], axis=1)
row_sum_re = tf.concat([tf.squeeze(row_sum), tf.squeeze(pool_row_sum), tf.squeeze(pool2_row_sum)], axis=1)

concat = tf.concat([col_sum_re, row_sum_re], axis=1)
"""
col_relu = tf.nn.relu(col_sum_re)
row_relu = tf.nn.relu(row_sum_re)

col_mat = tf.matmul(col_relu, sum_col_w)
row_mat = tf.matmul(row_relu, sum_row_w)
"""

mat = tf.matmul(concat, logit)
result = tf.nn.relu(mat)

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
            _, c= sess.run([optimizer, cost], {x: batch_xs, y: batch_ys})
            
            avg_cost += c/total_epochs
            
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:9f}".format(avg_cost))
            
    prediction = tf.equal(tf.argmax(result, 1), tf.argmax(y, 1))            
            
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
    
    sess.close()