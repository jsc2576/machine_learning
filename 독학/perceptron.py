# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 20:41:23 2018

@author: jsc5565
"""

import tensorflow as tf

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [1, 10])
y = tf.placeholder(tf.float32, [1])
w = tf.Variable(tf.ones([10,1]))
b = tf.Variable(tf.ones([1]))

L = tf.nn.relu(tf.add(tf.matmul(x,w), b))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L, labels=y))
cost = tf.train.AdamOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    
    sess.run(init)
    
    print(sess.run(L, feed_dict = {x:[[1,1,1,1,1,1,1,1,1,1]], y:[1]}))
    print(sess.run(L, feed_dict = {x:[[1,1,1,1,1,1,1,1,1,0]], y:[0]}))
    sess.run(cost, feed_dict = {x:[[1,1,1,1,1,1,1,1,1,1]], y:[1]})
    print(sess.run(L, feed_dict = {x:[[1,1,1,1,1,1,1,1,1,1]], y:[0]}))
    print(sess.run(loss, feed_dict = {x:[[1,1,1,1,1,1,1,1,1,1]], y:[0]}))