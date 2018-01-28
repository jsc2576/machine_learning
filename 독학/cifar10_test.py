# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 19:24:26 2018

@author: jsc5565
"""

import tensorflow as tf
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data

tf.reset_default_graph()

dropout_rate = 1.0
learning_rate = 0.0005
save_rate = 0.9
train_epochs = 30
batch_size = 1000
display_step = 1

X = tf.placeholder(tf.float32, shape=[None, 32,32,3])
Y = tf.placeholder(tf.float32, shape=[None, 10])

W1 = tf.get_variable(name="W1", shape=[3,3,3,32], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable(name="W2", shape=[3,3,32,64], initializer=tf.contrib.layers.xavier_initializer())

W_fully = tf.get_variable(name="W_fully", shape=[8*8*64,1024], initializer=tf.contrib.layers.xavier_initializer())
W_result = tf.get_variable(name="W_result", shape=[1024,10], initializer=tf.contrib.layers.xavier_initializer())

B_fully = tf.get_variable(name="B_fully", shape=[1024], initializer=tf.contrib.layers.xavier_initializer())

L1_c = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding="SAME")
L1_r = tf.nn.relu(L1_c)
L1_pool = tf.nn.max_pool(L1_r, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
L1_d = tf.nn.dropout(L1_pool, dropout_rate)

L2_c = tf.nn.conv2d(L1_d, W2, strides=[1,1,1,1], padding="SAME")
L2_r = tf.nn.relu(L2_c)
L2_pool = tf.nn.max_pool(L2_r, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
L2_d = tf.nn.dropout(L2_pool, dropout_rate)

L2_reshape = tf.reshape(L2_d, [-1, 8*8*64])
L_fully = tf.nn.relu(tf.add(tf.matmul(L2_reshape, W_fully), B_fully))

L_result = tf.matmul(L_fully, W_result)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L_result, labels=Y))
optimizer = tf.train.RMSPropOptimizer(learning_rate, save_rate).minimize(cost)

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
     sess.run(tf.global_variables_initializer())
     
     (x_train, y_train), (x_test, y_test) = load_data()
     
     yt_one_hot = tf.one_hot(y_train, 10)
     yt_squeeze = tf.squeeze(yt_one_hot).eval()
     
     #copy
     for epoch in range(train_epochs):
          avg_cost = 0
        
          total_batch = int(x_train.shape[0]/batch_size)
        
          for i in range(total_batch):
              i = i % total_batch
              batch_xs = x_train[(i)*batch_size:((i+1))*batch_size,]
              batch_ys = yt_squeeze[(i)*batch_size:((i+1))*batch_size,]
              #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
              _, c = sess.run([optimizer, cost], feed_dict = {X:batch_xs, Y:batch_ys})
              avg_cost += c/total_batch
     
          if (epoch+1) % display_step == 0:
               print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        
     correct_prediction = tf.equal(tf.argmax(L_result, 1), tf.argmax(Y, 1))
        
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     y_test_squeeze = tf.squeeze(tf.one_hot(y_test, 10)).eval()
     print("Accuracy:", accuracy.eval({X: x_test[:1000,], Y: y_test_squeeze[:1000,]}))
     sess.close()
    