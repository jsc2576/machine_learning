# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

tf.reset_default_graph()

start_time = time.time()

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

learning_rate = 0.001
training_epochs = 30
batch_size = 100
display_step = 1

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

dropout_rate = tf.placeholder("float")

W1 = tf.get_variable(name="W1", shape=[3,3,1,32], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable(name="W2", shape=[3,3,32,64], initializer=tf.contrib.layers.xavier_initializer())

L = tf.reshape(X, [-1, 28, 28, 1])

L1_c = tf.nn.conv2d(L, W1, strides=[1,1,1,1], padding='SAME')
L1_r = tf.nn.relu(L1_c)
L1 = tf.nn.max_pool(L1_r, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

L2_c = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2_r = tf.nn.relu(L2_c)
#L2 = tf.nn.max_pool(L2_r, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=L2_r, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0
        
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict = {X:batch_xs, Y:batch_ys, dropout_rate:0.6})
            
            avg_cost += c/total_batch

        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        
    correct_prediction = tf.equal(tf.argmax(L2_r, 1), tf.argmax(Y, 1))
        
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels, dropout_rate: 0.6}))
    sess.close()
    
end_time = time.time()

print("time: ", end_time-start_time)