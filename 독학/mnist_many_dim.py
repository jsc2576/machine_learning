# -*- coding: utf-8 -*-

#learning_rate = 0.005, cost= 2.301656052, Accuracy: 0.1135
#learning_rate = 0.001, cost= 6.574233471, Accuracy: 0.1136

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

dropout_rate = tf.placeholder("float")

W1 = tf.Variable(tf.random_normal([784, 392]))
W2 = tf.Variable(tf.random_normal([392, 196]))
W3 = tf.Variable(tf.random_normal([196, 98]))
W4 = tf.Variable(tf.random_normal([98, 49]))
W5 = tf.Variable(tf.random_normal([49, 25]))
W6 = tf.Variable(tf.random_normal([25, 10]))

B1 = tf.Variable(tf.random_normal([392]))
B2 = tf.Variable(tf.random_normal([196]))
B3 = tf.Variable(tf.random_normal([98]))
B4 = tf.Variable(tf.random_normal([49]))
B5 = tf.Variable(tf.random_normal([25]))
B6 = tf.Variable(tf.random_normal([10]))

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
result = tf.add(tf.matmul(L5, W6), B6)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=Y))
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
            _, c = sess.run([optimizer, cost], feed_dict = {X:batch_xs, Y:batch_ys, dropout_rate:0.9})
            
            avg_cost += c/total_batch

        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        
    correct_prediction = tf.equal(tf.argmax(result, 1), tf.argmax(Y, 1))
        
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels, dropout_rate: 0.9}))
    sess.close()