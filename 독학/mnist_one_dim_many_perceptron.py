# -*- coding: utf-8 -*-

#learning rate = 0.01 일 때 cost= 0.605163571, accuracy : 0.9732

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

learning_rate = 0.02
training_epochs = 15
batch_size = 100
display_step = 1

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

dropout_rate = tf.placeholder("float")

W1 = tf.Variable(tf.random_normal([784, 2048]))
W2 = tf.Variable(tf.random_normal([2048, 10]))

B1 = tf.Variable(tf.random_normal([2048]))
B2 = tf.Variable(tf.random_normal([10]))

L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
result = tf.add(tf.matmul(L1, W2), B2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0
        
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict = {X:batch_xs, Y:batch_ys})
            
            avg_cost += c/total_batch

        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        
    correct_prediction = tf.equal(tf.argmax(result, 1), tf.argmax(Y, 1))
        
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    sess.close()