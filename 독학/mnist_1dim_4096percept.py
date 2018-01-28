# -*- coding: utf-8 -*-

#weight가 2개, hidden_layer perceptron 수 4096개, learning_rate = 0.01 : cost= 0.038012832, Accuracy: 0.9739


#weight가 2개, hidden_layer perceptron 수 4096개, weight에 xavier_initializer 적용, learning_rate = 0.001 : cost= 0.013102306, Accuracy: 0.9791
#weight가 2개, hidden_layer perceptron 수 4096개, weight에 xavier_initializer 적용, learning_rate = 0.005 : cost= 0.030782673, Accuracy: 0.9747
#weight가 2개, hidden_layer perceptron 수 4096개, weight에 xavier_initializer 적용, learning_rate = 0.01 : cost= 0.045931397, Accuracy: 0.9704

#weight가 2개, hidden_layer perceptron 수 4096개, weight에 xavier_initializer 적용, dropout_rate 0.7, learning_rate = 0.001 : cost= 0.013506903, Accuracy: 0.9814, 0.975, 0.978
#weight가 2개, hidden_layer perceptron 수 4096개, weight에 xavier_initializer 적용, dropout_rate 0.7, learning_rate = 0.005 : cost= 0.061126313, Accuracy: 0.9733
#weight가 2개, hidden_layer perceptron 수 4096개, weight에 xavier_initializer 적용, dropout_rate 0.7, learning_rate = 0.01 : cost= 0.116265653, Accuracy: 0.9542

#weight가 2개, hidden_layer perceptron 수 4096개, weight에 xavier_initializer 적용, dropout_rate 0.6, learning_rate = 0.001 : cost= 0.020389751, Accuracy: 0.9793
#weight가 2개, hidden_layer perceptron 수 4096개, weight에 xavier_initializer 적용, dropout_rate 0.8, learning_rate = 0.001 : cost= 0.009965175, Accuracy: 0.9784, 0.9791

#weight가 2개, hidden_layer perceptron 수 4096개, weight, bias에 xavier_initializer 적용, dropout_rate 0.7, learning_rate = 0.001 : cost= 0.014963645, Accuracy: 0.9796
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

start_time = time.time()

tf.reset_default_graph() #xavier_initializer reuse 문제 해결을 위한 코드 

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

dropout_rate = tf.placeholder("float")  

#가중치 랜덤 적용
#W1 = tf.Variable(tf.random_normal([784, 4096]))
#W2 = tf.Variable(tf.random_normal([4096, 10]))

W1 = tf.get_variable(name="W1", shape=[784, 4096], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable(name="W2", shape=[4096, 10], initializer=tf.contrib.layers.xavier_initializer())

#B1 = tf.Variable(tf.random_normal([4096]))
#B2 = tf.Variable(tf.random_normal([10]))

B1 = tf.get_variable(name="B1", shape=[4096], initializer=tf.contrib.layers.xavier_initializer())
B2 = tf.get_variable(name="B2", shape=[10], initializer=tf.contrib.layers.xavier_initializer())

_L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L1 = tf.nn.dropout(_L1, dropout_rate)
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
            _, c = sess.run([optimizer, cost], feed_dict = {X:batch_xs, Y:batch_ys, dropout_rate:0.7})
            
            avg_cost += c/total_batch

        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        
    correct_prediction = tf.equal(tf.argmax(result, 1), tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels, dropout_rate:0.7}))
    sess.close()
    
end_time = time.time()

print("time: ", end_time-start_time)