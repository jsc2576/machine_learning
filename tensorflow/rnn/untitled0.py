# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:38:06 2018

@author: jsc5565
"""

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

tf.reset_default_graph()

char_rdic = ['h', 'e', 'l', 'o']
char_dic = {w: i for i, w in enumerate(char_rdic)}
x_data = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0],
                   [0,0,1,0]],
                    dtype='f')

sample = [char_dic[c] for c in "hello"]

w = tf.get_variable(name="w", shape=[4, 4], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable(name="b", shape=[4], initializer=tf.contrib.layers.xavier_initializer())

char_vocab_size = len(char_dic)
rnn_size = char_vocab_size
time_step_size = 4
batch_size = 1

rnn_cell = rnn_cell.BasicLSTMCell(rnn_size)
state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
X_split = tf.split(x_data, time_step_size)

outputs, states = rnn.static_rnn(rnn_cell, X_split, dtype=tf.float32)
output = tf.reshape(tf.concat(outputs, 1), [-1, rnn_size])
logits = tf.add(tf.matmul(output, w), b)

y = tf.reshape(sample[1:], [-1])

logit = tf.matmul(logits, w)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))
#softmax 함수는 값이 이상하게 나옴 
cost = tf.reduce_mean(tf.contrib.legacy_seq2seq.sequence_loss_by_example([output], [y], [w]))
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #print(y.eval())
    for i in range(200):
        sess.run(train)
        #result = sess.run(logits)
        result = sess.run(tf.argmax(output, 1))
        #print(result)
        #print("step:",i)
        #print(output.eval())
        
    print(output.eval())
    """
    for i in range(len(outputs)):
        print(outputs[i].eval())
    print(w.eval())
    print(mat.shape)
    mat = tf.reshape(mat, [4,-1])
    print(mat.eval())
    print(tf.argmax(mat,1).eval())
    """
    #print(outputs[0].eval())
    print(result, [char_rdic[t] for t in result])
    #print(y.eval())
    