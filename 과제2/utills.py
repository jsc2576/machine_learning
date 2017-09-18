# -*- coding: utf-8 -*-
"""
Created on Fri May 26 00:15:19 2017

@author: Minsooyeo
"""
import tensorflow as tf
import numpy as np

def get_weight(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)
    
def get_bias(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

    
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')
    
def MaxPool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    
def _CNNModel(data):
    x = tf.placeholder(tf.float32, shape=[None, data])
    
    weight1 = get_weight([5, 5, 1, 32])
    bias1 = get_bias([32])
    
    x_image = tf.reshape(x, [-1,28,28,1])
    
    relu1 = tf.nn.relu(conv2d(x_image, weight1) + bias1)
    pool1 = MaxPool_2x2(relu1)
    
    weight2 = get_weight([5, 5, 32, 64])
    bias2 = get_bias([64])
    
    relu2 = tf.nn.relu(conv2d(pool1, weight2) + bias2)
    pool2 = MaxPool_2x2(relu2)
    print(pool2)
    print(x)
    return pool2, x
    
def _FlatModel(cnn):
    weight = get_weight([7 * 7 * 64, 256])
    bias = get_bias([256])
    
    cnn_reshape = tf.reshape(cnn, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(cnn_reshape, weight) + bias)
    print(h_fc1)
    return h_fc1
    
def _DropOut(flat):
    
    drop_holder = tf.placeholder(tf.float32)
    drop_val = tf.nn.dropout(flat, drop_holder)    
    print(drop_val)
    print(drop_holder)
    return drop_val, drop_holder
    
def _SoftMax(drop):
    weight = get_weight([256, 10])
    bias = get_bias([10])
    
    softmax_val=tf.nn.softmax(tf.matmul(drop, weight) + bias)    
    print(softmax_val)
    return softmax_val
    
def _SetAccuracy(SoftMaxModel, x):
    y_ = tf.placeholder(tf.float32, shape=[None, x])
    mini_set = -tf.reduce_sum(y_*tf.log(SoftMaxModel))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(mini_set)
    correct_prediction = tf.equal(tf.argmax(SoftMaxModel,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return train_step,accuracy,y_,correct_prediction
    
def Nextbatch(TrainX, TrainY, batch_cnt):
    x = np.random.permutation(55000)[:batch_cnt]
    print(TrainX[x][0].size)
    print(TrainY[x].size)
    return TrainX[x], TrainY[x]