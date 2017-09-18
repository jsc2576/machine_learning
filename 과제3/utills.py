# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:42:28 2017

@author: Minsooyeo
"""
import os
import tensorflow as tf
import numpy as np

def search(dirname):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        return filenames
        


def get_weight(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)
    
def get_bias(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

    
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='VALID')
    
def MaxPool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    
def _CNNModel(data):
    x = tf.placeholder(tf.float32, shape=[None, data])
    
    weight1 = get_weight([5, 5, 1, 256])
    bias1 = get_bias([256])
    
    x_image = tf.reshape(x, [-1,72,40,1])
    
    relu1 = tf.nn.relu(conv2d(x_image, weight1) + bias1)
    pool1 = MaxPool_2x2(relu1)
    
    return pool1, x
    
def _FlatModel(cnn):
    weight = get_weight([34*18*256, 64])
    bias = get_bias([64])
    
    cnn_reshape = tf.reshape(cnn, [-1, 34*18*256])
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
    weight = get_weight([64, 5])
    bias = get_bias([5])
    
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
    x = np.random.permutation(2000)[:batch_cnt]
    return TrainX[x], TrainY[x]