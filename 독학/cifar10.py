# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 20:05:39 2018

@author: jsc5565
"""

import tensorflow as tf
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data


(x_train, y_train), (x_test, y_test) = load_data()

print(