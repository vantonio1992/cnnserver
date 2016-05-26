
#from tensorflow.examples.tutorials.mnist import input_data

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import tempfile

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

#standard packages

from PIL import Image

import csv
import numpy as np
import os
import tensorflow as tf
import timeit
import pickle

import datetime


#read input
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#start of implementation
x = tf.placeholder(tf.float32, [None, 784])

#weights, bias incorporation

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#model implementation
y = tf.nn.softmax(tf.matmul(x, W) + b)


#Training start

#cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#learning rate = 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#initialize the variables
init = tf.initialize_all_variables()

#launch Session
sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

 #model evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))