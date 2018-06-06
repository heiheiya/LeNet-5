import tensorflow as tf
import numpy as np
import util

class LeNet(object):
    def __init__(self, x, keep_prob, num_classes, batch_size, image_size, channels):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.BATCH_SIZE = batch_size
        self.IMAGE_SIZE = image_size
        self.CHANEELS = channels
        self.create()

    def create(self):
        self.X = tf.reshape(self.X, shape=[-1, self.IMAGE_SIZE, self.IMAGE_SIZE, self.CHANEELS])
        conv1 = util.conv_layer(self.X, ksize=[3, 3, 1, 32], strides=[1, 1, 1, 1], name='conv1')
        pool1 = util.max_pool_layer(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool1')
        conv2 = util.conv_layer(pool1, ksize=[3, 3, 32, 64], strides=[1, 1, 1, 1], name='conv2')
        pool2 = util.max_pool_layer(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool2')
        conv3 = util.conv_layer(pool2, ksize=[3, 3, 64, 128], strides=[1, 1, 1, 1], name='conv3')
        pool3 = util.max_pool_layer(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool3')
        fc4 = util.full_connected_layer(pool3, 625, name='fc4', keep_prob=self.KEEP_PROB)
        fc5 = util.full_connected_layer(fc4, 256, name='fc5', keep_prob=self.KEEP_PROB)
        self.fc6 = util.full_connected_layer(fc5, 10, name='fc6')



