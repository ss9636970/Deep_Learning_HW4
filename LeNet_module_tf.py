import tensorflow as tf
from tensorflow import keras
import numpy as np

class LeNet_module(keras.Model):
    def __init__(self):
        super(LeNet_module, self).__init__()
        self.maxpool = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv1 = keras.layers.Conv2D(6, 5, activation='relu', dilation_rate=1)
        self.conv2 = keras.layers.Conv2D(16, 5, activation='relu', dilation_rate=1)
        self.conv3 = keras.layers.Conv2D(26, 5, activation='relu', dilation_rate=1)
        self.conv4 = keras.layers.Conv2D(36, 5, activation='relu', dilation_rate=1)
        self.conv5 = keras.layers.Conv2D(46, 5, activation='relu', dilation_rate=1)

        self.linear1 = keras.layers.Dense(100, activation='relu')
        self.linear2 = keras.layers.Dense(50, activation='softmax')

    def call(self, inputs):
        n = inputs.shape[0]
        h1 = self.conv1(inputs)
        h1 = self.maxpool(h1)
        h2 = self.conv2(h1)
        h2 = self.maxpool(h2)
        h3 = self.conv3(h2)
        h3 = self.maxpool(h3)
        h4 = self.conv4(h3)
        h4 = self.maxpool(h4)
        h5 = self.conv5(h4)
        h5 = self.maxpool(h5)
        h5 = tf.reshape(h5, shape=(n, -1))
        h6 = self.linear1(h5)
        outputs = self.linear2(h6)
        return outputs