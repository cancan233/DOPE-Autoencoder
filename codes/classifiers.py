import tensorflow as tf
import numpy as np


class vanilla_classifier(tf.keras.Model):
    def __init__(self, input_shape):
        super(vanilla_classifier, self).__init__()
        self.fully_connected = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512, activation="relu", input_shape=input_shape),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(1, activation="softmax"),
            ]
        )

    def call(self, input):
        predict = self.fully_connected(input)
        return predict
