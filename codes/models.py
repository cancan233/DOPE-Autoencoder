import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.nn import (
    relu,
    sigmoid,
)


class vanilla_encoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim):
        super(vanilla_encoder, self).__init__()
        self.hidden_layer = Dense(
            units=intermediate_dim, activation=relu, kernel_initializer="he_uniform"
        )
        self.output_layer = Dense(units=intermediate_dim, activation=sigmoid)

    def call(self, input_features):
        x = self.hidden_layer(input_features)
        return self.output_layer(x)


class vanilla_decoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, input_dim):
        super(vanilla_decoder, self).__init__()
        self.hidden_layer = Dense(
            units=intermediate_dim, activation=relu, kernel_initializer="he_uniform"
        )
        self.output_layer = Dense(units=input_dim, activation=sigmoid)

    def call(self, code):
        x = self.hidden_layer(code)
        return self.output_layer(x)


class vanilla_autoencoder(tf.keras.Model):
    def __init__(self, intermediate_dim, input_dim):
        super(vanilla_autoencoder, self).__init__()
        self.vanilla_encoder = vanilla_encoder(intermediate_dim=intermediate_dim)
        self.vanilla_decoder = vanilla_decoder(
            intermediate_dim=intermediate_dim, input_dim=input_dim
        )

    def call(self, input_features):
        code = self.vanilla_encoder(input_features)
        decode = self.vanilla_decoder(code)
        return decode


class denoise_autoencoder(tf.keras.Model):
    def __init__(self, intermediate_dim, input_dim):
        super(denoise_autoencoder, self).__init__()
        pass

    def call(self, input_features):
        pass
