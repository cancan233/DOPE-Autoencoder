import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    MaxPool1D,
    UpSampling1D,
    BatchNormalization,
    Flatten,
)
from tensorflow.nn import (
    relu,
    sigmoid,
)


class vanilla_encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super(vanilla_encoder, self).__init__()
        self.hidden_layer = Dense(
            units=latent_dim, activation=relu, kernel_initializer="he_uniform"
        )
        self.output_layer = Dense(units=latent_dim, activation=sigmoid)

    def call(self, input_features):
        x = self.hidden_layer(input_features)
        return self.output_layer(x)


class vanilla_decoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, input_dim):
        super(vanilla_decoder, self).__init__()
        self.hidden_layer = Dense(
            units=latent_dim, activation=relu, kernel_initializer="he_uniform"
        )
        self.output_layer = Dense(units=input_dim, activation=sigmoid)

    def call(self, code):
        x = self.hidden_layer(code)
        return self.output_layer(x)


class vanilla_autoencoder(tf.keras.Model):
    def __init__(self, latent_dim, input_dim):
        super(vanilla_autoencoder, self).__init__()
        self.encoder = vanilla_encoder(latent_dim=latent_dim)
        self.decoder = vanilla_decoder(latent_dim=latent_dim, input_dim=input_dim)

    def call(self, input_features):
        code = self.encoder(input_features)
        decode = self.decoder(code)
        return decode


class denoise_autoencoder(tf.keras.Model):
    def __init__(self, latent_dim, input_dim):
        super(denoise_autoencoder, self).__init__()
        pass

    def call(self, input_features):
        pass


class convolutional_encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super(convolutional_encoder, self).__init__()
        self.hidden_layer = tf.keras.Sequential(
            [
                Conv1D(256, 5, 1, padding="same", activation="relu"),
                Conv1D(128, 5, 1, padding="same", activation="relu"),
                Conv1D(64, 5, 1, padding="same", activation="relu"),
            ]
        )
        self.output_layer = Dense(units=latent_dim, activation=sigmoid)

    def call(self, input_features):
        x = self.hidden_layer(input_features)
        return self.output_layer(x)


class convolutional_decoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, input_dim):
        super(convolutional_decoder, self).__init__()
        self.hidden_layer = tf.keras.Sequential(
            [
                Conv1D(64, 5, 1, padding="same", activation="relu"),
                UpSampling1D(),
                Conv1D(128, 5, 1, padding="same", activation="relu"),
                UpSampling1D(),
                Conv1D(256, 5, 1, padding="same", activation="relu"),
                UpSampling1D(),
            ]
        )
        self.output_layer = Conv1D(1, 5, 1, activation="sigmoid", padding="same")

    def call(self, code):
        x = self.hidden_layer(code)
        return self.output_layer(x)


class convolutional_autoencoder(tf.keras.Model):
    def __init__(self, latent_dim, input_dim):
        super(convolutional_autoencoder, self).__init__()
        self.encoder = convolutional_encoder(latent_dim=latent_dim)
        self.decoder = convolutional_decoder(latent_dim=latent_dim, input_dim=input_dim)

    def call(self, input_features):
        code = self.encoder(input_features)
        decode = self.decoder(code)
        return decode
