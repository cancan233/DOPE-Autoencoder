import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    MaxPool1D,
    UpSampling1D,
    BatchNormalization,
    Flatten,
    Reshape,
)
from tensorflow.nn import (
    relu,
    sigmoid,
)
import hyperparameters as hp


class vanilla_encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, intermediate_dim):
        super(vanilla_encoder, self).__init__()
        self.hidden_layer = Dense(
            units=intermediate_dim, activation=relu, kernel_initializer="he_uniform"
        )
        self.output_layer = Dense(units=latent_dim, activation=sigmoid)

    def call(self, input_features):
        x = self.hidden_layer(input_features)
        return self.output_layer(x)


class vanilla_decoder(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim, original_dim):
        super(vanilla_decoder, self).__init__()
        self.hidden_layer = Dense(
            units=intermediate_dim, activation=relu, kernel_initializer="he_uniform"
        )
        self.output_layer = Dense(units=original_dim, activation=sigmoid)

    def call(self, code):
        x = self.hidden_layer(code)
        return self.output_layer(x)


class vanilla_autoencoder(tf.keras.Model):
    def __init__(self, latent_dim, intermediate_dim, original_dim):
        super(vanilla_autoencoder, self).__init__()
        self.encoder = vanilla_encoder(latent_dim, intermediate_dim)
        self.decoder = vanilla_decoder(intermediate_dim, original_dim)

    def call(self, input_features):
        code = self.encoder(input_features)
        decode = self.decoder(code)
        return decode


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
        self.output_layer = tf.keras.Sequential(
            [Dense(units=latent_dim, activation=sigmoid)]
        )

    def call(self, input_features):
        x = self.hidden_layer(input_features)
        return self.output_layer(x)


class convolutional_decoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, original_dim):
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
    def __init__(self, latent_dim, original_dim):
        super(convolutional_autoencoder, self).__init__()
        self.encoder = convolutional_encoder(latent_dim=latent_dim)
        self.decoder = convolutional_decoder(
            latent_dim=latent_dim, original_dim=original_dim
        )

    def call(self, input_features):
        code = self.encoder(input_features)
        decode = self.decoder(code)
        return decode


class Sampling(tf.keras.layers.Layer):
    def __init__(self):
        super(Sampling, self).__init__()

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim, 1))
        output = z_mean + tf.math.multiply(tf.exp(0.5 * z_log_var), epsilon)
        return output


class variational_encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, intermediate_dim):
        super(variational_encoder, self).__init__()
        self.hidden_layer = Dense(
            units=intermediate_dim, activation=relu, kernel_initializer="he_uniform"
        )
        self.output_mean = Dense(units=latent_dim)
        self.output_var = Dense(units=latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.hidden_layer(inputs)
        # x = BatchNormalization(x)
        # x = LeakyReLU(x)
        z_mean = self.output_mean(x)
        z_var = self.output_var(x)
        z = self.sampling((z_mean, z_var))
        return z_mean, z_var, z


class variational_decoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, intermediate_dim, original_dim):
        super(variational_decoder, self).__init__()
        self.hidden_layer = Dense(
            units=intermediate_dim, activation=relu, kernel_initializer="he_uniform"
        )
        self.output_layer = Dense(units=original_dim, activation=sigmoid)

    def call(self, inputs):
        x = self.hidden_layer(inputs)
        # x = BatchNormalization(x)
        return self.output_layer(x)


class variational_autoencoder(tf.keras.Model):
    def __init__(self, original_dim, intermediate_dim, latent_dim):
        super(variational_autoencoder, self).__init__()
        self.original_dim = original_dim
        self.encoder = variational_encoder(latent_dim, intermediate_dim)
        self.decoder = variational_decoder(latent_dim, intermediate_dim, original_dim)

    def call(self, inputs):
        z_mean, z_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed
