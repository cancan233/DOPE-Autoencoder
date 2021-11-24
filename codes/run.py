import os
import argparse
import pandas as pd
import datetime
import tensorflow as tf
import numpy as np
import hyperparameters as hp
from models import vanilla_autoencoder, denoise_autoencoder

# diable all debugging logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="arguments parser for models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="vanilla_autoencoder",
        help="types of model to use, vanilla_autoencoder, ",
    )
    parser.add_argument(
        "--omics-data",
        default=".." + os.sep + "omics_data" + os.sep + "cnv_methyl_rnaseq.csv",
        help="omics data file name",
    )
    parser.add_argument("--biomed-data", default=None, help="biomed data file name")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="not save the checkpoints, logs and model. Used only for develop purpose",
    )
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help="path to model checkpoint file, should be similar to ./output/checkpoints/041721-201121/epoch19",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="skips training. evaluates the performance of the trained model.",
    )
    return parser.parse_args()


class CustomModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        super(CustomModelSaver, self).__init__()
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, logs=None):
        save_name = "epoch_{}".format(epoch)
        tf.keras.models.save_model(
            self.model, self.checkpoint_dir + os.sep + save_name, save_format="tf"
        )


def loss_fn(model, input_features):
    decode_error = tf.reduce_mean(
        tf.square(tf.subtract(model(input_features), input_features))
    )
    return decode_error


def train(loss_fn, model, optimizer, input_features, train_loss):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, input_features)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)


def main():
    time_now = datetime.datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    omics_data = pd.read_csv(ARGS.omics_data, index_col=0).T.astype("float32")
    (num_patients, num_features) = omics_data.shape
    tf.convert_to_tensor(omics_data)
    training_dataset = tf.data.Dataset.from_tensor_slices(omics_data)
    training_dataset = training_dataset.batch(hp.batch_size)
    training_dataset = training_dataset.shuffle(num_patients)
    training_dataset = training_dataset.prefetch(hp.batch_size * 4)

    if ARGS.model == "vanilla_autoencoder":
        model = vanilla_autoencoder(
            intermediate_dim=hp.intermediate_dim, input_dim=num_features
        )
    elif ARGS.model == "denoise_autoencoder":
        model = denoise_autoencoder(
            intermediate_dim=hp.intermediate_dim, input_dim=num_features
        )

    optimizer = tf.keras.optimizers.Adam(
        (
            tf.keras.optimizers.schedules.InverseTimeDecay(
                hp.learning_rate, decay_steps=1, decay_rate=5e-5
            )
        )
    )

    train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)

    writer = tf.summary.create_file_writer("tmp")
    with writer.as_default():
        with tf.summary.record_if(True):
            for epoch in range(hp.num_epochs):
                for step, batch_features in enumerate(training_dataset):
                    train(loss_fn, model, optimizer, batch_features, train_loss)
                    # loss_values = loss_fn(model, batch_features)
                tf.summary.scalar("loss", train_loss.result(), step=epoch)

                template = "Epoch {}, Loss {}"
                print(template.format(epoch + 1, train_loss.result()))
                train_loss.reset_states()


if __name__ == "__main__":
    ARGS = parse_args()
    main()
